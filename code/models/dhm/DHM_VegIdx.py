"""
DHM_VegIdx.py — Model D: Dual Hyperspectral Mamba for vegetation index prediction.

Architecture:
  - GHSB (Global Hyperspectral S4 Block): full-image SSM scan
  - LHSB (Local Hyperspectral S4 Block): window-partitioned SSM scan
  - DHSB: fuses Global + Local branches with mask gating
  - U-Net encoder-decoder with skip connections

Addresses Mamba's local context forgetting via dual-branch design.

Input:  [bs, 256, 422]       — CASSI 2D measurement
Output: [bs, 32, 256, 256]   — 32 vegetation indices

Reference: DHM (Dual Hyperspectral Mamba) — adapted for CASSI.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..mamba_common import MambaConfig, RMSNorm, pscan
except ImportError:
    from models.mamba_common import MambaConfig, RMSNorm, pscan


# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------
# SSM Core — shared by GHSB and LHSB
# ----------------------------------------------------------------

class SSMCore(nn.Module):
    """Minimal Mamba SSM without mask fusion (mask handled at DHSB level)."""
    def __init__(self, dim, d_state=16, expand_factor=2, d_conv=4, dt_rank="auto"):
        super().__init__()
        config = MambaConfig(
            d_model=dim, d_state=d_state, expand_factor=expand_factor,
            d_conv=d_conv, dt_rank=dt_rank,
        )
        self.config = config
        self.norm = RMSNorm(dim)

        self.in_proj = nn.Linear(dim, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(
            config.d_inner, config.d_inner, kernel_size=config.d_conv,
            padding=config.d_conv - 1, bias=config.conv_bias, groups=config.d_inner,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.out_proj = nn.Linear(config.d_inner, dim, bias=config.bias)

        # Parameter initialization
        dt_init_std = (config.dt_rank ** -0.5) * config.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(config.d_inner)
                * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            ).clamp(min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

    def forward(self, seq):
        """
        Args:
            seq: [B, L, dim]
        Returns:
            [B, L, dim]
        """
        seq = self.norm(seq)
        xz = self.in_proj(seq)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq.size(1)]
        x_branch = x_branch.transpose(1, 2).contiguous()
        x_branch = F.silu(x_branch)

        y = self._ssm(x_branch)
        z = F.silu(z_branch)
        return self.out_proj(y * z)

    def _ssm(self, x_ed):
        dtype = x_ed.dtype
        A = -torch.exp(self.A_log.to(dtype))
        D_ = self.D.to(dtype)

        deltaBC = self.x_proj(x_ed)
        delta, B_, C_ = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        delta = (self.dt_proj.weight.to(dtype) @ delta.transpose(1, 2)).transpose(1, 2).contiguous()
        delta = F.softplus(delta + self.dt_proj.bias.to(dtype))

        if self.config.pscan:
            deltaA = torch.exp(delta.unsqueeze(-1) * A)
            deltaB = delta.unsqueeze(-1) * B_.unsqueeze(2)
            BX = deltaB * x_ed.unsqueeze(-1)
            hs = pscan(deltaA, BX)
            y = (hs @ C_.unsqueeze(-1)).squeeze(-1)
        else:
            bsz, seq_len, ed = delta.shape
            n = self.config.d_state
            h = torch.zeros(bsz, ed, n, device=x_ed.device, dtype=dtype)
            y_list = []
            for t in range(seq_len):
                deltaA_t = torch.exp(delta[:, t].unsqueeze(-1) * A)
                deltaB_t = delta[:, t].unsqueeze(-1) * B_[:, t].unsqueeze(1)
                BX_t = deltaB_t * x_ed[:, t].unsqueeze(-1)
                h = deltaA_t * h + BX_t
                y_list.append((h @ C_[:, t].unsqueeze(-1)).squeeze(-1))
            y = torch.stack(y_list, dim=1)

        return y + D_ * x_ed


# ----------------------------------------------------------------
# GHSB — Global Hyperspectral S4 Block
# ----------------------------------------------------------------

class GHSB(nn.Module):
    """Full-image SSM: flatten (H,W) -> sequence of length H*W."""
    def __init__(self, dim, d_state=16, expand_factor=2, d_conv=4):
        super().__init__()
        self.ssm = SSMCore(dim, d_state, expand_factor, d_conv)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x):
        """x: [B, C, H, W]"""
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)    # [B, H*W, C]
        out = self.ssm(seq)                    # [B, H*W, C]
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out + self.pos_emb(x)


# ----------------------------------------------------------------
# LHSB — Local Hyperspectral S4 Block
# ----------------------------------------------------------------

class LHSB(nn.Module):
    """Window-partitioned SSM: each window scanned independently."""
    def __init__(self, dim, window_size=8, d_state=16, expand_factor=2, d_conv=4):
        super().__init__()
        self.window_size = window_size
        self.ssm = SSMCore(dim, d_state, expand_factor, d_conv)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x):
        """x: [B, C, H, W]"""
        b, c, h, w = x.shape
        ws = self.window_size

        # Pad if not divisible
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_pad = x

        _, _, hp, wp = x_pad.shape
        nH, nW = hp // ws, wp // ws

        # Window partition: [B, C, nH, ws, nW, ws] -> [B*nH*nW, ws*ws, C]
        x_win = x_pad.reshape(b, c, nH, ws, nW, ws)
        x_win = x_win.permute(0, 2, 4, 1, 3, 5).reshape(b * nH * nW, c, ws, ws)
        seq = x_win.flatten(2).transpose(1, 2)  # [B*n, ws*ws, C]

        out = self.ssm(seq)  # [B*n, ws*ws, C]
        out = out.transpose(1, 2).reshape(b * nH * nW, c, ws, ws)

        # Reverse window: [B*nH*nW, C, ws, ws] -> [B, C, H, W]
        out = out.reshape(b, nH, nW, c, ws, ws)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, hp, wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out + self.pos_emb(x)


# ----------------------------------------------------------------
# DHSB — Dual Hyperspectral S4 Block
# ----------------------------------------------------------------

class DHSB(nn.Module):
    """Dual-branch: Global + Local SSM with mask gating and fusion."""
    def __init__(self, dim, mask_ch=84, window_size=8,
                 d_state=16, expand_factor=2, d_conv=4):
        super().__init__()
        self.global_branch = GHSB(dim, d_state, expand_factor, d_conv)
        self.local_branch = LHSB(dim, window_size, d_state, expand_factor, d_conv)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(mask_ch, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fusion = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = LayerNorm2d(dim)
        self.ffn = FFN(dim)

    def forward(self, x, mask):
        """
        x:    [B, dim, H, W]
        mask: [B, 84, H', W']
        """
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode="nearest")
        mask_gate = self.mask_proj(mask)
        x_masked = x * mask_gate

        g = self.global_branch(x_masked)
        l = self.local_branch(x_masked)

        x = x + self.fusion(torch.cat([g, l], dim=1))
        x = x + self.ffn(self.norm(x))
        return x


# ----------------------------------------------------------------
# DHM_VegIdx — Full model
# ----------------------------------------------------------------

class DHM_VegIdx(nn.Module):
    """
    Dual Hyperspectral Mamba for vegetation index prediction.

    Args:
        dim: base feature dimension
        stage: U-Net stages
        num_blocks: DHSB blocks per stage
        window_size: local SSM window size
    """
    def __init__(self, dim=64, stage=3, num_blocks=None, window_size=8,
                 d_state=16, expand_factor=2, d_conv=4):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]

        self.dim = dim
        self.stage = stage

        self.embedding = nn.Conv2d(84, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        ssm_kw = dict(d_state=d_state, expand_factor=expand_factor, d_conv=d_conv)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([
                DHSB(dim_stage, mask_ch=84, window_size=window_size, **ssm_kw)
                for _ in range(num_blocks[i])
            ])
            self.encoder_layers.append(nn.ModuleList([
                blocks,
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(84, 84, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            DHSB(dim_stage, mask_ch=84, window_size=window_size, **ssm_kw)
            for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            blk_count = num_blocks[stage - 1 - i]
            blocks = nn.ModuleList([
                DHSB(dim_stage // 2, mask_ch=84, window_size=window_size, **ssm_kw)
                for _ in range(blk_count)
            ])
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                blocks,
            ]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, 32, 3, 1, 1, bias=False)

    def initial_x(self, y):
        nC, step = 84, 2
        bs, row, col = y.shape
        x = torch.zeros(bs, nC, row, col - (nC - 1) * step,
                         device=y.device, dtype=y.dtype)
        for i in range(nC):
            x[:, i, :, :] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        return x

    def forward(self, y, input_mask=None):
        x = self.initial_x(y)
        mask = input_mask
        fea = self.lrelu(self.embedding(x))

        fea_encoder = []
        masks = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = fea_down(fea)
            if mask is not None:
                mask = mask_down(mask)

        for blk in self.bottleneck:
            fea = blk(fea, mask)

        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask)

        return self.mapping(fea)
