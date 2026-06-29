"""
IFGNet.py — Model E: Index-Formula-Guided Network.

Dual-branch architecture:
  Branch 1 (spectral): lightweight U-Net -> 84-band spectral reconstruction -> formula indices
  Branch 2 (index):    deeper U-Net -> direct 32-index regression
  Fusion: learnable per-channel alpha blending

Uses vegetation index formulas as physical priors (via vegidx_formulas.py).

Input:  [bs, 256, 422]       — CASSI 2D measurement
Output: I_final  [bs, 32, 256, 256] — fused vegetation indices
        I_formula [bs, 32, 256, 256] — formula-computed indices
        R_hat    [bs, 84, 256, 256] — reconstructed HSI (for auxiliary loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vegidx_formulas import select_bands, compute_all_indices
except ImportError:
    from models.ifgnet.vegidx_formulas import select_bands, compute_all_indices


# ----------------------------------------------------------------
# Building blocks (lightweight Restormer-style)
# ----------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MDTA_Lite(nn.Module):
    """Simplified MDTA with optional mask gating on V."""
    def __init__(self, dim, num_heads, mask_ch=84, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.mask_proj = nn.Sequential(nn.Conv2d(mask_ch, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if mask is not None:
            if mask.shape[2] != h or mask.shape[3] != w:
                mask = F.interpolate(mask, size=(h, w), mode="nearest")
            v = v * self.mask_proj(mask)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn.softmax(dim=-1) @ v).reshape(b, c, h, w)
        return self.project_out(out)


class GDFN_Lite(nn.Module):
    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TBlock(nn.Module):
    """Transformer block = LN + MDTA + LN + GDFN."""
    def __init__(self, dim, num_heads, mask_ch=84, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA_Lite(dim, num_heads, mask_ch)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN_Lite(dim, ffn_expansion)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ----------------------------------------------------------------
# SimpleUNet — generic U-Net backbone
# ----------------------------------------------------------------

class SimpleUNet(nn.Module):
    def __init__(self, in_ch, out_ch, dim, stage, num_blocks, mask_ch=84):
        super().__init__()
        self.stage = stage
        self.embedding = nn.Conv2d(in_ch, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder = nn.ModuleList()
        dim_s = dim
        for i in range(stage):
            heads = max(1, dim_s // 32)
            blks = nn.ModuleList([TBlock(dim_s, heads, mask_ch) for _ in range(num_blocks[i])])
            self.encoder.append(nn.ModuleList([
                blks,
                nn.Conv2d(dim_s, dim_s * 2, 4, 2, 1, bias=False),
                nn.Conv2d(mask_ch, mask_ch, 4, 2, 1, bias=False),
            ]))
            dim_s *= 2

        # Bottleneck
        heads = max(1, dim_s // 32)
        self.bottleneck = nn.ModuleList([
            TBlock(dim_s, heads, mask_ch) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(stage):
            blk_idx = stage - 1 - i
            heads = max(1, (dim_s // 2) // 32)
            blks = nn.ModuleList([
                TBlock(dim_s // 2, heads, mask_ch) for _ in range(num_blocks[blk_idx])
            ])
            self.decoder.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_s, dim_s // 2, 2, 2),
                nn.Conv2d(dim_s, dim_s // 2, 1, 1, bias=False),
                blks,
            ]))
            dim_s //= 2

        self.mapping = nn.Conv2d(dim, out_ch, 3, 1, 1, bias=False)

    def forward(self, x, mask=None):
        fea = self.lrelu(self.embedding(x))
        enc_feats, enc_masks = [], []

        for blks, f_down, m_down in self.encoder:
            for blk in blks:
                fea = blk(fea, mask)
            enc_masks.append(mask)
            enc_feats.append(fea)
            fea = f_down(fea)
            if mask is not None:
                mask = m_down(mask)

        for blk in self.bottleneck:
            fea = blk(fea, mask)

        for i, (f_up, fusion, blks) in enumerate(self.decoder):
            fea = f_up(fea)
            fea = fusion(torch.cat([fea, enc_feats[self.stage - 1 - i]], dim=1))
            mask = enc_masks[self.stage - 1 - i]
            for blk in blks:
                fea = blk(fea, mask)

        return self.mapping(fea)


# ----------------------------------------------------------------
# IFGNet — Full model
# ----------------------------------------------------------------

class IFGNet(nn.Module):
    """
    Index-Formula-Guided Network.

    Args:
        dim_spec: feature dim for spectral reconstruction branch
        dim_idx: feature dim for index regression branch
        stage: U-Net stages
        num_blocks_spec: blocks per stage for spectral branch
        num_blocks_idx: blocks per stage for index branch
    """
    def __init__(self, dim_spec=32, dim_idx=48, stage=3,
                 num_blocks_spec=None, num_blocks_idx=None):
        super().__init__()

        if num_blocks_spec is None:
            num_blocks_spec = [1, 1, 1]
        if num_blocks_idx is None:
            num_blocks_idx = [2, 2, 2]

        # Branch 1: spectral reconstruction (lightweight)
        self.spectral_branch = SimpleUNet(
            in_ch=84, out_ch=84, dim=dim_spec,
            stage=stage, num_blocks=num_blocks_spec,
        )

        # Branch 2: index regression (deeper)
        self.index_branch = SimpleUNet(
            in_ch=84, out_ch=32, dim=dim_idx,
            stage=stage, num_blocks=num_blocks_idx,
        )

        # Learnable per-channel fusion weight
        self.alpha = nn.Parameter(torch.ones(1, 32, 1, 1) * 0.5)

    def initial_x(self, y):
        nC, step = 84, 2
        bs, row, col = y.shape
        out_w = col - (nC - 1) * step
        starts = torch.arange(nC, device=y.device) * step
        idx = starts.unsqueeze(1) + torch.arange(out_w, device=y.device)
        return y[:, :, idx].permute(0, 2, 1, 3)

    def forward(self, y, input_mask=None):
        """
        Returns:
            I_final:   [bs, 32, 256, 256] — fused output (primary)
            I_formula: [bs, 32, 256, 256] — formula-computed indices
            R_hat:     [bs, 84, 256, 256] — reconstructed HSI
        """
        x = self.initial_x(y)  # [bs, 84, 256, 256]

        # Branch 1: spectral reconstruction -> formula indices
        R_hat = self.spectral_branch(x, input_mask)           # [bs, 84, 256, 256]
        R_hat_clamped = torch.clamp(R_hat, min=0)             # reflectance >= 0
        R_35 = select_bands(R_hat_clamped)                     # [bs, 35, 256, 256]
        I_formula = compute_all_indices(R_35)                  # [bs, 32, 256, 256]

        # Branch 2: direct regression
        I_regress = self.index_branch(x, input_mask)           # [bs, 32, 256, 256]

        # Fusion
        alpha = torch.sigmoid(self.alpha)
        I_final = alpha * I_regress + (1 - alpha) * I_formula

        return I_final, I_formula, R_hat
