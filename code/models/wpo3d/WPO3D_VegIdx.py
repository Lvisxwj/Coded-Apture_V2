"""
WPO3D_VegIdx.py — Model B: 3D Wave Propagation Operator for vegetation index prediction.

Physics-informed spectral-spatial processing via damped wave equation in frequency domain.
Ported from CASSI/wpo3d.py with channel adaptations (84 in, 32 out).

Input:  [bs, 256, 422]       — CASSI 2D measurement
Output: [bs, 32, 256, 256]   — 32 vegetation indices
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm on channels-first [B, C, H, W] tensors."""
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
# MaskGateA — soft mask gating (ported from CASSI/mask_ops.py)
# ----------------------------------------------------------------

class MaskGateA(nn.Module):
    """Mask soft gate: project 84-ch mask to current dim, sigmoid gate."""
    def __init__(self, dim, mask_ch=84):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(mask_ch, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, mask):
        return self.proj(mask)


# ----------------------------------------------------------------
# WPO3D — 3D Wave Propagation Operator
# ----------------------------------------------------------------

class WPO3D(nn.Module):
    """
    3D Wave Propagation Operator.

    Learnable physics parameters:
        alpha — damping coefficient
        vs    — spatial wave speed
        vl    — spectral (channel) wave speed
        t     — propagation time

    Operates in 3D Fourier domain on (C, H, W).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.vs = nn.Parameter(torch.tensor(1.0))
        self.vl = nn.Parameter(torch.tensor(0.5))
        self.t = nn.Parameter(torch.tensor(1.0))

        # Semantic encode / decode
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
        )

    def forward(self, x, mask_gate=None):
        """
        Args:
            x: [B, C, H, W]
            mask_gate: [B, C, H, W] sigmoid gate (optional)
        """
        u0 = self.phi(x)
        if mask_gate is not None:
            u0 = u0 * mask_gate

        alpha = F.softplus(self.alpha)
        vs = F.softplus(self.vs)
        vl = F.softplus(self.vl)
        t = F.softplus(self.t)

        B, C, H, W = u0.shape

        # 3D rFFT over (C, H, W)
        u0_fft = torch.fft.rfftn(u0, dim=(-3, -2, -1))

        # Frequency grids
        freq_c = torch.fft.fftfreq(C, device=x.device).view(1, C, 1, 1)
        freq_h = torch.fft.fftfreq(H, device=x.device).view(1, 1, H, 1)
        freq_w = torch.fft.rfftfreq(W, device=x.device).view(1, 1, 1, -1)

        # Dispersion relation
        omega_sq = (vs * 2 * math.pi) ** 2 * (freq_h ** 2 + freq_w ** 2) \
                   + (vl * 2 * math.pi) ** 2 * freq_c ** 2
        omega_d_sq = omega_sq - (alpha / 2) ** 2

        # Under-damped vs over-damped branches
        is_under = omega_d_sq > 0
        omega_d = torch.sqrt(torch.clamp(omega_d_sq.abs(), min=1e-12))

        decay = torch.exp(-alpha * t / 2)

        cos_term = torch.where(is_under, torch.cos(omega_d * t), torch.cosh(omega_d * t))
        sin_coeff = alpha / (2 * omega_d + 1e-12)
        sin_term = torch.where(is_under, torch.sin(omega_d * t), torch.sinh(omega_d * t))

        kernel = decay * (cos_term + sin_coeff * sin_term)

        out_fft = u0_fft * kernel
        out = torch.fft.irfftn(out_fft, s=(C, H, W), dim=(-3, -2, -1))

        return self.psi(out)


# ----------------------------------------------------------------
# WPO3D Block = Norm + WPO3D + MaskGate + FFN
# ----------------------------------------------------------------

class WPO3D_Block(nn.Module):
    def __init__(self, dim, mask_ch=84):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo = WPO3D(dim)
        self.mask_gate = MaskGateA(dim, mask_ch)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FFN(dim)

    def forward(self, x, mask):
        gate = self.mask_gate(mask)
        h = self.wpo(self.norm1(x), mask_gate=gate)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# ----------------------------------------------------------------
# WPO3D_VegIdx — Full model
# ----------------------------------------------------------------

class WPO3D_VegIdx(nn.Module):
    """
    WPO3D vegetation index predictor.

    Args:
        dim: base feature dimension
        stage: number of encoder/decoder stages
        num_blocks: list of WPO3D blocks per stage (len = stage, last used for bottleneck)
    """
    def __init__(self, dim=64, stage=3, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]

        self.dim = dim
        self.stage = stage

        self.embedding = nn.Conv2d(84, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([WPO3D_Block(dim_stage) for _ in range(num_blocks[i])])
            self.encoder_layers.append(nn.ModuleList([
                blocks,
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(84, 84, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            WPO3D_Block(dim_stage) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            blk_count = num_blocks[stage - 1 - i]
            blocks = nn.ModuleList([
                WPO3D_Block(dim_stage // 2) for _ in range(blk_count)
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
