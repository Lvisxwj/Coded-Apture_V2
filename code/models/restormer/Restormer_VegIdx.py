"""
Restormer_VegIdx.py — Model C: Restormer with MDTA for vegetation index prediction.

Architecture:
  - Multi-Dconv Head Transposed Attention (MDTA): channel-dim attention, O(C^2) not O(N^2)
  - Gated-Dconv Feed-Forward Network (GDFN)
  - Mask-guided V modulation via learned projection
  - U-Net encoder-decoder with skip connections

Input:  [bs, 256, 422]       — CASSI 2D measurement
Output: [bs, 32, 256, 256]   — 32 vegetation indices

Reference: Restormer (CVPR 2022) — adapted for CASSI vegetation index prediction.
"""

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
        # x: [B, C, H, W] -> [B, H, W, C] -> norm -> [B, C, H, W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ----------------------------------------------------------------
# MDTA — Multi-Dconv Head Transposed Attention
# ----------------------------------------------------------------

class MDTA(nn.Module):
    """
    Channel-dimension transposed attention.
    Q, K, V each: 1x1 Conv -> 3x3 DWConv.
    Attention on channel dim: [B, heads, C/heads, HW] x [B, heads, HW, C/heads].
    """
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, heads, C/h, C/h]
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class MaskMDTA(nn.Module):
    """MDTA with mask-guided V modulation."""
    def __init__(self, dim, num_heads, mask_ch=84, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Mask projection: 84 -> dim, sigmoid gate
        self.mask_proj = nn.Sequential(
            nn.Conv2d(mask_ch, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Mask gate on V before attention
        if mask is not None:
            if mask.shape[2] != h or mask.shape[3] != w:
                mask_resized = F.interpolate(mask, size=(h, w), mode="nearest")
            else:
                mask_resized = mask
            v = v * self.mask_proj(mask_resized)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


# ----------------------------------------------------------------
# GDFN — Gated-Dconv Feed-Forward Network
# ----------------------------------------------------------------

class GDFN(nn.Module):
    """
    Gated DWConv FFN.
    Split into two paths after DWConv: GELU(x1) * x2 -> project_out.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ----------------------------------------------------------------
# Transformer Block = LayerNorm + MaskMDTA + LayerNorm + GDFN
# ----------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66,
                 mask_ch=84, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MaskMDTA(dim, num_heads, mask_ch, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ----------------------------------------------------------------
# Restormer_VegIdx — Full model
# ----------------------------------------------------------------

class Restormer_VegIdx(nn.Module):
    """
    Restormer-based vegetation index predictor.

    Args:
        dim: base feature dimension
        num_heads: list of attention heads per stage (len = stage+1 for bottleneck)
        num_blocks: list of transformer blocks per stage (len = stage+1)
        ffn_expansion_factor: GDFN expansion ratio
        in_channels: input HSI channels (84 after initial_x)
        out_channels: output vegetation indices (32)
    """
    def __init__(self, dim=48, num_heads=None, num_blocks=None,
                 ffn_expansion_factor=2.66, in_channels=84, out_channels=32):
        super().__init__()

        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]

        # stage = len(num_blocks) - 1 (last entry is bottleneck)
        self.stage = len(num_blocks) - 1
        self.dim = dim

        # Embedding: 84 -> dim
        self.embedding = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(self.stage):
            blocks = nn.ModuleList([
                TransformerBlock(dim_stage, num_heads[i], ffn_expansion_factor, mask_ch=84)
                for _ in range(num_blocks[i])
            ])
            self.encoder_layers.append(nn.ModuleList([
                blocks,
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),   # feature downsample
                nn.Conv2d(84, 84, 4, 2, 1, bias=False),                     # mask downsample (keep 84ch)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            TransformerBlock(dim_stage, num_heads[-1], ffn_expansion_factor, mask_ch=84)
            for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(self.stage):
            blk_idx = self.stage - 1 - i
            blocks = nn.ModuleList([
                TransformerBlock(dim_stage // 2, num_heads[blk_idx],
                                 ffn_expansion_factor, mask_ch=84)
                for _ in range(num_blocks[blk_idx])
            ])
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),  # fusion
                blocks,
            ]))
            dim_stage //= 2

        # Output mapping: dim -> 32
        self.mapping = nn.Conv2d(self.dim, out_channels, 3, 1, 1, bias=False)

    def initial_x(self, y):
        """
        Convert 2D CASSI measurement to initial 3D estimate.
        Input:  y [bs, 256, 422]
        Output: x [bs, 84, 256, 256]
        """
        nC, step = 84, 2
        bs, row, col = y.shape
        x = torch.zeros(bs, nC, row, col - (nC - 1) * step,
                         device=y.device, dtype=y.dtype)
        for i in range(nC):
            x[:, i, :, :] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        return x

    def forward(self, y, input_mask=None):
        """
        Args:
            y: [bs, 256, 422] — 2D CASSI measurement
            input_mask: [bs, 84, 256, 256] — cropped CASSI mask
        Returns:
            [bs, 32, 256, 256] — 32 vegetation indices
        """
        x = self.initial_x(y)  # [bs, 84, 256, 256]
        mask = input_mask

        fea = self.lrelu(self.embedding(x))

        # Encoder
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

        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask)

        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask)

        return self.mapping(fea)
