"""
MST_Mamba_VegIdx.py — Model A: MST_Mamba adapted for vegetation index prediction.

Fixes from original MST_Mamba.py:
  1. Input channels: 28 -> 84 (via initial_x)
  2. Output channels: 28 -> 32 (vegetation indices)
  3. Mask channels: kept at 84 throughout (not doubled per stage)
  4. No residual shortcut (84 != 32)
  5. Added mask_proj to bridge 84-ch mask to current stage dim

Input:  [bs, 256, 422]       — CASSI 2D measurement
Output: [bs, 32, 256, 256]   — 32 vegetation indices
"""

import torch
import torch.nn as nn

try:
    from .MST_Mamba import (
        MaskAwareMambaMixer,
        FeedForward,
        PreNorm,
        validate_stage_reductions,
    )
except ImportError:
    from models.mst_mamba.MST_Mamba import (
        MaskAwareMambaMixer,
        FeedForward,
        PreNorm,
        validate_stage_reductions,
    )


# ----------------------------------------------------------------
# MSMB_VegIdx — Mamba block with 84-ch mask projection
# ----------------------------------------------------------------

class MSMB_VegIdx(nn.Module):
    """MSMB adapted for 84-channel mask input."""
    def __init__(self, dim, mask_ch=84, num_blocks=2, **mamba_kwargs):
        super().__init__()
        self.mask_proj = nn.Conv2d(mask_ch, dim, 1, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MaskAwareMambaMixer(dim=dim, **mamba_kwargs),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x, mask):
        """
        x:    [B, dim, H, W]
        mask: [B, 84, H', W'] — may differ in spatial size from x
        """
        # Project mask channels: 84 -> dim
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask_resized = nn.functional.interpolate(
                mask, size=(x.shape[2], x.shape[3]), mode="nearest"
            )
        else:
            mask_resized = mask
        mask_proj = self.mask_proj(mask_resized)

        # Process blocks (channels-last for MaskAwareMambaMixer)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, dim]
        for mixer, ff in self.blocks:
            x = mixer(x, mask=mask_proj) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------
# MST_Mamba_VegIdx — Full model
# ----------------------------------------------------------------

class MST_Mamba_VegIdx(nn.Module):
    """
    MST_Mamba adapted for vegetation index prediction.

    Args:
        dim: base feature dimension (default 64)
        stage: U-Net stages (default 3)
        num_blocks: transformer blocks per stage
        mask_fusion: mask fusion mode for MaskAwareMambaMixer
    """
    def __init__(
        self,
        dim=64,
        stage=3,
        num_blocks=(2, 2, 2),
        d_state=16,
        expand_factor=2,
        d_conv=4,
        dt_rank="auto",
        pscan_parallel=True,
        mask_fusion="input_mul",
        stage_reductions=None,
    ):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.stage_reductions = validate_stage_reductions(stage, stage_reductions)

        # Input: 84 channels (after initial_x)
        self.embedding = nn.Conv2d(84, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Common Mamba kwargs
        mamba_kw = dict(
            d_state=d_state, expand_factor=expand_factor, d_conv=d_conv,
            dt_rank=dt_rank, pscan_parallel=pscan_parallel, mask_fusion=mask_fusion,
        )

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSMB_VegIdx(dim_stage, mask_ch=84, num_blocks=num_blocks[i],
                            spatial_reduction=self.stage_reductions[i], **mamba_kw),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(84, 84, 4, 2, 1, bias=False),  # mask stays 84ch
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSMB_VegIdx(
            dim_stage, mask_ch=84, num_blocks=num_blocks[-1],
            spatial_reduction=self.stage_reductions[stage], **mamba_kw
        )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSMB_VegIdx(dim_stage // 2, mask_ch=84,
                            num_blocks=num_blocks[stage - 1 - i],
                            spatial_reduction=self.stage_reductions[stage + 1 + i],
                            **mamba_kw),
            ]))
            dim_stage //= 2

        # Output: 32 vegetation indices (no residual)
        self.mapping = nn.Conv2d(self.dim, 32, 3, 1, 1, bias=False)

    def initial_x(self, y):
        nC, step = 84, 2
        bs, row, col = y.shape
        out_w = col - (nC - 1) * step
        starts = torch.arange(nC, device=y.device) * step
        idx = starts.unsqueeze(1) + torch.arange(out_w, device=y.device)
        return y[:, :, idx].permute(0, 2, 1, 3)

    def forward(self, y, input_mask=None):
        """
        Args:
            y: [bs, 256, 422] — CASSI 2D measurement
            input_mask: [bs, 84, 256, 256] — cropped CASSI mask
        Returns:
            [bs, 32, 256, 256] — 32 vegetation indices
        """
        x = self.initial_x(y)
        mask = input_mask

        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks = []
        for msmb, fea_down, mask_down in self.encoder_layers:
            fea = msmb(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = fea_down(fea)
            mask = mask_down(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask)

        # Decoder
        for i, (fea_up, fusion, msmb) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = msmb(fea, mask)

        return self.mapping(fea)
