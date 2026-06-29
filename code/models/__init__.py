# models/ — All vegetation index prediction models
#
# Directory structure:
#   mamba_common/   — Shared Mamba SSM core (MambaConfig, RMSNorm, pscan)
#   mst_mamba/      — Model A: MST_Mamba (depends on mamba_common)
#   wpo3d/          — Model B: WPO3D (self-contained)
#   restormer/      — Model C: Restormer (self-contained)
#   dhm/            — Model D: DHM (depends on mamba_common)
#   ifgnet/         — Model E: IFGNet (contains vegidx_formulas.py)
#   baseline/       — MST_Corrected original baseline (self-contained)
#
# All models share the same interface:
#   forward(y, input_mask) -> [bs, 32, 256, 256]
