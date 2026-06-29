# Shared Mamba SSM core — used by mst_mamba (Model A) and dhm (Model D)
from .mamba_core import MambaConfig, RMSNorm
from .pscan import pscan

__all__ = ["MambaConfig", "RMSNorm", "pscan"]
