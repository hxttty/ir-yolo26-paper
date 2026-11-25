"""Custom model/loss/assigner modules for Phase 1, V2, and V3 ablations."""

from .hooks import attach_phase1_callbacks, build_model_with_switches
from .tc_gate import TCGateWrapper
from .tcad import BiTCStemWrapper, TCADWrapper, extract_tcad_gate_stats
from .v3_arch import BorderSafeBiTCStemWrapper, BorderSafeCAAResWrapper, HaarResWrapper
from .v3_losses import V3ODCSConfig, attach_v3_odcs_hook

__all__ = [
    "BiTCStemWrapper",
    "BorderSafeBiTCStemWrapper",
    "BorderSafeCAAResWrapper",
    "HaarResWrapper",
    "TCADWrapper",
    "TCGateWrapper",
    "V3ODCSConfig",
    "attach_phase1_callbacks",
    "attach_v3_odcs_hook",
    "build_model_with_switches",
    "extract_tcad_gate_stats",
]
