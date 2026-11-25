"""Custom model/loss/assigner modules for Phase 1 & V2 ablations."""

from .hooks import attach_phase1_callbacks, build_model_with_switches
from .tc_gate import TCGateWrapper
from .tcad import BiTCStemWrapper, TCADWrapper, extract_tcad_gate_stats

__all__ = [
    "BiTCStemWrapper",
    "TCADWrapper",
    "TCGateWrapper",
    "attach_phase1_callbacks",
    "build_model_with_switches",
    "extract_tcad_gate_stats",
]
