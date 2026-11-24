"""Custom model/loss/assigner modules for Phase 1 ablations (Mod A, B, C1, C2)."""

from .hooks import attach_phase1_callbacks, build_model_with_switches
from .tcad import TCADWrapper, extract_tcad_gate_stats

__all__ = [
    "TCADWrapper",
    "attach_phase1_callbacks",
    "build_model_with_switches",
    "extract_tcad_gate_stats",
]
