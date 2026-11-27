"""Custom model/loss/assigner/KD modules for Phase 1, V2, V3, and Phase 3 ablations."""

from .hooks import attach_phase1_callbacks, build_model_with_switches
from .kd_yolo26 import (
    NeckChannelAdapters,
    YOLO26DistillCriterion,
    YOLO26KDConfig,
    attach_yolo26_kd_hook,
)
from .paired_rgbt_data import PairedRGBIRDataset, attach_paired_rgbt_dataloader_hook
from .tc_gate import TCGateWrapper
from .tcad import BiTCStemWrapper, TCADWrapper, extract_tcad_gate_stats
from .v3_arch import BorderSafeBiTCStemWrapper, BorderSafeCAAResWrapper, HaarResWrapper
from .v3_losses import V3ODCSConfig, attach_v3_odcs_hook

__all__ = [
    "BiTCStemWrapper",
    "BorderSafeBiTCStemWrapper",
    "BorderSafeCAAResWrapper",
    "HaarResWrapper",
    "NeckChannelAdapters",
    "TCADWrapper",
    "TCGateWrapper",
    "V3ODCSConfig",
    "YOLO26DistillCriterion",
    "YOLO26KDConfig",
    "attach_phase1_callbacks",
    "attach_v3_odcs_hook",
    "attach_yolo26_kd_hook",
    "build_model_with_switches",
    "extract_tcad_gate_stats",
]
