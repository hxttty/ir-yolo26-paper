"""Custom model/loss/assigner/KD modules for Phase 1, V2, V3, and Phase 3 ablations."""

from pathlib import Path
from typing import Any

from configs.experiments import ExperimentConfig
from . import hooks
from .kd_yolo26 import (
    NeckChannelAdapters,
    YOLO26DistillCriterion,
    YOLO26KDConfig,
    attach_yolo26_kd_hook,
)
from .tc_gate import TCGateWrapper
from .tcad import BiTCStemWrapper, TCADWrapper, extract_tcad_gate_stats
from .v3_arch import BorderSafeBiTCStemWrapper, BorderSafeCAAResWrapper, HaarResWrapper
from .v3_losses import V3ODCSConfig, attach_v3_odcs_hook


def __getattr__(name: str) -> Any:
    # Standalone KD losses do not require the Ultralytics dataloader dependency.
    if name in ("PairedRGBIRDataset", "attach_paired_rgbt_dataloader_hook"):
        from . import paired_rgbt_data

        return getattr(paired_rgbt_data, name)
    raise AttributeError(name)


def _configure_track_b(model: Any, experiment: ExperimentConfig) -> None:
    """Apply experiment options to the configuration shared by the registered KD callbacks."""
    if experiment.use_kd:
        cfg = getattr(model, "_yolo26_kd_config", None)
        if cfg is not None:
            cfg.cm_illum_mode = experiment.kd_cm_illum_mode
            cfg.night_q50_thresh = experiment.kd_night_q50_thresh
            cfg.night_macro_std_thresh = experiment.kd_night_macro_std_thresh
            cfg.cm_feat_quality_gate = experiment.kd_cm_feat_quality_gate
            cfg.cm_cls_fg_norm = experiment.kd_cm_cls_fg_norm


def attach_phase1_callbacks(model: Any, cfg: ExperimentConfig) -> None:
    hooks.attach_phase1_callbacks(model, cfg)
    _configure_track_b(model, cfg)


def build_model_with_switches(cfg: ExperimentConfig, repo_root: Path) -> Any:
    model = hooks.build_model_with_switches(cfg, repo_root)
    _configure_track_b(model, cfg)
    return model


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
