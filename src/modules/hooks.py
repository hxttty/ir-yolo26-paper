"""Model builder and Phase 1 switch dispatcher.

Keeps baseline training completely clean and prevents accidentally running
unimplemented Phase 1 flags before `inspect_yolo26.py` verifies exact Colab
Ultralytics internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.experiments import ExperimentConfig


def build_model_with_switches(cfg: ExperimentConfig, repo_root: Path) -> Any:
    """Instantiate Ultralytics model according to `ExperimentConfig` switches."""
    from ultralytics import YOLO

    active_phase1 = [
        name
        for name, enabled in (
            ("use_mod_a (Pruned P5 PAN + Light DWConv P2-P4)", cfg.use_mod_a),
            ("use_mod_b (Zero-Init Bidirectional TCAD)", cfg.use_mod_b),
            ("use_mod_c1 (Scale-Gated NWD Assigner)", cfg.use_mod_c1),
            ("use_mod_c2 (Scale-Gated NWD BboxLoss)", cfg.use_mod_c2),
        )
        if enabled
    ]
    if active_phase1:
        raise NotImplementedError(
            "Phase 1 custom switches requested before source audit patch-in:\n"
            f"  Active flags: {', '.join(active_phase1)}\n"
            "Run Phase 0 `python -m src.inspect_yolo26` on Colab first so we patch "
            "the exact installed Ultralytics version cleanly without mismatches."
        )

    model_spec = cfg.model_cfg
    local_candidate = repo_root / model_spec
    if local_candidate.exists():
        model_spec = str(local_candidate)

    print(f"[MODEL] Loading model spec: {model_spec}")
    model = YOLO(model_spec)

    if cfg.pretrained_weights:
        print(f"[MODEL] Applying partial pretrained weight transfer <- {cfg.pretrained_weights}")
        model = model.load(cfg.pretrained_weights)

    return model
