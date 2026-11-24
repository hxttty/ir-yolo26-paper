"""Model builder and Phase 1 switch dispatcher (Mod A, B, C1, C2).

All patches are verified against installed `ultralytics 8.4.218`:
- Mod A (`use_mod_a`): Uses `configs/models/yolo26n_p2p4.yaml` (pruned P5 PAN + Head,
  compressed 32ch P2 + DWConv) + partial weight transfer from `yolo26n.pt`.
- Mod B (`use_mod_b`): Wraps `trainer.setup_model` inside `on_pretrain_routine_start`
  so `TCADWrapper` is injected IMMEDIATELY after the final `nc`-matched model + COCO
  pretrained weights are built, and BEFORE `.to(device)`, `ModelEMA`, and `build_optimizer()`.
- Mod C1/C2 (`use_mod_c1`, `use_mod_c2`): Hooks `on_train_start` to patch both `one2many`
  and `one2one` branches of `E2ELoss` with Scale-Gated NWD (training-only, zero inference cost).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.experiments import ExperimentConfig
from src.modules.nwd_e2e import attach_nwd_e2e_hook
from src.modules.tcad import inject_tcad_into_model


def attach_phase1_callbacks(model: Any, cfg: ExperimentConfig) -> None:
    """Attach Mod B and Mod C1/C2 callbacks onto an Ultralytics YOLO instance (fresh or resumed)."""
    if cfg.use_mod_b:
        def _on_pretrain_routine_start(trainer: Any) -> None:
            orig_setup_model = trainer.setup_model

            def _wrapped_setup_model() -> Any:
                ckpt = orig_setup_model()
                inject_tcad_into_model(trainer.model, stages=cfg.mod_b_stages)
                return ckpt

            trainer.setup_model = _wrapped_setup_model

        model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
        print(f"[MOD B] Registered setup_model wrapper for stages {cfg.mod_b_stages}.")

    if cfg.use_mod_c1 or cfg.use_mod_c2:
        attach_nwd_e2e_hook(
            model,
            use_c1=cfg.use_mod_c1,
            use_c2=cfg.use_mod_c2,
            area_thresh=cfg.nwd_area_thresh,
            tau=cfg.nwd_tau,
            w_max=cfg.nwd_w_max,
            c=cfg.nwd_constant_c,
        )


def build_model_with_switches(cfg: ExperimentConfig, repo_root: Path) -> Any:
    """Instantiate Ultralytics model and register enabled Phase 1 switches."""
    from ultralytics import YOLO

    model_spec = cfg.model_cfg
    local_candidate = repo_root / model_spec
    if local_candidate.exists():
        model_spec = str(local_candidate)

    print(
        f"[MODEL] Loading {model_spec} | "
        f"ModA={cfg.use_mod_a} ModB={cfg.use_mod_b} ModC1={cfg.use_mod_c1} ModC2={cfg.use_mod_c2}"
    )
    model = YOLO(model_spec)

    if cfg.pretrained_weights:
        print(f"[MODEL] Applying partial pretrained weight transfer <- {cfg.pretrained_weights}")
        model = model.load(cfg.pretrained_weights)

    attach_phase1_callbacks(model, cfg)
    return model
