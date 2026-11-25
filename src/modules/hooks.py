"""Model builder and Phase 1 / V2 switch dispatcher.

All patches verified against installed `ultralytics 8.4.218`:
- Mod A v1 (`use_mod_a`): Pruned P5 PAN+Head + P2-P4 YAML (kept as negative empirical control).
- Mod A v2 (`use_tc_gate`): Zero-Init Thermal Clutter Suppression Gate (`TC-Gate`) wrapped on
  pretrained neck P3/P4 (#16, #19) AFTER `setup_model()` -> preserves 708/708 pretrained weights!
- Mod B (`use_mod_b`):
    - `mod_b_variant="v2"` (`BiTC-Stem`, recommended): Band-pass `|Avg3 - Avg5|` on Stage 1 only.
    - `mod_b_variant="v1"` (legacy): Max/Min/Avg TCAD on Stages (0, 1).
- Mod C1/C2 (`use_mod_c1`, `use_mod_c2`): FP32 Decoupled Scale-Gated NWD (`DSG-NWD`) hooked
  into `E2ELoss` (`one2many` + `one2one`) during training only (zero inference overhead).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.experiments import ExperimentConfig
from src.modules.nwd_e2e import attach_nwd_e2e_hook
from src.modules.tc_gate import inject_tc_gate_into_model
from src.modules.tcad import inject_tcad_into_model


def attach_phase1_callbacks(model: Any, cfg: ExperimentConfig) -> None:
    """Attach structural (BiTC-Stem / TC-Gate) and supervision (DSG-NWD) hooks."""
    if cfg.use_mod_b or cfg.use_tc_gate:
        def _on_pretrain_routine_start(trainer: Any) -> None:
            orig_setup_model = trainer.setup_model

            def _wrapped_setup_model() -> Any:
                ckpt = orig_setup_model()
                if cfg.use_mod_b:
                    inject_tcad_into_model(
                        trainer.model,
                        stages=cfg.mod_b_stages,
                        variant=cfg.mod_b_variant,
                    )
                if cfg.use_tc_gate:
                    inject_tc_gate_into_model(
                        trainer.model,
                        neck_indices=cfg.tc_gate_indices,
                    )
                return ckpt

            trainer.setup_model = _wrapped_setup_model

        model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
        print(
            f"[STRUCT HOOK] Registered setup_model wrapper: "
            f"BiTC({cfg.mod_b_variant})={cfg.use_mod_b}@{cfg.mod_b_stages}, "
            f"TC-Gate={cfg.use_tc_gate}@{cfg.tc_gate_indices}"
        )

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
    """Instantiate Ultralytics model and register enabled switches."""
    from ultralytics import YOLO

    model_spec = cfg.model_cfg
    local_candidate = repo_root / model_spec
    if local_candidate.exists():
        model_spec = str(local_candidate)

    print(
        f"[MODEL] Loading {model_spec} | "
        f"ModA_v1={cfg.use_mod_a} TC-Gate={cfg.use_tc_gate} "
        f"ModB({cfg.mod_b_variant})={cfg.use_mod_b} "
        f"C1={cfg.use_mod_c1} C2={cfg.use_mod_c2}"
    )
    model = YOLO(model_spec)

    if cfg.pretrained_weights:
        print(f"[MODEL] Applying partial pretrained weight transfer <- {cfg.pretrained_weights}")
        model = model.load(cfg.pretrained_weights)

    attach_phase1_callbacks(model, cfg)
    return model
