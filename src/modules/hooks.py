"""Model builder and Phase 1 / V2 / V3 switch dispatcher.

All patches verified against installed `ultralytics 8.4.218`:
- Legacy V1/V2 switches remain 100% intact for reproducibility & checkpoint loading.
- V3 switches (strictly pretrained-preserving 708/708 + zero-init residual + border-safe):
  - Pillar 1 (`use_v3_odcs`): Zero-inference-overhead `ODCS` supervision (Protected-Floor
    ProgLoss + Selective O2M->O2O Self-Distillation + O2O-only Annealed Assignment with
    dead-anchor floor + PIoU2 / AAAI 2025 SD Loss).
  - Pillar 2 (`use_v3_bitc`): `BorderSafeBiTCStemWrapper` on Stage 1 (+608 params).
  - Pillar 3 (`use_v3_neck`): `BorderSafeCAAResWrapper` (`"caa"`, +11.8K params) or
    `HaarResWrapper` (`"haar"`, +8.6K params) on Neck P3/P4 (#16, #19).
  - Protocol alignment (`align_val_rect_false`): Forces in-training val dataloader to use
    `rect=False` (640x640) matching test_o2o_640 so `best.pt` selection has zero border drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.experiments import ExperimentConfig
from src.modules.nwd_e2e import attach_nwd_e2e_hook
from src.modules.tc_gate import inject_tc_gate_into_model
from src.modules.tcad import inject_tcad_into_model
from src.modules.v3_arch import inject_v3_bitc_into_model, inject_v3_neck_into_model
from src.modules.v3_losses import V3ODCSConfig, attach_v3_odcs_hook


def _transfer_shifted_p2_pretrained(model: Any, weights_path: str) -> None:
    """Transfer shifted PAN (17..22 -> 23..28) & Detect P3/P4/P5 (0..2 -> 1..3) weights
    from 3-scale `yolo26n.pt` into 4-scale `yolo26n-p2.yaml` (preserving full P5 pretraining).
    """
    import torch

    try:
        from ultralytics import YOLO

        src_yolo = YOLO(weights_path)
        src_sd = src_yolo.model.state_dict()
        dst_sd = model.state_dict()
        del src_yolo
    except Exception as exc:
        print(f"[WARN] Smart P2 pretrained transfer skipped ({exc}).")
        return

    # Layer-index mapping: yolo26.yaml (24 layers) -> yolo26-p2.yaml (30 layers)
    # Layers 0..16 match 1:1 (already transferred by Ultralytics); 17..22 shift by +6 -> 23..28.
    layer_map = {17: 23, 18: 24, 19: 25, 20: 26, 21: 27, 22: 28}
    transferred = 0
    with torch.no_grad():
        for src_k, src_v in src_sd.items():
            dst_k = None
            parts = src_k.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit():
                idx = int(parts[1])
                if idx in layer_map:
                    dst_k = ".".join(["model", str(layer_map[idx])] + parts[2:])
                elif idx == 23 and len(parts) >= 4 and parts[2] in (
                    "cv2", "cv3", "one2one_cv2", "one2one_cv3"
                ) and parts[3].isdigit():
                    head_idx = int(parts[3])
                    dst_k = ".".join(["model", "29", parts[2], str(head_idx + 1)] + parts[4:])
            if dst_k and dst_k in dst_sd and dst_sd[dst_k].shape == src_v.shape:
                dst_sd[dst_k].copy_(src_v.to(dtype=dst_sd[dst_k].dtype, device=dst_sd[dst_k].device))
                transferred += 1

    print(
        f"[P2 PRETRAIN] Smart shifted weight transfer <- {weights_path}: "
        f"copied {transferred} extra PAN(23..28)+Detect(P3-P5) tensors."
    )


def attach_phase1_callbacks(model: Any, cfg: ExperimentConfig) -> None:
    """Attach structural (V1/V2/V3), protocol alignment, and supervision hooks."""
    needs_struct = cfg.use_mod_b or cfg.use_tc_gate or cfg.use_v3_bitc or cfg.use_v3_neck
    needs_p2_transfer = bool(cfg.pretrained_weights and "p2" in cfg.model_cfg.lower())
    if needs_struct or needs_p2_transfer or cfg.align_val_rect_false:
        def _on_pretrain_routine_start(trainer: Any) -> None:
            if needs_struct or needs_p2_transfer:
                orig_setup_model = trainer.setup_model

                def _wrapped_setup_model() -> Any:
                    ckpt = orig_setup_model()
                    if needs_p2_transfer:
                        m = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                        if hasattr(m, "model") and len(m.model) == 30:
                            _transfer_shifted_p2_pretrained(m, str(cfg.pretrained_weights))
                    # Legacy V1/V2 structural hooks
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
                    # V3 orthogonal border-safe structural hooks
                    if cfg.use_v3_bitc:
                        inject_v3_bitc_into_model(
                            trainer.model,
                            stages=cfg.v3_bitc_stages,
                        )
                    if cfg.use_v3_neck:
                        inject_v3_neck_into_model(
                            trainer.model,
                            neck_indices=cfg.v3_neck_indices,
                            variant=cfg.v3_neck_variant,
                            reduction=cfg.v3_caa_reduction,
                            pool_kernel=cfg.v3_caa_pool_kernel,
                            strip_kernel=cfg.v3_caa_strip_kernel,
                        )
                    return ckpt

                trainer.setup_model = _wrapped_setup_model

            # Align in-training validation (`best.pt` selection) to rect=False (640x640)
            # without touching training `mosaic=1.0`!
            if cfg.align_val_rect_false and hasattr(trainer, "build_dataset"):
                import inspect

                orig_build_dataset = trainer.build_dataset

                def _val_square_build_dataset(img_path: Any, mode: str = "train", batch: Any = None) -> Any:
                    if mode == "val":
                        try:
                            from ultralytics.data import build_yolo_dataset

                            sig_params = set(inspect.signature(build_yolo_dataset).parameters.keys())
                            m = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                            gs = max(int(m.stride.max() if hasattr(m, "stride") else 32), 32)
                            kwargs = {
                                "mode": "val",
                                "rect": False,
                                "stride": gs,
                            }
                            if "multi_modal" in sig_params:
                                kwargs["multi_modal"] = False
                            ds = build_yolo_dataset(
                                trainer.args,
                                img_path,
                                batch,
                                trainer.data,
                                **{k: v for k, v in kwargs.items() if k in sig_params},
                            )
                            print("[IR-CLEAN] Built square rect=False (640x640) in-training validation dataset.")
                            return ds
                        except Exception as exc:
                            print(f"[WARN] Could not rebuild val dataset with rect=False ({exc}); falling back.")
                    return orig_build_dataset(img_path, mode=mode, batch=batch)

                trainer.build_dataset = _val_square_build_dataset

        model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
        if needs_struct:
            print(
                f"[STRUCT HOOK] Registered setup_model wrapper: "
                f"V2(BiTC={cfg.use_mod_b}, TCGate={cfg.use_tc_gate}) | "
                f"V3(BorderSafeBiTC={cfg.use_v3_bitc}@{cfg.v3_bitc_stages}, "
                f"Neck={cfg.use_v3_neck}:{cfg.v3_neck_variant}@{cfg.v3_neck_indices})"
            )

    # Legacy V2 DSG-NWD hook
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

    # V3 Pillar 1 (ODCS) hook
    if cfg.use_v3_odcs:
        odcs_cfg = V3ODCSConfig(
            use_prog_floor=cfg.v3_prog_floor,
            o2m_floor=cfg.v3_o2m_floor,
            o2o_ramp_frac=cfg.v3_o2o_ramp_frac,
            use_o2m_kd=cfg.v3_o2m_kd,
            kd_weight=cfg.v3_kd_weight,
            kd_sup_weight=cfg.v3_kd_sup_weight,
            kd_warmup_epochs=cfg.v3_kd_warmup_epochs,
            kd_topk_runners=cfg.v3_kd_topk_runners,
            use_o2o_annealed_assign=cfg.v3_o2o_annealed_assign,
            area_thresh=cfg.v3_nwd_area_thresh,
            tau=cfg.v3_nwd_tau,
            w_max=cfg.v3_nwd_w_max,
            nwd_c=cfg.v3_nwd_c,
            anneal_frac=cfg.v3_anneal_frac,
            reg_floor_eps=cfg.v3_reg_floor_eps,
            reg_loss_type=cfg.v3_reg_loss_type,
            piou_lambda=cfg.v3_piou_lambda,
            sd_delta=cfg.v3_sd_delta,
            sd_b_max=cfg.v3_sd_b_max,
            reg_both_heads=cfg.v3_reg_both_heads,
        )
        attach_v3_odcs_hook(model, odcs_cfg)


def build_model_with_switches(cfg: ExperimentConfig, repo_root: Path) -> Any:
    """Instantiate Ultralytics model and register enabled switches."""
    from ultralytics import YOLO

    model_spec = cfg.model_cfg
    local_candidate = repo_root / model_spec
    if local_candidate.exists():
        model_spec = str(local_candidate)

    print(
        f"[MODEL] Loading {model_spec} | "
        f"V2(A1={cfg.use_mod_a}, TCGate={cfg.use_tc_gate}, B={cfg.use_mod_b}, C1={cfg.use_mod_c1}, C2={cfg.use_mod_c2}) | "
        f"V3(ODCS={cfg.use_v3_odcs}, BiTC={cfg.use_v3_bitc}, Neck={cfg.use_v3_neck}:{cfg.v3_neck_variant})"
    )
    model = YOLO(model_spec)

    if cfg.pretrained_weights:
        print(f"[MODEL] Applying partial pretrained weight transfer <- {cfg.pretrained_weights}")
        model = model.load(cfg.pretrained_weights)

    attach_phase1_callbacks(model, cfg)
    return model
