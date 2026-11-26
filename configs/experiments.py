"""Central switchboard for all HIT-UAV YOLO26 experiments.

Design philosophy ("改了跑、跑了改"):
- Every baseline, ablation, and hyperparameter toggle lives here.
- Phase 0 baselines work out of the box.
- Phase 1 custom modifications (Mod A / B / C1 / C2) are controlled by explicit
  boolean flags and gated until Phase 0 source inspection confirms exact
  installed Ultralytics YOLO26 internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class ExperimentConfig:
    # --- Identity & Phase ---
    exp_name: str = "p0_yolo26n_musgd"
    phase: str = "phase0"
    notes: str = ""

    # --- Model & Weights ---
    model_cfg: str = "yolo26n.pt"          # Official .pt or custom .yaml path
    pretrained_weights: Optional[str] = None  # Partial weight load for custom yaml (e.g. "yolo26n.pt")
    is_e2e_model: bool = True              # True for YOLO26 / YOLOv10 (enables nms=False + Δ_E2E eval)

    # --- Dataset Protocol ---
    dataset_mode: str = "4cls"             # "4cls" (primary: drop DontCare) or "5cls" (appendix)
    dontcare_mode: str = "drop"            # "drop" or "mask_blur" (train-only masking for 4cls)

    # --- Training Schedule ---
    epochs: int = 200
    patience: int = 50
    batch: int = 32
    imgsz: int = 640
    optimizer: str = "auto"                # "auto" (MuSGD for YOLO26), "SGD", "AdamW"
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    close_mosaic: int = 10
    workers: int = 4
    cache: bool | str = True               # True ("ram"), "disk" (100% deterministic), or False
    amp: bool = True
    seed: int = 42
    deterministic: bool = True

    # --- Optional IR-Clean Augmentation & Val Alignment Overrides (None = Ultralytics default) ---
    hsv_h: Optional[float] = None          # IR-Clean: 0.0 (LWIR is single-band thermal, no hue)
    hsv_s: Optional[float] = None          # IR-Clean: 0.0 (no saturation)
    hsv_v: Optional[float] = None          # IR-Clean: 0.25 (models thermal gain / contrast drift)
    erasing: Optional[float] = None        # IR-Clean: 0.0 (prevents erasing 8x10px IR pedestrians)
    align_val_rect_false: bool = False     # True: align in-training val (`best.pt`) to 640x640 square

    # --- Evaluation & Benchmark Switches ---
    eval_rect_640x512: bool = True         # Also benchmark native 640x512 rectangular inference
    export_onnx_bench: bool = False        # Export ONNX & measure CPU/GPU latency (enable for final table runs)
    save_failure_vis: bool = True          # Save qualitative val/test prediction samples

    # --- Safety & Colab Compute Unit Guards ---
    smoke_test: bool = False               # If True: overrides to 2 epochs, quick pipeline verification
    resume: bool = True                    # Auto-resume from Drive checkpoint if interrupted
    save_period: int = 25                  # Sync periodic checkpoints to Drive every N epochs

    # --- Phase 1 Core Innovation Switches (v1 legacy + v2 recommended) ---
    use_mod_a: bool = False                  # Mod A v1: Prune P5 PAN+Head -> P2-P4 YAML (negative control)
    mod_a_p2_channels: int = 32
    mod_a_keep_p4_psa: bool = False

    use_tc_gate: bool = False                # Mod A v2 (TC-Gate): Zero-Init Thermal Clutter Gate on Neck P3/P4
    tc_gate_indices: tuple[int, ...] = (16, 19)

    use_mod_b: bool = False                  # Mod B: Zero-Init Gated Residual Thermal Contrast Downsampling
    mod_b_variant: str = "v2"                # "v2" (BiTC-Stem |Avg3-Avg5| on Stage 1) or "v1" (legacy Stage 0+1)
    mod_b_stages: tuple[int, ...] = (1,)     # Default Stage 1 only for v2 (preserves crisp Stage 0 & low latency)

    use_mod_c1: bool = False                 # Mod C1 v2 (DSG-Assign): FP32 Decoupled Scale-Gated NWD in Assigner
    use_mod_c2: bool = False                 # Mod C2 v2 (DSG-Loss): FP32 Scale-Gated NWD in DFL-Free BboxLoss
    nwd_area_thresh: float = 1024.0          # Area threshold A_0 (px^2, COCO small = 32x32 = 1024)
    nwd_tau: float = 256.0                   # Smooth sigmoid temperature for area gating
    nwd_w_max: float = 0.25                  # Calibrated NWD weight ceiling for tiny IR targets
    nwd_constant_c: float = 8.0              # Tighter NWD constant matched to 8x10 px IR pedestrians

    # --- Phase 1 V3 Innovation Switches (Orthogonal, Border-Safe, Sacred-O2M ODCS) ---
    use_v3_odcs: bool = False                # Pillar 1 master switch (zero inference overhead)
    v3_prog_floor: bool = False              # Pillar 1a: Protected-Floor ProgLoss (o2m >= 0.45, o2o ramp 60%)
    v3_o2m_floor: float = 0.45
    v3_o2o_ramp_frac: float = 0.60
    v3_o2m_kd: bool = False                  # Pillar 1b: Selective O2M -> O2O Self-Distillation
    v3_kd_weight: float = 0.30
    v3_kd_sup_weight: float = 0.10
    v3_kd_warmup_epochs: int = 10
    v3_kd_topk_runners: int = 7
    v3_o2o_annealed_assign: bool = False     # Pillar 1c: O2O-only annealed NWD assign (A0=256) + dead-anchor floor
    v3_nwd_area_thresh: float = 256.0
    v3_nwd_tau: float = 64.0
    v3_nwd_w_max: float = 0.25
    v3_nwd_c: float = 8.0
    v3_anneal_frac: float = 0.60
    v3_reg_floor_eps: float = 0.25
    v3_reg_loss_type: str = "ciou"           # "ciou", "piou2" (Neural Networks 2024), or "sdloss" (AAAI 2025)
    v3_piou_lambda: float = 1.3
    v3_sd_delta: float = 0.5
    v3_sd_b_max: float = 256.0
    v3_reg_both_heads: bool = False          # False = sacred O2M teacher; True = also apply PIoU2/SDLoss to O2M

    use_v3_bitc: bool = False                # Pillar 2: Border-Safe BiTC-Stem (replicate pad + Charbonnier, +608 params)
    v3_bitc_stages: tuple[int, ...] = (1,)

    use_v3_neck: bool = False                # Pillar 3: Border-Safe CAA-Res ("caa") or Haar-Res ("haar") on P3/P4
    v3_neck_variant: str = "caa"             # "caa" (+11.8K params) or "haar" (+8.6K params)
    v3_neck_indices: tuple[int, ...] = (16, 19)
    v3_caa_reduction: int = 4
    v3_caa_pool_kernel: int = 7
    v3_caa_strip_kernel: int = 11

    # --- Extra Ultralytics train kwargs ---
    extra_train_kwargs: Dict[str, object] = field(default_factory=dict)

    def apply_smoke_test(self) -> "ExperimentConfig":
        """Shrink run to a ~60-90s end-to-end sanity check without mutating presets."""
        if self.smoke_test:
            self.epochs = 2
            self.patience = 2
            self.batch = min(self.batch, 16)
            self.close_mosaic = 0
            self.save_period = -1
            self.export_onnx_bench = False
            self.exp_name = f"{self.exp_name}_smoke"
        return self

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PRESETS: Dict[str, ExperimentConfig] = {
    # =========================================================================
    # Phase 0: Baseline Benchmarking + Optimizer & End-to-End Gap Diagnostics
    # =========================================================================
    "p0_yolo26n_musgd": ExperimentConfig(
        exp_name="p0_yolo26n_musgd",
        phase="phase0",
        notes="YOLO26n official baseline with optimizer=auto (selects AdamW lr=0.00125 on HIT-UAV 4cls)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
    ),
    "p0_yolo26n_true_musgd": ExperimentConfig(
        exp_name="p0_yolo26n_true_musgd",
        phase="phase0",
        notes="YOLO26n explicit optimizer=MuSGD check vs auto(AdamW) and SGD on HIT-UAV 4cls",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="MuSGD",
        lr0=0.01,
        epochs=200,
        batch=32,
    ),
    "p0_yolo26n_sgd": ExperimentConfig(
        exp_name="p0_yolo26n_sgd",
        phase="phase0",
        notes="YOLO26n optimizer check: standard SGD vs default MuSGD on 2,029-image HIT-UAV",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="SGD",
        lr0=0.01,
        epochs=200,
        batch=32,
    ),
    "p0_yolo11n": ExperimentConfig(
        exp_name="p0_yolo11n",
        phase="phase0",
        notes="YOLO11n NMS baseline under identical HIT-UAV 4cls protocol",
        model_cfg="yolo11n.pt",
        is_e2e_model=False,
        optimizer="SGD",
        epochs=200,
        batch=32,
    ),
    "p0_yolov8n": ExperimentConfig(
        exp_name="p0_yolov8n",
        phase="phase0",
        notes="YOLOv8n NMS baseline under identical HIT-UAV 4cls protocol",
        model_cfg="yolov8n.pt",
        is_e2e_model=False,
        optimizer="SGD",
        epochs=200,
        batch=32,
    ),
    "p0_yolo26n_5cls": ExperimentConfig(
        exp_name="p0_yolo26n_5cls",
        phase="phase0",
        notes="Appendix baseline: YOLO26n on full 5-class HIT-UAV (including DontCare)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        dataset_mode="5cls",
        optimizer="auto",
        epochs=200,
        batch=32,
    ),

    # =========================================================================
    # Phase 0.5 / SOTA Comparison Baselines (Same Split & Unified Protocol)
    # =========================================================================
    "sota_yolov10n": ExperimentConfig(
        exp_name="sota_yolov10n",
        phase="sota",
        notes="YOLOv10n end-to-end baseline on HIT-UAV 4cls",
        model_cfg="yolov10n.pt",
        is_e2e_model=True,
        optimizer="SGD",
        epochs=200,
        batch=32,
    ),
    "sota_yolo12n": ExperimentConfig(
        exp_name="sota_yolo12n",
        phase="sota",
        notes="YOLO12n attention-centric baseline on HIT-UAV 4cls",
        model_cfg="yolo12n.pt",
        is_e2e_model=False,
        optimizer="SGD",
        epochs=200,
        batch=32,
    ),
    "sota_rtdetr_r18": ExperimentConfig(
        exp_name="sota_rtdetr_r18",
        phase="sota",
        notes="RT-DETR-R18 end-to-end Transformer baseline on HIT-UAV 4cls",
        model_cfg="rtdetr-l.pt",  # Update to rtdetr-r18 yaml/pt once verified in Phase 0
        is_e2e_model=True,
        optimizer="AdamW",
        lr0=0.0002,
        epochs=150,
        batch=16,
    ),
    "sota_yolo26n_p2_official": ExperimentConfig(
        exp_name="sota_yolo26n_p2_official",
        phase="sota",
        notes="Strong 4-head P2-P5 control (yolo26-p2.yaml with partial load of yolo26n.pt)",
        model_cfg="yolo26n-p2.yaml",
        pretrained_weights="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=24,
    ),

    # =========================================================================
    # Phase 1: Controlled Ablation Matrix (Unlocked After Phase 0 Source Audit)
    # =========================================================================
    "p1_mod_a": ExperimentConfig(
        exp_name="p1_mod_a",
        phase="phase1",
        notes="Ablation +A: Pruned P5 PAN+Head + Lightweight DWConv P2-P4 + partial load",
        model_cfg="configs/models/yolo26n_p2p4.yaml",
        pretrained_weights="yolo26n.pt",
        is_e2e_model=True,
        use_mod_a=True,
        epochs=200,
        batch=24,
    ),
    "p1_mod_b": ExperimentConfig(
        exp_name="p1_mod_b",
        phase="phase1",
        notes="Ablation +B (v1 legacy): Zero-init Max/Min/Avg TCAD on Stages (0,1)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_b=True,
        mod_b_variant="v1",
        mod_b_stages=(0, 1),
        epochs=200,
        batch=32,
    ),
    "p1_mod_c1": ExperimentConfig(
        exp_name="p1_mod_c1",
        phase="phase1",
        notes="Ablation +C1: Scale-Gated NWD in TaskAlignedAssigner (O2M + O2O) only",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_c1=True,
        epochs=200,
        batch=32,
    ),
    "p1_mod_c12": ExperimentConfig(
        exp_name="p1_mod_c12",
        phase="phase1",
        notes="Ablation +C1+C2: Scale-Gated NWD in Assigner + BboxLoss",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_c1=True,
        use_mod_c2=True,
        epochs=200,
        batch=32,
    ),
    "p1_mod_ab": ExperimentConfig(
        exp_name="p1_mod_ab",
        phase="phase1",
        notes="Ablation +A+B: Lightweight P2-P4 + Zero-init TCAD",
        model_cfg="configs/models/yolo26n_p2p4.yaml",
        pretrained_weights="yolo26n.pt",
        is_e2e_model=True,
        use_mod_a=True,
        use_mod_b=True,
        epochs=200,
        batch=24,
    ),
    "p1_mod_ac1": ExperimentConfig(
        exp_name="p1_mod_ac1",
        phase="phase1",
        notes="Ablation +A+C1: Lightweight P2-P4 + Scale-Gated NWD Assigner",
        model_cfg="configs/models/yolo26n_p2p4.yaml",
        pretrained_weights="yolo26n.pt",
        is_e2e_model=True,
        use_mod_a=True,
        use_mod_c1=True,
        epochs=200,
        batch=24,
    ),
    "p1_full_abc1": ExperimentConfig(
        exp_name="p1_full_abc1",
        phase="phase1",
        notes="Full candidate 1 (+A+B+C1): avoids potential TCAD vs NWD-regression gradient conflict",
        model_cfg="configs/models/yolo26n_p2p4.yaml",
        pretrained_weights="yolo26n.pt",
        is_e2e_model=True,
        use_mod_a=True,
        use_mod_b=True,
        use_mod_c1=True,
        export_onnx_bench=True,
        epochs=200,
        batch=24,
    ),
    "p1_full_abc12": ExperimentConfig(
        exp_name="p1_full_abc12",
        phase="phase1",
        notes="Full candidate 2 (+A+B+C1+C2): full Scale-Gated NWD in both Assigner and BboxLoss",
        model_cfg="configs/models/yolo26n_p2p4.yaml",
        pretrained_weights="yolo26n.pt",
        is_e2e_model=True,
        use_mod_a=True,
        use_mod_b=True,
        use_mod_c1=True,
        use_mod_c2=True,
        export_onnx_bench=True,
        epochs=200,
        batch=24,
    ),

    # =========================================================================
    # Phase 1 V2: Pretrained-Preserving (708/708) + FP32 Decoupled DSG-NWD Suite
    # =========================================================================
    "p1v2_bitc": ExperimentConfig(
        exp_name="p1v2_bitc",
        phase="p1v2",
        notes="V2 +BiTC-Stem: Band-pass |Avg3-Avg5| on Stage 1 only (~576 params, <0.15ms, 708/708 pretrained)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_b=True,
        mod_b_variant="v2",
        mod_b_stages=(1,),
        epochs=200,
        batch=32,
    ),
    "p1v2_tcgate": ExperimentConfig(
        exp_name="p1v2_tcgate",
        phase="p1v2",
        notes="V2 +TC-Gate: Zero-init local-contrast spatial clutter gate on Neck P3/P4 (~230 params, <0.12ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_tc_gate=True,
        tc_gate_indices=(16, 19),
        epochs=200,
        batch=32,
    ),
    "p1v2_c1": ExperimentConfig(
        exp_name="p1v2_c1",
        phase="p1v2",
        notes="V2 +DSG-Assign (C1): FP32 Decoupled Scale-Gated NWD in Assigner only (IoU-calibrated target_scores, 0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_c1=True,
        epochs=200,
        batch=32,
    ),
    "p1v2_c2": ExperimentConfig(
        exp_name="p1v2_c2",
        phase="p1v2",
        notes="V2 +DSG-Loss (C2): FP32 Scale-Gated NWD in DFL-Free BboxLoss only (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_c2=True,
        epochs=200,
        batch=32,
    ),
    "p1v2_c12": ExperimentConfig(
        exp_name="p1v2_c12",
        phase="p1v2",
        notes="V2 +DSG-NWD (C1+C2): FP32 Decoupled Assigner + Scale-Gated BboxLoss (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_c1=True,
        use_mod_c2=True,
        epochs=200,
        batch=32,
    ),
    "p1v2_bitc_tcgate": ExperimentConfig(
        exp_name="p1v2_bitc_tcgate",
        phase="p1v2",
        notes="V2 +BiTC-Stem +TC-Gate: Front-end + Neck thermal clutter synergy (708/708 pretrained)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_b=True,
        mod_b_variant="v2",
        mod_b_stages=(1,),
        use_tc_gate=True,
        tc_gate_indices=(16, 19),
        epochs=200,
        batch=32,
    ),
    "p1v2_full_c1": ExperimentConfig(
        exp_name="p1v2_full_c1",
        phase="p1v2",
        notes="V2 Full Candidate 1: BiTC-Stem + TC-Gate + DSG-Assign (C1 only, zero regression conflict)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_b=True,
        mod_b_variant="v2",
        mod_b_stages=(1,),
        use_tc_gate=True,
        tc_gate_indices=(16, 19),
        use_mod_c1=True,
        epochs=200,
        batch=32,
    ),
    "p1v2_full_c12": ExperimentConfig(
        exp_name="p1v2_full_c12",
        phase="p1v2",
        notes="V2 Full Candidate 2: BiTC-Stem + TC-Gate + DSG-NWD (C1+C2 all 3 pillars)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_b=True,
        mod_b_variant="v2",
        mod_b_stages=(1,),
        use_tc_gate=True,
        tc_gate_indices=(16, 19),
        use_mod_c1=True,
        use_mod_c2=True,
        epochs=200,
        batch=32,
    ),

    # =========================================================================
    # Phase 1 V3: Unified IR-Clean Protocol + Orthogonal 3-Pillar Breakthrough
    #   Shared IR-Clean baseline: hsv_h=0, hsv_s=0, hsv_v=0.25, erasing=0.0,
    #   close_mosaic=15, deterministic=True, cache="disk", align_val_rect_false=True.
    #   Preserves 708/708 pretrained weights & zero collision with 2026 competitors!
    # =========================================================================
    "v3_e0_base": ExperimentConfig(
        exp_name="v3_e0_base",
        phase="v3",
        notes="V3 E0 IR-Clean Baseline YOLO26n (AdamW auto, hsv_h/s=0, hsv_v=0.25, erasing=0, val rect=F)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
    ),
    "v3_e1_p1_distill": ExperimentConfig(
        exp_name="v3_e1_p1_distill",
        phase="v3",
        notes="V3 E1 Pillar 1a+1b: Protected-Floor ProgLoss (o2m>=0.45) + Selective O2M->O2O KD (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_prog_floor=True,
        v3_o2m_kd=True,
    ),
    "v3_e2_p1_piou": ExperimentConfig(
        exp_name="v3_e2_p1_piou",
        phase="v3",
        notes="V3 E2 Pillar 1c (PIoU2): O2O-only Annealed NWD(A0=256) + Dead-Anchor Floor + PIoU2 (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
    ),
    "v3_e2b_p1_sdloss": ExperimentConfig(
        exp_name="v3_e2b_p1_sdloss",
        phase="v3",
        notes="V3 E2b Pillar 1c (AAAI25 SD Loss): O2O-only Annealed NWD(A0=256) + Dead-Anchor Floor + SDB (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="sdloss",
    ),
    "v3_e3_p1_full": ExperimentConfig(
        exp_name="v3_e3_p1_full",
        phase="v3",
        notes="V3 E3 Full Pillar 1 ODCS (ProgFloor + O2M->O2O KD + O2O Anneal + PIoU2, +0 params / +0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_prog_floor=True,
        v3_o2m_kd=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
    ),
    "v3_e3b_p1_full_sd": ExperimentConfig(
        exp_name="v3_e3b_p1_full_sd",
        phase="v3",
        notes="V3 E3b Full Pillar 1 ODCS (ProgFloor + O2M->O2O KD + O2O Anneal + SD Loss, +0 params / +0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_prog_floor=True,
        v3_o2m_kd=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="sdloss",
    ),
    "v3_e4_bitc_safe": ExperimentConfig(
        exp_name="v3_e4_bitc_safe",
        phase="v3",
        notes="V3 E4 Pillar 2: Border-Safe BiTC-Stem (replicate pad + Charbonnier @ Stage 1, +608 params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_bitc=True,
    ),
    "v3_e5_caa_res": ExperimentConfig(
        exp_name="v3_e5_caa_res",
        phase="v3",
        notes="V3 E5 Pillar 3: Border-Safe CAA-Res (Avg7 + 1x11/11x1 strips @ Neck P3/P4, +11.8K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v3_e5b_haar_res": ExperimentConfig(
        exp_name="v3_e5b_haar_res",
        phase="v3",
        notes="V3 E5b Pillar 3 Fallback: Zero-Init 2D Haar-Res @ Neck P3/P4 (+8.6K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_neck=True,
        v3_neck_variant="haar",
    ),
    "v3_e6_arch_combo": ExperimentConfig(
        exp_name="v3_e6_arch_combo",
        phase="v3",
        notes="V3 E6 Pillar 2+3 Orthogonality Check: Border-Safe BiTC-Stem + CAA-Res (+12.4K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v3_e7_full": ExperimentConfig(
        exp_name="v3_e7_full",
        phase="v3",
        notes="V3 E7 Full Model (Pillar 1 ODCS-PIoU2 + Pillar 2 BiTC-Safe + Pillar 3 CAA-Res)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_prog_floor=True,
        v3_o2m_kd=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v3_e7b_full_sd": ExperimentConfig(
        exp_name="v3_e7b_full_sd",
        phase="v3",
        notes="V3 E7b Full Model (Pillar 1 ODCS-SDLoss + Pillar 2 BiTC-Safe + Pillar 3 CAA-Res)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=15,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        erasing=0.0,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_prog_floor=True,
        v3_o2m_kd=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="sdloss",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),

    # =========================================================================
    # Phase 1 V3.1: IR-Reg Golden Protocol + Clean 3-Pillar Champion
    #   Diagnostic findings from V3 (seed=42):
    #     1. Dropping erasing (0.4->0.0) & hsv_v (0.4->0.25) collapsed baseline
    #        from 87.67/57.51 -> 84.37/52.23 (Ep-44 overfitting on 2,029 imgs).
    #     2. ProgFloor + hard runner-up KD (v3_e1) degraded O2M ceiling & 3cls (-0.54%).
    #     3. Pillar 1c (v3_e2_p1_piou, 0ms) surged to 88.80% O2O (+4.43% over e0)
    #        with sacred O2M=89.77%; Pillar 2+3 (v3_e6) hit 92.72% 3cls & 58.3% rect m95!
    #   IR-Reg restores erasing=0.4, hsv_v=0.4 (AGC gain jitter), close_mosaic=10,
    #   while keeping hsv_h=0, hsv_s=0 (LWIR physical clean), cache="disk", val rect=F.
    # =========================================================================
    "v31_e0_base": ExperimentConfig(
        exp_name="v31_e0_base",
        phase="v31",
        notes="V3.1 E0 IR-Reg Baseline YOLO26n (hsv_h/s=0, hsv_v=0.4, erasing=0.4, close_mosaic=10, val rect=F)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
    ),
    "v31_e1_p1c_piou": ExperimentConfig(
        exp_name="v31_e1_p1c_piou",
        phase="v31",
        notes="V3.1 E1 Pillar 1c (ODCS-PIoU2): O2O Annealed NWD(A0=256) + Dead-Anchor Floor + PIoU2 (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
    ),
    "v31_e2_bitc_safe": ExperimentConfig(
        exp_name="v31_e2_bitc_safe",
        phase="v31",
        notes="V3.1 E2 Pillar 2: Border-Safe BiTC-Stem (replicate pad + Charbonnier @ Stage 1, +608 params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
        use_v3_bitc=True,
    ),
    "v31_e3_caa_res": ExperimentConfig(
        exp_name="v31_e3_caa_res",
        phase="v31",
        notes="V3.1 E3 Pillar 3: Border-Safe CAA-Res (Avg7 + 1x11/11x1 strips @ Neck P3/P4, +11.8K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v31_e4_arch_combo": ExperimentConfig(
        exp_name="v31_e4_arch_combo",
        phase="v31",
        notes="V3.1 E4 Pillar 2+3: Border-Safe BiTC-Stem + CAA-Res (+12.4K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v31_e5_champion": ExperimentConfig(
        exp_name="v31_e5_champion",
        phase="v31",
        notes="V3.1 Champion (IR-Reg): Pillar 1c (O2O Anneal+PIoU2) + Pillar 2 (BiTC-Safe) + Pillar 3 (CAA-Res)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v31_e5b_champ_boxkd": ExperimentConfig(
        exp_name="v31_e5b_champ_boxkd",
        phase="v31",
        notes="V3.1 Champion + Gentle Box/Top1 KD only (no ProgFloor, kd_sup=0.0, kd_w=0.15)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        epochs=200,
        batch=32,
        close_mosaic=10,
        cache="disk",
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        erasing=0.4,
        align_val_rect_false=True,
        use_v3_odcs=True,
        v3_prog_floor=False,
        v3_o2m_kd=True,
        v3_kd_weight=0.15,
        v3_kd_sup_weight=0.0,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    # =========================================================================
    # Phase 1 V3.2: Official Ultralytics Protocol (ZERO protocol drift vs p0!)
    #   Root-cause autopsy of V3.1 (seed=42, L4):
    #     - Relative to v31_e0_base (84.26/54.19, O2M 86.32), ALL 3 pillars gave
    #       positive gains & crushed Δ_E2E (+2.06% -> +0.44%~+0.61%).
    #     - HOWEVER, align_val_rect_false=True + cache="disk" + hsv_h/s=0
    #       caused v31_e0_base to lock a suboptimal best.pt at Ep 69 (-3.41% mAP50
    #       / -3.32% mAP50-95 / -3.66% O2M vs p0_yolo26n_musgd 87.67/57.51)!
    #   V3.2 restores 100% official Ultralytics train/val defaults (val rect=True,
    #   cache=True, default hsv/erasing/close_mosaic=10) so V3.2 stands on the
    #   exact same ~90% O2M ceiling as p0_yolo26n_musgd.
    # =========================================================================
    "v32_e0_base": ExperimentConfig(
        exp_name="v32_e0_base",
        phase="v32",
        notes="V3.2 E0 Official-Protocol Baseline YOLO26n (exact p0_yolo26n_musgd recipe: val rect=T, cache=ram, default aug)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
        cache=True,
        align_val_rect_false=False,
    ),
    "v32_e1_p1c_piou": ExperimentConfig(
        exp_name="v32_e1_p1c_piou",
        phase="v32",
        notes="V3.2 E1 Official-Protocol + Pillar 1c (O2O Annealed NWD A0=256 + Dead-Anchor Floor + PIoU2, 0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
        cache=True,
        align_val_rect_false=False,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
    ),
    "v32_e2_bitc_safe": ExperimentConfig(
        exp_name="v32_e2_bitc_safe",
        phase="v32",
        notes="V3.2 E2 Official-Protocol + Pillar 2: Border-Safe BiTC-Stem (+608 params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
        cache=True,
        align_val_rect_false=False,
        use_v3_bitc=True,
    ),
    "v32_e3_caa_res": ExperimentConfig(
        exp_name="v32_e3_caa_res",
        phase="v32",
        notes="V3.2 E3 Official-Protocol + Pillar 3: Border-Safe CAA-Res (+11.8K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
        cache=True,
        align_val_rect_false=False,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v32_e4_arch_combo": ExperimentConfig(
        exp_name="v32_e4_arch_combo",
        phase="v32",
        notes="V3.2 E4 Official-Protocol + Pillar 2+3: Border-Safe BiTC-Stem + CAA-Res (+12.4K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
        cache=True,
        align_val_rect_false=False,
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    "v32_e5_champion": ExperimentConfig(
        exp_name="v32_e5_champion",
        phase="v32",
        notes="V3.2 Champion (Official-Protocol): Pillar 1c (O2O Anneal+PIoU2) + Pillar 2 (BiTC-Safe) + Pillar 3 (CAA-Res)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=200,
        batch=32,
        cache=True,
        align_val_rect_false=False,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
    ),
    # =========================================================================
    # Phase 1 V3.3: 100-Epoch Cosine Schedule + Clean Close-Mosaic (HIT-UAV 2K)
    #   Root-cause autopsy of V3.2 (seed=42, L4):
    #     - Restoring official val rect=True recovered v32_e0_base to 87.29/57.39
    #       (O2M 89.59/59.17, Best Ep 84), matching p0 (87.67/57.51, Ep 84).
    #     - Under epochs=200 linear decay, e1/e5 picked late best.pt (Ep 97/161),
    #       pushing Val to 91.31/92.09 while overfitting 290-img Val & dropping
    #       Test O2M (-2.73%/-3.37%), and NEVER reaching close_mosaic=10!
    #     - Also isolates PIoU2 vs native CIoU under Pillar 1c.
    #   V3.3 uses epochs=100, patience=100, cos_lr=True, close_mosaic=10 so EVERY
    #   model completes full cosine decay + 10-epoch clean 640x512 fine-tuning
    #   without late-epoch Val overfitting (~18-30 min/run on L4).
    # =========================================================================
    "v33_e0_base": ExperimentConfig(
        exp_name="v33_e0_base",
        phase="v33",
        notes="V3.3 E0 Baseline YOLO26n (100ep, cos_lr=T, close_mosaic=10, val rect=T)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_e1_p1c_ciou": ExperimentConfig(
        exp_name="v33_e1_p1c_ciou",
        phase="v33",
        notes="V3.3 E1 Pillar 1c-CIoU (100ep cos_lr): O2O Annealed NWD(A0=256) + Dead-Anchor Floor + Native CIoU (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="ciou",
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_e1b_p1c_piou": ExperimentConfig(
        exp_name="v33_e1b_p1c_piou",
        phase="v33",
        notes="V3.3 E1b Pillar 1c-PIoU2 (100ep cos_lr): O2O Annealed NWD(A0=256) + Dead-Anchor Floor + PIoU2 (0ms)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="piou2",
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_e2_arch_combo": ExperimentConfig(
        exp_name="v33_e2_arch_combo",
        phase="v33",
        notes="V3.3 E2 Pillar 2+3 (100ep cos_lr): Border-Safe BiTC-Stem + CAA-Res (+12.4K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_e3_full_ciou": ExperimentConfig(
        exp_name="v33_e3_full_ciou",
        phase="v33",
        notes="V3.3 E3 Full (100ep cos_lr): Pillar 1c-CIoU + Border-Safe BiTC-Stem + CAA-Res",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="ciou",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
        extra_train_kwargs={"cos_lr": True},
    ),
    # --- V3.3 Single-Module Ablations (same 100ep cos_lr protocol) ---
    "v33_e2a_bitc_safe": ExperimentConfig(
        exp_name="v33_e2a_bitc_safe",
        phase="v33",
        notes="V3.3 E2a Pillar 2 only (100ep cos_lr): Border-Safe BiTC-Stem (+608 params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        use_v3_bitc=True,
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_e2b_caa_res": ExperimentConfig(
        exp_name="v33_e2b_caa_res",
        phase="v33",
        notes="V3.3 E2b Pillar 3 only (100ep cos_lr): Border-Safe CAA-Res (+11.8K params)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        align_val_rect_false=False,
        use_v3_neck=True,
        v3_neck_variant="caa",
        extra_train_kwargs={"cos_lr": True},
    ),
    # --- V3.3 Unified-Protocol SOTA Baselines (100ep cos_lr, HIT-UAV 4cls) ---
    "v33_sota_yolov8n": ExperimentConfig(
        exp_name="v33_sota_yolov8n",
        phase="sota",
        notes="SOTA (100ep cos_lr): YOLOv8n (NMS)",
        model_cfg="yolov8n.pt",
        is_e2e_model=False,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_sota_yolov10n": ExperimentConfig(
        exp_name="v33_sota_yolov10n",
        phase="sota",
        notes="SOTA (100ep cos_lr): YOLOv10n (NMS-free E2E)",
        model_cfg="yolov10n.pt",
        is_e2e_model=True,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_sota_yolo11n": ExperimentConfig(
        exp_name="v33_sota_yolo11n",
        phase="sota",
        notes="SOTA (100ep cos_lr): YOLO11n (NMS)",
        model_cfg="yolo11n.pt",
        is_e2e_model=False,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_sota_yolo12n": ExperimentConfig(
        exp_name="v33_sota_yolo12n",
        phase="sota",
        notes="SOTA (100ep cos_lr): YOLO12n (Attention-centric NMS)",
        model_cfg="yolo12n.pt",
        is_e2e_model=False,
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        extra_train_kwargs={"cos_lr": True},
    ),
    # --- V3.3 5-Class HIT-UAV Appendix (includes DontCare) ---
    "v33_5cls_base": ExperimentConfig(
        exp_name="v33_5cls_base",
        phase="appendix",
        notes="HIT-UAV 5cls Appendix (100ep cos_lr): YOLO26n baseline",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        dataset_mode="5cls",
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        extra_train_kwargs={"cos_lr": True},
    ),
    "v33_5cls_full": ExperimentConfig(
        exp_name="v33_5cls_full",
        phase="appendix",
        notes="HIT-UAV 5cls Appendix (100ep cos_lr): V3.3 Full (P1c-CIoU + BiTC-Safe + CAA-Res)",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        dataset_mode="5cls",
        optimizer="auto",
        epochs=100,
        patience=100,
        batch=32,
        close_mosaic=10,
        cache=True,
        use_v3_odcs=True,
        v3_o2o_annealed_assign=True,
        v3_reg_loss_type="ciou",
        use_v3_bitc=True,
        use_v3_neck=True,
        v3_neck_variant="caa",
        extra_train_kwargs={"cos_lr": True},
    ),
}


def get_experiment_config(
    exp_name: str,
    *,
    smoke_test: Optional[bool] = None,
    seed: Optional[int] = None,
    optimizer: Optional[str] = None,
    dataset_mode: Optional[str] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    resume: Optional[bool] = None,
    export_onnx_bench: Optional[bool] = None,
) -> ExperimentConfig:
    """Return a deep copy of a named preset with optional runtime overrides."""
    if exp_name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"Unknown experiment preset '{exp_name}'. Available: {available}")

    base = PRESETS[exp_name]
    cfg = ExperimentConfig(**{**asdict(base), "extra_train_kwargs": dict(base.extra_train_kwargs)})

    if seed is not None:
        cfg.seed = seed
    if optimizer is not None:
        cfg.optimizer = optimizer
    if dataset_mode is not None:
        cfg.dataset_mode = dataset_mode
    if epochs is not None:
        cfg.epochs = epochs
    if batch is not None:
        cfg.batch = batch
    if resume is not None:
        cfg.resume = resume
    if export_onnx_bench is not None:
        cfg.export_onnx_bench = export_onnx_bench
    if smoke_test is not None:
        cfg.smoke_test = smoke_test

    if cfg.seed != 42:
        cfg.exp_name = f"{cfg.exp_name}_s{cfg.seed}"

    return cfg.apply_smoke_test()


def list_presets() -> str:
    lines = ["Available experiment presets:"]
    for name, cfg in PRESETS.items():
        flags = []
        if cfg.use_mod_a:
            flags.append("A_v1")
        if cfg.use_tc_gate:
            flags.append("TCGate")
        if cfg.use_mod_b:
            flags.append(f"B_{cfg.mod_b_variant}")
        if cfg.use_mod_c1:
            flags.append("C1")
        if cfg.use_mod_c2:
            flags.append("C2")
        if cfg.use_v3_odcs:
            sub = []
            if cfg.v3_prog_floor:
                sub.append("PF")
            if cfg.v3_o2m_kd:
                sub.append("KD")
            if cfg.v3_o2o_annealed_assign:
                sub.append("AA")
            if cfg.v3_reg_loss_type != "ciou":
                sub.append(cfg.v3_reg_loss_type.upper())
            flags.append("ODCS(" + ",".join(sub) + ")")
        if cfg.use_v3_bitc:
            flags.append("BiTC_v3")
        if cfg.use_v3_neck:
            flags.append(f"Neck_{cfg.v3_neck_variant}")
        flag_str = "+".join(flags) if flags else "baseline"
        lines.append(
            f"  - {name:<26} [{cfg.phase:<6}] model={cfg.model_cfg:<30} mods={flag_str:<10} ({cfg.notes})"
        )
    return "\n".join(lines)
