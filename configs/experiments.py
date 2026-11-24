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
    cache: bool = True                     # Cache 2,029 640x512 IR images in Colab L4 53GB RAM (huge speedup)
    amp: bool = True
    seed: int = 42
    deterministic: bool = True

    # --- Evaluation & Benchmark Switches ---
    eval_rect_640x512: bool = True         # Also benchmark native 640x512 rectangular inference
    export_onnx_bench: bool = False        # Export ONNX & measure CPU/GPU latency (enable for final table runs)
    save_failure_vis: bool = True          # Save qualitative val/test prediction samples

    # --- Safety & Colab Compute Unit Guards ---
    smoke_test: bool = False               # If True: overrides to 2 epochs, quick pipeline verification
    resume: bool = True                    # Auto-resume from Drive checkpoint if interrupted
    save_period: int = 25                  # Sync periodic checkpoints to Drive every N epochs

    # --- Phase 1 Core Innovation Switches (A / B / C1 / C2) ---
    use_mod_a: bool = False                # Mod A: Prune P5 PAN+Head -> Lightweight DWConv P2-P4 Head
    mod_a_p2_channels: int = 32            # Compressed P2 channel width to control FLOPs/latency
    mod_a_keep_p4_psa: bool = False        # Optional: retain lightweight PSA block on P4 output

    use_mod_b: bool = False                # Mod B: Zero-Init Gated Residual Bidirectional TCAD
    mod_b_stages: tuple[int, ...] = (0, 1) # Shallow downsampling stages to wrap with TCAD

    use_mod_c1: bool = False               # Mod C1: Scale-Gated NWD inside TaskAlignedAssigner (O2M + O2O conflict)
    use_mod_c2: bool = False               # Mod C2: Scale-Gated NWD inside BboxLoss (split from C1 to test B+C2 conflict)
    nwd_area_thresh: float = 1024.0        # Area threshold A_0 (px^2, COCO small = 32x32 = 1024)
    nwd_tau: float = 256.0                 # Smooth sigmoid temperature for area gating
    nwd_w_max: float = 0.4                 # Maximum NWD weight ω_max for tiny targets
    nwd_constant_c: float = 12.8           # NWD normalization constant C

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
        notes="YOLO26n official baseline with default optimizer (MuSGD) on HIT-UAV 4cls",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        optimizer="auto",
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
        notes="Ablation +B: Zero-init gated residual bidirectional TCAD on P3-P5 baseline",
        model_cfg="yolo26n.pt",
        is_e2e_model=True,
        use_mod_b=True,
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
            flags.append("A")
        if cfg.use_mod_b:
            flags.append("B")
        if cfg.use_mod_c1:
            flags.append("C1")
        if cfg.use_mod_c2:
            flags.append("C2")
        flag_str = "+".join(flags) if flags else "baseline"
        lines.append(
            f"  - {name:<26} [{cfg.phase:<6}] model={cfg.model_cfg:<30} mods={flag_str:<10} ({cfg.notes})"
        )
    return "\n".join(lines)
