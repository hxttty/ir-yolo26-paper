"""Autonomous Phase 3 Overnight Experiment Orchestrator for HIT-UAV YOLO26 KD.

Zero-babysitting, CU-safe, leak-free decision engine callable in two ways:
  1. Direct Colab Notebook zero-edit mode: set `EXP_LIST = ["auto_phase3_overnight"]`,
     `SMOKE_TEST = False`, `AUTO_UNASSIGN = True`.
  2. Standalone CLI:
     `python -m src.run_phase3_auto --dataset-root /content/datasets --work-dir /content/work --drive-dir /content/drive/MyDrive/ir-yolo26-paper`

Hardened against all adversarial audit edge cases:
  - STRICT Validation-only model/teacher/winner selection (ZERO test-set leakage into branching!).
  - Strict fail-fast on ANY non-zero subprocess exit or NaN/missing metrics (NEVER mislabels
    crashes as `SCENARIO_C_BRAKE`!).
  - Hard per-subprocess wall-clock timeout bounded by remaining `--max-hours` budget so a hung
    training or Drive I/O call forcefully exits in time for `AUTO_UNASSIGN=True`.
  - Gate 1 verifies: (a) formal batch size (`batch=32`) peak VRAM & finite metrics,
    (b) `kd_health.json` confirms `adapter_steps >= 1` & `total_kd_batches >= 1`, and
    (c) exported `best.pt` purity check (zero `kd_yolo26` / `NeckChannelAdapters` refs,
    native `YOLO26n` state_dict keys, and `params_M <= 2.55`).
  - Missing optional fields (`rect_map50`, `delta_e2e_50`) stay `NaN` (never silently replaced),
    and malformed/empty local JSON falls through cleanly to Drive JSON / `master_results.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# Verified `v33_e0_base` metrics on HIT-UAV 4cls (100ep cos_lr, Drive master_results.csv)
_FALLBACK_BASE_METRICS: Dict[int, Dict[str, float]] = {
    42: {
        "val_mAP50": 0.90641,
        "val_mAP50_95": 0.60549,
        "mAP50": 0.86475,
        "mAP50_95": 0.56341,
        "main3": 0.92829,
        "rect50": 0.85973,
        "delta_e2e_50": 0.00730,
    },
    0: {
        "val_mAP50": 0.89981,
        "val_mAP50_95": 0.59867,
        "mAP50": 0.85492,
        "mAP50_95": 0.56322,
        "main3": 0.92763,
        "rect50": 0.85627,
        "delta_e2e_50": 0.01285,
    },
    3407: {
        "val_mAP50": 0.90188,
        "val_mAP50_95": 0.58902,
        "mAP50": 0.87924,
        "mAP50_95": 0.56690,
        "main3": 0.92458,
        "rect50": 0.87733,
        "delta_e2e_50": 0.00913,
    },
}


@dataclass
class RunMetrics:
    exp_name: str
    params_m: float
    gflops: float
    val_map50: float
    val_map50_95: float
    test_map50: float
    test_map50_95: float
    test_main3: float
    rect_map50: float
    delta_e2e_50: float
    total_ms: float

    def is_finite(self) -> bool:
        return all(
            math.isfinite(v)
            for v in (self.params_m, self.val_map50, self.val_map50_95, self.test_map50, self.test_map50_95)
        )


def _safe_float(val: Any, default: float = float("nan")) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt_pct(val: float, signed: bool = False) -> str:
    if not math.isfinite(val):
        return "N/A"
    return f"{val*100:+.2f}%" if signed else f"{val*100:.2f}%"


def load_run_metrics(
    exp_dir_name: str,
    *,
    work_dir: Path,
    drive_dir: Optional[Path],
) -> Optional[RunMetrics]:
    """Read metrics for `exp_dir_name` from `eval_summary.json` (local/Drive) or `master_results.csv`.

    Never substitutes missing optional fields (`rect_map50`, `delta_e2e_50`) with fake values,
    and falls through to the next source if a JSON file is empty/malformed/non-finite.
    """
    json_candidates: List[Path] = [work_dir / "runs" / exp_dir_name / "eval_summary.json"]
    if drive_dir is not None:
        json_candidates.append(drive_dir / "runs" / exp_dir_name / "eval_summary.json")

    for jp in json_candidates:
        if not jp.is_file():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("evals"):
                continue
            is_e2e = bool(data.get("is_e2e_model", True))
            evals = data.get("evals", {}) or {}
            val_d = evals.get("val_o2o_640" if is_e2e else "val_nms_640", {}) or {}
            test_d = evals.get("test_o2o_640" if is_e2e else "test_nms_640", {}) or {}
            rect_d = evals.get("test_o2o_rect640x512" if is_e2e else "test_nms_rect640x512", {}) or {}
            main3 = _safe_float(data.get("test_3cls_mAP50"))
            if not math.isfinite(main3):
                pc = test_d.get("per_class_ap50", {}) or {}
                vals = [_safe_float(pc.get(c)) for c in ("Person", "Car", "Bicycle")]
                main3 = sum(vals) / 3.0 if all(math.isfinite(v) for v in vals) else float("nan")
            m = RunMetrics(
                exp_name=exp_dir_name,
                params_m=_safe_float(data.get("params_M")),
                gflops=_safe_float(data.get("gflops")),
                val_map50=_safe_float(val_d.get("mAP50")),
                val_map50_95=_safe_float(val_d.get("mAP50_95")),
                test_map50=_safe_float(test_d.get("mAP50")),
                test_map50_95=_safe_float(test_d.get("mAP50_95")),
                test_main3=main3,
                rect_map50=_safe_float(rect_d.get("mAP50")),
                delta_e2e_50=_safe_float(data.get("test_delta_e2e_map50")),
                total_ms=_safe_float(test_d.get("total_ms")),
            )
            if m.is_finite():
                return m
        except Exception as exc:
            print(f"[AUTO WARN] Failed parsing {jp}: {exc}")

    # Fallback: read the latest matching finite row from master_results.csv
    csv_candidates: List[Path] = []
    if drive_dir is not None:
        csv_candidates.append(drive_dir / "results" / "master_results.csv")
    csv_candidates.append(work_dir / "master_results.csv")

    for cp in csv_candidates:
        if not cp.is_file():
            continue
        last_row: Optional[Dict[str, str]] = None
        try:
            with cp.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("exp_name") == exp_dir_name:
                        last_row = row
        except Exception:
            continue
        if last_row is not None:
            p_ap = _safe_float(last_row.get("test_AP50_Person"))
            c_ap = _safe_float(last_row.get("test_AP50_Car"))
            b_ap = _safe_float(last_row.get("test_AP50_Bicycle"))
            main3 = (p_ap + c_ap + b_ap) / 3.0 if all(math.isfinite(v) for v in (p_ap, c_ap, b_ap)) else float("nan")
            m = RunMetrics(
                exp_name=exp_dir_name,
                params_m=_safe_float(last_row.get("params_M")),
                gflops=_safe_float(last_row.get("gflops")),
                val_map50=_safe_float(last_row.get("val_mAP50")),
                val_map50_95=_safe_float(last_row.get("val_mAP50_95")),
                test_map50=_safe_float(last_row.get("test_mAP50")),
                test_map50_95=_safe_float(last_row.get("test_mAP50_95")),
                test_main3=main3,
                rect_map50=_safe_float(last_row.get("rect640x512_mAP50")),
                delta_e2e_50=_safe_float(last_row.get("test_delta_E2E_mAP50")),
                total_ms=_safe_float(last_row.get("test_total_ms")),
            )
            if m.is_finite():
                return m
    return None


def load_kd_health(exp_dir_name: str, *, work_dir: Path, drive_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    cands = [work_dir / "runs" / exp_dir_name / "kd_health.json"]
    if drive_dir is not None:
        cands.append(drive_dir / "runs" / exp_dir_name / "kd_health.json")
    for p in cands:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


def _classification_outputs(head):
    for name in ("cv3", "one2one_cv3"):
        branches = getattr(head, name, None)
        if branches is not None:  # End-to-end fusion can remove the one-to-many branches.
            for i, branch in enumerate(branches):
                yield f"{name}.{i}", branch


def _native_student_references(detection_model, yaml, nc):
    from torch import nn

    for initial_nc in dict.fromkeys((nc, 80)):
        ref = detection_model(yaml, ch=3, nc=initial_nc, verbose=False)
        origin = "native"
        if initial_nc != nc:
            # Fine-tuning can retain the COCO classifier's intermediate width at every scale.
            for _, branch in _classification_outputs(ref.model[-1]):
                conv = branch[-1]
                branch[-1] = nn.Conv2d(conv.in_channels, nc, 1, bias=conv.bias is not None)
            origin = "COCO-adapted native"
        yield origin, ref
        yield f"fused {origin}", ref.fuse()


def verify_exported_student_purity(
    exp_dir_name: str,
    *,
    work_dir: Path,
    drive_dir: Optional[Path],
    max_params_m: float = 2.55,
    expected_yaml: str = "yolo26n.yaml",
    expected_nc: Optional[int] = None,
) -> Tuple[bool, str]:
    """Verify exported `best.pt` is 100% pure native `YOLO26n` (zero KD / adapter / teacher leakage)."""
    cands = [work_dir / "runs" / exp_dir_name / "weights" / "best.pt"]
    if drive_dir is not None:
        cands.append(drive_dir / "runs" / exp_dir_name / "weights" / "best.pt")
    best_pt = next((p for p in cands if p.is_file()), None)
    if best_pt is None:
        return False, f"Exported best.pt missing for {exp_dir_name}"

    # 1. Inspect PyTorch zip/pickle byte stream directly for forbidden KD module strings
    forbidden = (
        b"kd_yolo26",
        b"NeckChannelAdapters",
        b"YOLO26DistillCriterion",
        b"_FeatureCaptureHook",
        b"paired_rgbt_data",
    )
    try:
        if zipfile.is_zipfile(best_pt):
            with zipfile.ZipFile(best_pt, "r") as zf:
                for member in zf.namelist():
                    if member.endswith("data.pkl"):
                        raw = zf.read(member)
                        for tok in forbidden:
                            if tok in raw:
                                return False, f"Forbidden KD reference `{tok.decode()}` found in {best_pt.name}!"
    except Exception as exc:
        return False, f"Failed scanning {best_pt} archive: {exc}"

    # 2. If torch is available (always true on Colab), also verify exact param count, hook cleanliness & schema
    try:
        import torch

        ckpt = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        model_obj = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
        if model_obj is None:
            return False, f"No model/ema object inside {best_pt}"
        mod_origin = str(type(model_obj).__module__)
        if "src.modules" in mod_origin or "kd_yolo26" in mod_origin:
            return False, f"Non-native model class `{mod_origin}.{type(model_obj).__name__}` in {best_pt}"
        # Verify zero residual hooks serialized onto any submodule
        leaked_hooks = [
            name
            for name, m in model_obj.named_modules()
            if getattr(m, "_forward_hooks", None)
            or getattr(m, "_forward_pre_hooks", None)
            or getattr(m, "_backward_hooks", None)
            or getattr(m, "_full_backward_hooks", None)
        ]
        if leaked_hooks:
            return False, f"Residual PyTorch hooks found inside {best_pt}: {leaked_hooks[:5]}"
        sd = model_obj.state_dict()
        bad_keys = [k for k in sd.keys() if any(w in k.lower() for w in ("adapter", "teacher", "kd_", "aux_"))]
        if bad_keys:
            return False, f"Unexpected KD keys in best.pt state_dict: {bad_keys[:5]}"
        n_params_m = sum(p.numel() for p in model_obj.parameters()) / 1e6
        if n_params_m > max_params_m:
            return False, f"Exported best.pt has {n_params_m:.4f}M params (> {max_params_m:.2f}M)"

        # 3. Exact native state_dict key+shape schema comparison against fresh YOLO26n(nc)
        schema_note = ""
        if expected_yaml:
            try:
                from ultralytics.nn.tasks import DetectionModel

                nc = expected_nc if expected_nc is not None else getattr(model_obj, "nc", None)
                if nc is None and isinstance(getattr(model_obj, "names", None), dict):
                    nc = len(model_obj.names)
                if nc is None or int(nc) <= 0:
                    return False, f"Cannot verify native schema: missing valid nc on {best_pt}"
                nc = int(nc)
                outputs = list(_classification_outputs(model_obj.model[-1]))
                if not outputs:
                    return False, "Missing final classification convolutions"
                for name, branch in outputs:
                    conv = branch[-1]
                    if not isinstance(conv, torch.nn.Conv2d) or conv.out_channels != nc:
                        return False, f"Expected final {name} classification convolution with {nc} outputs"

                actual_schema = {k: tuple(v.shape) for k, v in sd.items()}
                closest_distance = float("inf")
                for variant, ref in _native_student_references(DetectionModel, expected_yaml, nc):
                    schema = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
                    if actual_schema == schema:
                        schema_note = f", exact {variant} {expected_yaml}(nc={nc}) schema matched"
                        break
                    extra = actual_schema.keys() - schema.keys()
                    missing = schema.keys() - actual_schema.keys()
                    wrong_shapes = [k for k in actual_schema.keys() & schema.keys() if actual_schema[k] != schema[k]]
                    distance = len(extra) + len(missing) + len(wrong_shapes)
                    if distance < closest_distance:
                        closest_distance = distance
                        shape_diff = [(k, actual_schema[k], schema[k]) for k in sorted(wrong_shapes)[:5]]
                        mismatch = (
                            f"State_dict schema mismatch vs {variant} {expected_yaml}(nc={nc}): "
                            f"extra={sorted(extra)[:5]}, missing={sorted(missing)[:5]}, shape_diff={shape_diff}"
                        )
                else:
                    return False, mismatch
            except ImportError as exc:
                if expected_nc is not None:
                    return False, f"Native schema verification required nc={expected_nc}, but Ultralytics unavailable: {exc}"
            except Exception as exc:
                return False, f"Native schema verification failed against {expected_yaml}: {exc}"

        return True, (
            f"Pure native YOLO26n verified ({n_params_m:.4f}M params, {len(sd)} state_dict tensors, "
            f"0 KD refs, 0 hooks{schema_note})"
        )
    except ImportError:
        return True, f"Pickle byte scan clean (0 KD refs in {best_pt.name})"
    except Exception as exc:
        return False, f"torch.load purity verification failed on {best_pt}: {exc}"


def teacher_weights_exist(run_name: str, *, work_dir: Path, drive_dir: Optional[Path]) -> bool:
    cands = [work_dir / "runs" / run_name / "weights" / "best.pt"]
    if drive_dir is not None:
        cands.append(drive_dir / "runs" / run_name / "weights" / "best.pt")
    return any(p.is_file() for p in cands)


def select_best_teacher(
    metrics_s: Optional[RunMetrics],
    metrics_m: Optional[RunMetrics],
    *,
    has_s_weights: bool,
    has_m_weights: bool,
) -> Tuple[str, str]:
    """Select best teacher tag (`'s'` or `'m'`) STRICTLY on Validation metrics (zero test leakage)."""
    scored: List[Tuple[float, str, str]] = []
    for tag, m, has_w in (("s", metrics_s, has_s_weights), ("m", metrics_m, has_m_weights)):
        if not has_w:
            continue
        if m is not None and m.is_finite() and m.val_map50 >= 0.85:
            score = 0.35 * m.val_map50 + 0.65 * m.val_map50_95
            desc = (
                f"teacher_yolo26{tag}: Val={_fmt_pct(m.val_map50)}/{_fmt_pct(m.val_map50_95)} "
                f"(val_score={score:.5f}; passive Test={_fmt_pct(m.test_map50)}/{_fmt_pct(m.test_map50_95)})"
            )
            scored.append((score, tag, desc))
        else:
            scored.append((0.0, tag, f"teacher_yolo26{tag}: weights found (val metrics unavailable)"))

    if not scored:
        raise FileNotFoundError(
            "Neither teacher_yolo26s nor teacher_yolo26m weights/best.pt found on Drive or local SSD!"
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_tag, _ = scored[0]
    summary_str = " | ".join(item[2] for item in scored)
    return best_tag, f"Selected teacher_yolo26{best_tag} by Validation (val_score={best_score:.5f}) [{summary_str}]"


def evaluate_candidate_vs_base(cand: RunMetrics, base: RunMetrics) -> Tuple[bool, float, str]:
    """STRICT Validation-only promotion gate vs `v33_e0_base` (zero test-set leakage!)."""
    dv50 = cand.val_map50 - base.val_map50
    dv95 = cand.val_map50_95 - base.val_map50_95

    rank_score = 0.35 * dv50 + 0.65 * dv95
    passes_gain = (dv50 >= 0.0030) or (dv95 >= 0.0030)
    passes_safety = (dv95 >= -0.0030) and (dv50 >= -0.0050) and (cand.params_m <= 2.55)
    is_promising = bool(passes_gain and passes_safety)

    detail = (
        f"{cand.exp_name}: Val={_fmt_pct(cand.val_map50)}({_fmt_pct(dv50, True)}) / "
        f"{_fmt_pct(cand.val_map50_95)}({_fmt_pct(dv95, True)}) | val_rank={rank_score*100:+.3f} | "
        f"PASS={is_promising} [passive Test={_fmt_pct(cand.test_map50)}/{_fmt_pct(cand.test_map50_95)}, "
        f"3cls={_fmt_pct(cand.test_main3)}, Δ_E2E={_fmt_pct(cand.delta_e2e_50, True)}]"
    )
    return is_promising, rank_score, detail


def compute_paired_stats(
    base_vals: Sequence[float],
    cand_vals: Sequence[float],
) -> Dict[str, float]:
    """Compute paired mean, sample std, mean delta, and Student t-statistic."""
    n = min(len(base_vals), len(cand_vals))
    if n == 0:
        return {"n": 0, "base_mean": float("nan"), "cand_mean": float("nan"), "delta_mean": float("nan"), "t_stat": 0.0}
    b = [float(x) for x in base_vals[:n]]
    c = [float(x) for x in cand_vals[:n]]
    diffs = [ci - bi for bi, ci in zip(b, c)]
    b_mean = sum(b) / n
    c_mean = sum(c) / n
    d_mean = sum(diffs) / n
    if n >= 2:
        b_std = math.sqrt(sum((x - b_mean) ** 2 for x in b) / (n - 1))
        c_std = math.sqrt(sum((x - c_mean) ** 2 for x in c) / (n - 1))
        d_std = math.sqrt(sum((x - d_mean) ** 2 for x in diffs) / (n - 1))
        t_stat = (d_mean / (d_std / math.sqrt(n))) if d_std > 1e-12 else (math.copysign(99.0, d_mean) if d_mean != 0 else 0.0)
    else:
        b_std = c_std = d_std = 0.0
        t_stat = 0.0
    return {
        "n": float(n),
        "base_mean": b_mean,
        "base_std": b_std,
        "cand_mean": c_mean,
        "cand_std": c_std,
        "delta_mean": d_mean,
        "delta_std": d_std,
        "t_stat": t_stat,
    }


def _resolved_run_dir_name(exp_spec: str, *, smoke_test: bool = False) -> str:
    """Match `configs.experiments.get_experiment_config` naming exactly."""
    base = exp_spec
    seed = 42
    if "@s" in exp_spec:
        left, right = exp_spec.rsplit("@s", 1)
        if right.isdigit():
            base = left
            seed = int(right)
    name = f"{base}_s{seed}" if seed != 42 else base
    if smoke_test:
        name = f"{name}_smoke"
    return name


class _ReportJournal:
    """Persistent Markdown decision log mirrored live to Google Drive."""

    def __init__(self, work_dir: Path, drive_dir: Optional[Path]) -> None:
        self._paths: List[Path] = [work_dir / "logs" / "auto_phase3_report.md"]
        if drive_dir is not None:
            self._paths.append(drive_dir / "logs" / "auto_phase3_report.md")
        self._lines: List[str] = [
            "# Phase 3 Autonomous Overnight Run Report (Val-Only Selection)",
            f"- Started: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
        ]
        self.flush()

    def log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[AUTO {stamp}] {msg}", flush=True)
        self._lines.append(f"- `{stamp}` {msg}")
        self.flush()

    def section(self, title: str) -> None:
        print("\n" + "=" * 72 + f"\n[AUTO] {title}\n" + "=" * 72, flush=True)
        self._lines.extend(["", f"## {title}", ""])
        self.flush()

    def flush(self) -> None:
        text = "\n".join(self._lines) + "\n"
        for p in self._paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            except Exception:
                pass


def _default_subprocess_runner(
    exp_spec: str,
    *,
    dataset_root: Path,
    work_dir: Path,
    drive_dir: Optional[Path],
    smoke_test: bool,
    resume: bool,
    export_onnx: bool,
    timeout_sec: Optional[float] = None,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.train_runner",
        "--exp",
        exp_spec,
        "--dataset-root",
        str(dataset_root),
        "--work-dir",
        str(work_dir),
    ]
    if drive_dir is not None:
        cmd += ["--drive-dir", str(drive_dir)]
    if smoke_test:
        cmd.append("--smoke-test")
    if not resume:
        cmd.append("--no-resume")
    if export_onnx:
        cmd.append("--export-onnx")
    print(f"\n[AUTO EXEC] {' '.join(cmd)} (timeout={timeout_sec:.0f}s)" if timeout_sec else f"\n[AUTO EXEC] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, timeout=timeout_sec)
    return int(proc.returncode)


def run_overnight_orchestrator(
    *,
    dataset_root: Path,
    work_dir: Path,
    drive_dir: Optional[Path],
    resume: bool = True,
    smoke_only: bool = False,
    export_onnx: bool = False,
    max_hours: float = 8.5,
    force_teacher: Optional[str] = None,
    runner_fn: Optional[Callable[..., int]] = None,
    metrics_loader: Optional[Callable[[str], Optional[RunMetrics]]] = None,
    teacher_exists_fn: Optional[Callable[[str], bool]] = None,
    health_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    purity_checker: Optional[Callable[[str], Tuple[bool, str]]] = None,
) -> Dict[str, Any]:
    """Execute the full Phase 3 Val-only decision tree with fail-fast & hard walltime guards."""
    t0 = time.monotonic()
    journal = _ReportJournal(work_dir, drive_dir)
    exec_fn = runner_fn or _default_subprocess_runner

    def _load(name: str) -> Optional[RunMetrics]:
        if metrics_loader is not None:
            return metrics_loader(name)
        return load_run_metrics(name, work_dir=work_dir, drive_dir=drive_dir)

    def _has_teacher(run_name: str) -> bool:
        if teacher_exists_fn is not None:
            return teacher_exists_fn(run_name)
        return teacher_weights_exist(run_name, work_dir=work_dir, drive_dir=drive_dir)

    def _load_health(run_dir: str) -> Optional[Dict[str, Any]]:
        if health_loader is not None:
            return health_loader(run_dir)
        return load_kd_health(run_dir, work_dir=work_dir, drive_dir=drive_dir)

    def _check_purity(run_dir: str) -> Tuple[bool, str]:
        if purity_checker is not None:
            return purity_checker(run_dir)
        return verify_exported_student_purity(run_dir, work_dir=work_dir, drive_dir=drive_dir)

    def _elapsed_h() -> float:
        return (time.monotonic() - t0) / 3600.0

    def _remaining_sec() -> float:
        return max(float(max_hours) * 3600.0 - (time.monotonic() - t0), 0.0)

    def _has_budget(need_hours: float) -> bool:
        return (_elapsed_h() + need_hours) <= float(max_hours)

    def _run_and_load(exp_spec: str, *, smoke: bool = False, onnx: bool = False) -> RunMetrics:
        rem = _remaining_sec()
        if rem < 120.0:
            raise TimeoutError(f"Hard walltime budget exhausted before `{exp_spec}` ({_elapsed_h():.2f}h / {max_hours:.1f}h).")
        rc = exec_fn(
            exp_spec,
            dataset_root=dataset_root,
            work_dir=work_dir,
            drive_dir=drive_dir,
            smoke_test=smoke,
            resume=resume,
            export_onnx=onnx,
            timeout_sec=rem,
        )
        run_dir = _resolved_run_dir_name(exp_spec, smoke_test=smoke)
        if rc != 0:
            journal.log(f"🛑 FATAL: Sub-experiment `{exp_spec}` failed with exit code {rc}!")
            raise RuntimeError(f"Sub-experiment `{exp_spec}` exited with code {rc}.")
        m = _load(run_dir)
        if m is None or not m.is_finite():
            journal.log(f"🛑 FATAL: `{run_dir}` exited 0 but required Val/Test metrics are missing or non-finite!")
            raise RuntimeError(f"Missing or non-finite required metrics for `{run_dir}`.")
        journal.log(
            f"✅ `{run_dir}`: Params={m.params_m:.4f}M | Val=`{_fmt_pct(m.val_map50)} / {_fmt_pct(m.val_map50_95)}` "
            f"| Test O2O=`{_fmt_pct(m.test_map50)} / {_fmt_pct(m.test_map50_95)}` "
            f"| 3cls=`{_fmt_pct(m.test_main3)}` | Δ_E2E=`{_fmt_pct(m.delta_e2e_50, True)}`"
        )
        return m

    result_state: Dict[str, Any] = {
        "teacher": None,
        "smoke_passed": False,
        "scenario": None,
        "winner": None,
        "executed": [],
    }

    # =========================================================================
    # Gate 0: Automatic Best Teacher Selection (STRICT Validation-Only)
    # =========================================================================
    journal.section("Gate 0: Teacher Selection (`teacher_yolo26s` vs `teacher_yolo26m`, Val-Only)")
    if force_teacher in ("s", "m"):
        t_tag = force_teacher
        journal.log(f"Using forced teacher tag: `teacher_yolo26{t_tag}`")
    else:
        ms = _load("teacher_yolo26s")
        mm = _load("teacher_yolo26m")
        t_tag, reason = select_best_teacher(
            ms,
            mm,
            has_s_weights=_has_teacher("teacher_yolo26s"),
            has_m_weights=_has_teacher("teacher_yolo26m"),
        )
        journal.log(reason)
    result_state["teacher"] = f"teacher_yolo26{t_tag}"

    # Load baseline reference for seed=42
    base_42 = _load("v33_e0_base")
    if base_42 is None:
        fb = _FALLBACK_BASE_METRICS[42]
        base_42 = RunMetrics(
            exp_name="v33_e0_base",
            params_m=2.5353,
            gflops=6.035,
            val_map50=fb["val_mAP50"],
            val_map50_95=fb["val_mAP50_95"],
            test_map50=fb["mAP50"],
            test_map50_95=fb["mAP50_95"],
            test_main3=fb["main3"],
            rect_map50=fb["rect50"],
            delta_e2e_50=fb["delta_e2e_50"],
            total_ms=2.45,
        )
    journal.log(
        f"Locked Baseline `v33_e0_base` (s42): Val=`{_fmt_pct(base_42.val_map50)} / {_fmt_pct(base_42.val_map50_95)}` "
        f"[passive Test=`{_fmt_pct(base_42.test_map50)} / {_fmt_pct(base_42.test_map50_95)}`, 3cls=`{_fmt_pct(base_42.test_main3)}`]"
    )

    # =========================================================================
    # Gate 1: Mandatory 2-Epoch Smoke Gate (Formal Batch VRAM + Adapter Step + Purity Lock)
    # =========================================================================
    journal.section("Gate 1: Mandatory 2-Epoch Smoke Lock (Formal Batch + Adapter Update + Purity)")
    smoke_preset = f"p3_kd_full_{t_tag}"
    smoke_dir = _resolved_run_dir_name(smoke_preset, smoke_test=True)
    m_sm = _run_and_load(smoke_preset, smoke=True, onnx=False)
    result_state["executed"].append(smoke_dir)

    if m_sm.params_m > 2.55:
        journal.log(f"🛑 ZERO-OVERHEAD VIOLATION: exported best.pt has {m_sm.params_m:.4f}M params (> 2.55M)!")
        result_state["scenario"] = "ABORT_PARAM_LEAK"
        raise RuntimeError(f"Zero-overhead check failed: exported best.pt has {m_sm.params_m:.4f}M params.")

    health = _load_health(smoke_dir)
    if health is None or int(health.get("adapter_steps", 0)) < 1 or int(health.get("total_kd_batches", 0)) < 1:
        journal.log(f"🛑 KD HEALTH CHECK FAILED: kd_health.json={health} (adapters did not step!).")
        result_state["scenario"] = "ABORT_KD_INACTIVE"
        raise RuntimeError(f"Gate 1 KD health check failed for `{smoke_dir}`: {health}")

    pure_ok, pure_msg = _check_purity(smoke_dir)
    if not pure_ok:
        journal.log(f"🛑 CHECKPOINT PURITY VIOLATION: {pure_msg}")
        result_state["scenario"] = "ABORT_PURITY_FAIL"
        raise RuntimeError(f"Gate 1 checkpoint purity failed: {pure_msg}")

    result_state["smoke_passed"] = True
    journal.log(
        f"🟢 SMOKE GATE PASSED! {pure_msg} | adapter_steps={health.get('adapter_steps')}, "
        f"kd_batches={health.get('total_kd_batches')}, peak_vram={health.get('peak_vram_gb', 0.0):.2f} GB"
    )
    if smoke_only:
        journal.log("`smoke_only=True` requested -> stopping cleanly after smoke verification.")
        result_state["scenario"] = "SMOKE_ONLY_PASS"
        return result_state

    # =========================================================================
    # Gate 2: Core Probe Trio (`seed=42`, Val-Only Promotion) + 0.3x Gentle Rescue
    # =========================================================================
    journal.section("Gate 2: Core KD Probes (`seed=42`, Val-Only Winner Selection)")
    probe_specs = [
        f"p3_kd_full_{t_tag}",
        f"p3_kd_full_norun_{t_tag}",
        f"p3_kd_m12_cross_{t_tag}",
    ]
    candidates: Dict[str, Tuple[bool, float, RunMetrics]] = {}
    for spec in probe_specs:
        if not _has_budget(0.5):
            journal.log(f"⏱️ Walltime budget reached ({_elapsed_h():.2f}h / {max_hours:.1f}h) during Gate 2.")
            break
        m = _run_and_load(spec)
        result_state["executed"].append(spec)
        ok, score, detail = evaluate_candidate_vs_base(m, base_42)
        candidates[spec] = (ok, score, m)
        journal.log(f"Probe eval -> {detail}")

    promising = [(score, spec, m) for spec, (ok, score, m) in candidates.items() if ok]
    if not promising and _has_budget(1.0):
        journal.log(
            "🟡 None of default 1.0/0.5/1.0 probes cleared Val promotion threshold -> "
            "Auto-triggering Gentle 0.3x Rescue (`p3_kd_gentle`) + Pure `M1 F-PKD` probe!"
        )
        for spec in (f"p3_kd_gentle_{t_tag}", f"p3_kd_m1_fpkd_{t_tag}"):
            if not _has_budget(0.5):
                break
            m = _run_and_load(spec)
            result_state["executed"].append(spec)
            ok, score, detail = evaluate_candidate_vs_base(m, base_42)
            candidates[spec] = (ok, score, m)
            journal.log(f"Rescue eval -> {detail}")
        promising = [(score, spec, m) for spec, (ok, score, m) in candidates.items() if ok]

    # =========================================================================
    # Gate 3: Branch Execution (Scenario A/B Full Pack vs Scenario C Anti-Waste Brake)
    # =========================================================================
    sota_queue = ["v33_sota_yolov8n", "v33_sota_yolov10n", "v33_sota_yolo11n", "v33_sota_yolo12n"]

    if not promising:
        journal.section("Gate 3 -> Scenario C (Anti-Waste Brake Triggered on Val)")
        result_state["scenario"] = "SCENARIO_C_BRAKE"
        journal.log(
            "🔴 All completed KD candidates fell below the Val promotion threshold on 2K HIT-UAV. "
            "Braking multi-seed KD repeats to save Colab CUs; running ONLY the 4 SOTA baselines."
        )
        for spec in sota_queue:
            if not _has_budget(0.45):
                journal.log(f"⏱️ Stopping before `{spec}` to stay safely inside {max_hours:.1f}h budget.")
                break
            _run_and_load(spec)
            result_state["executed"].append(spec)
        journal.log(
            f"🏁 Finished Scenario C cleanly in {_elapsed_h():.2f}h. "
            "Next recommended move: pivot main firepower to DroneVehicle (18K aligned RGB+IR)."
        )
        return result_state

    promising.sort(key=lambda x: x[0], reverse=True)
    win_score, winner_spec, winner_m42 = promising[0]
    result_state["winner"] = winner_spec
    result_state["scenario"] = (
        "SCENARIO_B_RESCUED" if winner_spec in (f"p3_kd_gentle_{t_tag}", f"p3_kd_m1_fpkd_{t_tag}") else "SCENARIO_A_WIN"
    )
    journal.section(f"Gate 3 -> {result_state['scenario']}! Winner Locked by Val: `{winner_spec}` (val_rank={win_score*100:+.3f})")

    # Step 3.1: Complete remaining single-module ablations (seed=42)
    ablation_rest = [s for s in (f"p3_kd_m1_fpkd_{t_tag}", f"p3_kd_m3_qg_{t_tag}") if s not in candidates]
    for spec in ablation_rest:
        if not _has_budget(0.55):
            break
        _run_and_load(spec)
        result_state["executed"].append(spec)

    # Step 3.2: Paired 3-Seed Truth Test (`seeds = 42, 0, 3407`)
    journal.section(f"Step 3.2: Paired 3-Seed Verification (`{winner_spec}` vs `v33_e0_base`)")
    val_pairs_50: Tuple[List[float], List[float]] = ([base_42.val_map50], [winner_m42.val_map50])
    val_pairs_95: Tuple[List[float], List[float]] = ([base_42.val_map50_95], [winner_m42.val_map50_95])
    test_pairs_50: Tuple[List[float], List[float]] = ([base_42.test_map50], [winner_m42.test_map50])
    test_pairs_95: Tuple[List[float], List[float]] = ([base_42.test_map50_95], [winner_m42.test_map50_95])
    test_pairs_3c: Tuple[List[float], List[float]] = ([base_42.test_main3], [winner_m42.test_main3])

    for sd in (0, 3407):
        if not _has_budget(0.55):
            break
        w_spec = f"{winner_spec}@s{sd}"
        m_w = _run_and_load(w_spec)
        result_state["executed"].append(w_spec)
        m_b = _load(f"v33_e0_base_s{sd}")
        if m_b is None and sd in _FALLBACK_BASE_METRICS:
            fb = _FALLBACK_BASE_METRICS[sd]
            m_b = RunMetrics(
                f"v33_e0_base_s{sd}",
                2.5353,
                6.035,
                fb["val_mAP50"],
                fb["val_mAP50_95"],
                fb["mAP50"],
                fb["mAP50_95"],
                fb["main3"],
                fb["rect50"],
                fb["delta_e2e_50"],
                2.48,
            )
        if m_b is not None:
            val_pairs_50[0].append(m_b.val_map50)
            val_pairs_50[1].append(m_w.val_map50)
            val_pairs_95[0].append(m_b.val_map50_95)
            val_pairs_95[1].append(m_w.val_map50_95)
            test_pairs_50[0].append(m_b.test_map50)
            test_pairs_50[1].append(m_w.test_map50)
            test_pairs_95[0].append(m_b.test_map50_95)
            test_pairs_95[1].append(m_w.test_map50_95)
            if math.isfinite(m_b.test_main3) and math.isfinite(m_w.test_main3):
                test_pairs_3c[0].append(m_b.test_main3)
                test_pairs_3c[1].append(m_w.test_main3)

    sv50 = compute_paired_stats(*val_pairs_50)
    sv95 = compute_paired_stats(*val_pairs_95)
    st50 = compute_paired_stats(*test_pairs_50)
    st95 = compute_paired_stats(*test_pairs_95)
    st3c = compute_paired_stats(*test_pairs_3c)
    journal.log(
        f"📊 3-Seed Paired Stats (n={int(sv50['n'])}): "
        f"Val Δ50=`{_fmt_pct(sv50['delta_mean'], True)}` / Δ95=`{_fmt_pct(sv95['delta_mean'], True)}` | "
        f"[Passive Test] mAP50 `{st50['base_mean']*100:.2f}±{st50['base_std']*100:.2f}% -> {st50['cand_mean']*100:.2f}±{st50['cand_std']*100:.2f}%` "
        f"(`Δ={st50['delta_mean']*100:+.2f}%, t={st50['t_stat']:+.2f}`), "
        f"mAP50-95 `Δ={st95['delta_mean']*100:+.2f}% (t={st95['t_stat']:+.2f})`, 3cls `Δ={_fmt_pct(st3c['delta_mean'], True)}`"
    )

    # Step 3.3: Fair 2026 Literature KD Baselines (`seed=42`)
    journal.section("Step 3.3: 2026 Literature KD Baselines (`PR #24894 L2`, `QG-KD 1-IoU`, `CWD`)")
    for spec in (f"p3_kd_base_l2_{t_tag}", f"p3_kd_base_qg_iou_{t_tag}", f"p3_kd_base_cwd_{t_tag}"):
        if not _has_budget(0.55):
            journal.log(f"⏱️ Budget guard: skipping `{spec}` (elapsed {_elapsed_h():.2f}h).")
            break
        _run_and_load(spec)
        result_state["executed"].append(spec)

    # Step 3.4: Unified 100-ep SOTA Baselines (`seed=42`)
    journal.section("Step 3.4: Unified 100-ep SOTA Baselines (`YOLOv8n/10n/11n/12n`)")
    for spec in sota_queue:
        if not _has_budget(0.45):
            journal.log(f"⏱️ Budget guard: skipping `{spec}` (elapsed {_elapsed_h():.2f}h).")
            break
        _run_and_load(spec)
        result_state["executed"].append(spec)

    # Step 3.5: Bonus 5-Seed Upgrade (`seeds=2025, 2026`) gated STRICTLY on 3-Seed Val gain + time budget!
    three_seed_val_solid = (sv50["n"] >= 3) and ((sv50["delta_mean"] >= 0.0025) or (sv95["delta_mean"] >= 0.0025))
    if three_seed_val_solid and _has_budget(1.6):
        journal.section("Step 3.5: Bonus 5-Seed Paired Upgrade (`seeds=2025, 2026`)")
        for sd in (2025, 2026):
            if not _has_budget(0.95):
                journal.log(f"⏱️ Budget guard reached before seed={sd} pair (elapsed {_elapsed_h():.2f}h).")
                break
            b_spec = f"v33_e0_base@s{sd}"
            w_spec = f"{winner_spec}@s{sd}"
            m_b = _run_and_load(b_spec)
            result_state["executed"].append(b_spec)
            m_w = _run_and_load(w_spec, onnx=(export_onnx and sd == 2026))
            result_state["executed"].append(w_spec)
            val_pairs_50[0].append(m_b.val_map50)
            val_pairs_50[1].append(m_w.val_map50)
            val_pairs_95[0].append(m_b.val_map50_95)
            val_pairs_95[1].append(m_w.val_map50_95)
            test_pairs_50[0].append(m_b.test_map50)
            test_pairs_50[1].append(m_w.test_map50)
            test_pairs_95[0].append(m_b.test_map50_95)
            test_pairs_95[1].append(m_w.test_map50_95)
        st50_f = compute_paired_stats(*test_pairs_50)
        st95_f = compute_paired_stats(*test_pairs_95)
        journal.log(
            f"🏆 Final {int(st50_f['n'])}-Seed Passive Test Paired Stats: "
            f"mAP50 `{st50_f['base_mean']*100:.2f}±{st50_f['base_std']*100:.2f}% -> {st50_f['cand_mean']*100:.2f}±{st50_f['cand_std']*100:.2f}%` "
            f"(`Δ={st50_f['delta_mean']*100:+.2f}%, t={st50_f['t_stat']:+.2f}`) | "
            f"mAP50-95 `{st95_f['base_mean']*100:.2f}% -> {st95_f['cand_mean']*100:.2f}%` "
            f"(`Δ={st95_f['delta_mean']*100:+.2f}%, t={st95_f['t_stat']:+.2f}`)"
        )
    else:
        journal.log(
            f"Skipping 5-seed extension (three_seed_val_solid={three_seed_val_solid}, elapsed={_elapsed_h():.2f}h / {max_hours:.1f}h)."
        )

    journal.log(f"🎉 Autonomous Phase 3 overnight orchestration completed cleanly in {_elapsed_h():.2f}h!")
    return result_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Phase 3 Overnight YOLO26 KD Orchestrator.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--max-hours", type=float, default=8.5, help="Hard wall-clock budget before graceful stop.")
    parser.add_argument("--force-teacher", type=str, choices=["s", "m"], default=None)
    parser.add_argument("--smoke-only", action="store_true", help="Stop right after Gate 1 smoke test.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    run_overnight_orchestrator(
        dataset_root=args.dataset_root,
        work_dir=args.work_dir,
        drive_dir=args.drive_dir,
        resume=not args.no_resume,
        smoke_only=args.smoke_only,
        export_onnx=args.export_onnx,
        max_hours=args.max_hours,
        force_teacher=args.force_teacher,
    )


if __name__ == "__main__":
    main()
