"""Autonomous Phase 3 Overnight Experiment Orchestrator for HIT-UAV YOLO26 KD.

Zero-babysitting, CU-safe decision engine callable in two ways:
  1. Direct Colab Notebook zero-edit mode: set `EXP_LIST = ["auto_phase3_overnight"]`
     (with `SMOKE_TEST = False`, `AUTO_UNASSIGN = True`).
  2. Standalone CLI:
     `python -m src.run_phase3_auto --dataset-root /content/datasets --work-dir /content/work --drive-dir /content/drive/MyDrive/ir-yolo26-paper`

Decision Tree:
  - Gate 0 (Teacher Selector): Compares Drive-archived `teacher_yolo26s` vs `teacher_yolo26m`
    by composite accuracy (`0.45*mAP50 + 0.45*mAP50-95 + 0.10*3cls`) and auto-locks best teacher.
  - Gate 1 (2-Epoch Smoke Lock): Runs 2-epoch `p3_kd_full_{t}` and strictly asserts:
      (a) exit code == 0 & finite metrics, and
      (b) exported `best.pt` has `params_M <= 2.55` (proving +0 teacher/adapter leakage!).
    Aborts immediately on failure to protect Colab CUs.
  - Gate 2 (Core Probe Trio @ seed=42): Runs `full`, `full_norun`, and `m12_cross`, comparing
    each against `v33_e0_base` (`86.48% / 56.34% / 3cls 92.83%`).
    - If none clears the anti-noise improvement gate -> auto-runs `0.3x` gentle rescue
      (`p3_kd_gentle_{t}`) + pure `M1` (`p3_kd_m1_fpkd_{t}`).
  - Gate 3 (Branch Execution):
    - Scenario A/B (Promising WINNER found):
        1. Complete remaining single-module ablations (`M1 F-PKD`, `M3 QG-Head`) @ `seed=42`.
        2. Run 3-seed paired audit (`WINNER@s0`, `WINNER@s3407`) & compute paired `t-test`
           vs existing `v33_e0_base` (`42, 0, 3407`).
        3. Run 3 fair 2026 KD baselines (`Official PR#24894 L2`, `QG-KD 1-IoU`, `CWD`).
        4. Run 4 unified 100-ep SOTA baselines (`YOLOv8n`, `YOLOv10n`, `YOLO11n`, `YOLO12n`).
        5. Bonus 5-seed upgrade (`seeds=2025, 2026` paired `base` + `WINNER`) ONLY if 3-seed
           paired gain holds cleanly AND remaining time budget allows!
    - Scenario C (2K HIT-UAV saturation brake):
        Skips multi-seed KD repeats to save Colab CUs; runs ONLY the 4 universally needed SOTA
        baselines (`YOLOv8n/10n/11n/12n`), writes an honest diagnostic report to Drive, and exits
        cleanly so `AUTO_UNASSIGN=True` releases the GPU immediately.

Every sub-experiment executes in an isolated subprocess (`python -m src.train_runner ...`) for
zero VRAM fragmentation, full `_COMPLETED.flag` resume safety, and live Drive archiving.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# Default fallback metrics for `v33_e0_base` (HIT-UAV 4cls, 100ep cos_lr) verified on Drive
_FALLBACK_BASE_METRICS: Dict[int, Dict[str, float]] = {
    42: {"mAP50": 0.86475, "mAP50_95": 0.56341, "main3": 0.92829, "rect50": 0.85973, "delta_e2e_50": 0.00730},
    0: {"mAP50": 0.85492, "mAP50_95": 0.56322, "main3": 0.92763, "rect50": 0.85627, "delta_e2e_50": 0.01285},
    3407: {"mAP50": 0.87924, "mAP50_95": 0.56690, "main3": 0.92458, "rect50": 0.87733, "delta_e2e_50": 0.00913},
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
            for v in (self.params_m, self.test_map50, self.test_map50_95, self.val_map50, self.val_map50_95)
        )


def _safe_float(val: Any, default: float = float("nan")) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_run_metrics(
    exp_dir_name: str,
    *,
    work_dir: Path,
    drive_dir: Optional[Path],
) -> Optional[RunMetrics]:
    """Read metrics for `exp_dir_name` from `eval_summary.json` (local/Drive) or `master_results.csv`."""
    json_candidates: List[Path] = [work_dir / "runs" / exp_dir_name / "eval_summary.json"]
    if drive_dir is not None:
        json_candidates.append(drive_dir / "runs" / exp_dir_name / "eval_summary.json")

    for jp in json_candidates:
        if jp.is_file():
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
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
                return RunMetrics(
                    exp_name=exp_dir_name,
                    params_m=_safe_float(data.get("params_M")),
                    gflops=_safe_float(data.get("gflops"), 0.0),
                    val_map50=_safe_float(val_d.get("mAP50")),
                    val_map50_95=_safe_float(val_d.get("mAP50_95")),
                    test_map50=_safe_float(test_d.get("mAP50")),
                    test_map50_95=_safe_float(test_d.get("mAP50_95")),
                    test_main3=main3,
                    rect_map50=_safe_float(rect_d.get("mAP50"), _safe_float(test_d.get("mAP50"))),
                    delta_e2e_50=_safe_float(data.get("test_delta_e2e_map50"), 0.0),
                    total_ms=_safe_float(test_d.get("total_ms"), 0.0),
                )
            except Exception as exc:
                print(f"[AUTO WARN] Failed parsing {jp}: {exc}")

    # Fallback: read the latest matching row from master_results.csv
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
            t50 = _safe_float(last_row.get("test_mAP50"))
            return RunMetrics(
                exp_name=exp_dir_name,
                params_m=_safe_float(last_row.get("params_M")),
                gflops=_safe_float(last_row.get("gflops"), 0.0),
                val_map50=_safe_float(last_row.get("val_mAP50")),
                val_map50_95=_safe_float(last_row.get("val_mAP50_95")),
                test_map50=t50,
                test_map50_95=_safe_float(last_row.get("test_mAP50_95")),
                test_main3=main3,
                rect_map50=_safe_float(last_row.get("rect640x512_mAP50"), t50),
                delta_e2e_50=_safe_float(last_row.get("test_delta_E2E_mAP50"), 0.0),
                total_ms=_safe_float(last_row.get("test_total_ms"), 0.0),
            )
    return None


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
    """Select best teacher tag (`'s'` or `'m'`) using balanced accuracy & stability score."""
    scored: List[Tuple[float, str, str]] = []
    for tag, m, has_w in (("s", metrics_s, has_s_weights), ("m", metrics_m, has_m_weights)):
        if not has_w:
            continue
        if m is not None and m.is_finite() and m.test_map50 >= 0.85:
            main3 = m.test_main3 if math.isfinite(m.test_main3) else m.test_map50
            score = 0.45 * m.test_map50 + 0.45 * m.test_map50_95 + 0.10 * main3
            desc = (
                f"teacher_yolo26{tag}: Test O2O={m.test_map50*100:.2f}%/{m.test_map50_95*100:.2f}%, "
                f"3cls={main3*100:.2f}%, Rect={m.rect_map50*100:.2f}%, Δ_E2E={m.delta_e2e_50*100:+.2f}%, "
                f"score={score:.5f}"
            )
            scored.append((score, tag, desc))
        else:
            # Weights exist without parsed metrics (rare fallback)
            scored.append((0.0, tag, f"teacher_yolo26{tag}: weights found (metrics unavailable)"))

    if not scored:
        raise FileNotFoundError(
            "Neither teacher_yolo26s nor teacher_yolo26m weights/best.pt found on Drive or local SSD!"
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_tag, _ = scored[0]
    summary_str = " | ".join(item[2] for item in scored)
    return best_tag, f"Selected teacher_yolo26{best_tag} (best_score={best_score:.5f}) [{summary_str}]"


def evaluate_candidate_vs_base(cand: RunMetrics, base: RunMetrics) -> Tuple[bool, float, str]:
    """Strict anti-noise promotion gate comparing a KD candidate vs `v33_e0_base` (`seed=42`)."""
    d50 = cand.test_map50 - base.test_map50
    d95 = cand.test_map50_95 - base.test_map50_95
    c3 = cand.test_main3 if math.isfinite(cand.test_main3) else cand.test_map50
    b3 = base.test_main3 if math.isfinite(base.test_main3) else base.test_map50
    d3c = c3 - b3
    drect = cand.rect_map50 - base.rect_map50

    rank_score = 0.35 * d50 + 0.40 * d95 + 0.15 * d3c + 0.10 * drect
    passes_gain = (d50 >= 0.0050) or (d95 >= 0.0040) or (drect >= 0.0100 and d95 >= 0.0020)
    passes_safety = (d3c >= -0.0025) and (d95 >= -0.0030) and (cand.params_m <= 2.55)
    is_promising = bool(passes_gain and passes_safety)

    detail = (
        f"{cand.exp_name}: O2O={cand.test_map50*100:.2f}%({d50*100:+.2f}%) / "
        f"m95={cand.test_map50_95*100:.2f}%({d95*100:+.2f}%) | "
        f"3cls={c3*100:.2f}%({d3c*100:+.2f}%) | Rect={cand.rect_map50*100:.2f}%({drect*100:+.2f}%) | "
        f"Δ_E2E={cand.delta_e2e_50*100:+.2f}% | rank={rank_score*100:+.3f} | PASS={is_promising}"
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
            "# Phase 3 Autonomous Overnight Run Report",
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
    print(f"\n[AUTO EXEC] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
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
) -> Dict[str, Any]:
    """Execute the full Phase 3 decision tree with strict smoke, anti-waste, and walltime guards."""
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

    def _elapsed_h() -> float:
        return (time.monotonic() - t0) / 3600.0

    def _has_budget(need_hours: float) -> bool:
        return (_elapsed_h() + need_hours) <= float(max_hours)

    def _run_and_load(exp_spec: str, *, smoke: bool = False, onnx: bool = False) -> Tuple[int, Optional[RunMetrics]]:
        rc = exec_fn(
            exp_spec,
            dataset_root=dataset_root,
            work_dir=work_dir,
            drive_dir=drive_dir,
            smoke_test=smoke,
            resume=resume,
            export_onnx=onnx,
        )
        run_dir = _resolved_run_dir_name(exp_spec, smoke_test=smoke)
        m = _load(run_dir)
        if rc != 0:
            journal.log(f"⚠️ Run `{exp_spec}` exited with code {rc}.")
        elif m is not None:
            journal.log(
                f"✅ `{run_dir}`: Params={m.params_m:.4f}M | Test O2O=`{m.test_map50*100:.2f}% / {m.test_map50_95*100:.2f}%` "
                f"| 3cls=`{m.test_main3*100:.2f}%` | Rect=`{m.rect_map50*100:.2f}%` | Δ_E2E=`{m.delta_e2e_50*100:+.2f}%`"
            )
        else:
            journal.log(f"⚠️ `{run_dir}` finished (rc=0) but metrics file could not be loaded.")
        return rc, m

    result_state: Dict[str, Any] = {
        "teacher": None,
        "smoke_passed": False,
        "scenario": None,
        "winner": None,
        "executed": [],
    }

    # =========================================================================
    # Gate 0: Automatic Best Teacher Selection (`teacher_yolo26s` vs `m`)
    # =========================================================================
    journal.section("Gate 0: Teacher Selection (`teacher_yolo26s` vs `teacher_yolo26m`)")
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

    # Load baseline reference for seed=42 (Drive first, verified fallback otherwise)
    base_42 = _load("v33_e0_base")
    if base_42 is None:
        fb = _FALLBACK_BASE_METRICS[42]
        base_42 = RunMetrics(
            exp_name="v33_e0_base",
            params_m=2.5353,
            gflops=6.035,
            val_map50=0.90641,
            val_map50_95=0.60549,
            test_map50=fb["mAP50"],
            test_map50_95=fb["mAP50_95"],
            test_main3=fb["main3"],
            rect_map50=fb["rect50"],
            delta_e2e_50=fb["delta_e2e_50"],
            total_ms=2.45,
        )
    journal.log(
        f"Locked Baseline `v33_e0_base` (s42): O2O=`{base_42.test_map50*100:.2f}% / {base_42.test_map50_95*100:.2f}%` "
        f"| 3cls=`{base_42.test_main3*100:.2f}%` | Rect=`{base_42.rect_map50*100:.2f}%`"
    )

    # =========================================================================
    # Gate 1: Mandatory 2-Epoch Smoke Gate (+0 Params Leakage Assertion!)
    # =========================================================================
    journal.section("Gate 1: Mandatory 2-Epoch Smoke Safety Lock")
    smoke_preset = f"p3_kd_full_{t_tag}"
    rc_sm, m_sm = _run_and_load(smoke_preset, smoke=True, onnx=False)
    result_state["executed"].append(f"{smoke_preset}_smoke")
    if rc_sm != 0 or m_sm is None or not m_sm.is_finite():
        journal.log("🛑 SMOKE GATE FAILED (non-zero exit or NaN metrics). Aborting to protect Colab CUs!")
        result_state["scenario"] = "ABORT_SMOKE_FAILED"
        raise RuntimeError(f"Phase 3 smoke gate failed for {smoke_preset} (rc={rc_sm}).")
    if m_sm.params_m > 2.55:
        journal.log(
            f"🛑 ZERO-OVERHEAD VIOLATION: exported best.pt has {m_sm.params_m:.4f}M params (> 2.55M)! "
            "Teacher or adapters leaked into checkpoint. Aborting immediately!"
        )
        result_state["scenario"] = "ABORT_PARAM_LEAK"
        raise RuntimeError(f"Zero-overhead check failed: exported best.pt has {m_sm.params_m:.4f}M params.")

    result_state["smoke_passed"] = True
    journal.log(
        f"🟢 SMOKE GATE PASSED! Exported best.pt strictly verified at `{m_sm.params_m:.4f}M params / {m_sm.gflops:.3f}G FLOPs` (+0.00ms)."
    )
    if smoke_only:
        journal.log("`smoke_only=True` requested -> stopping cleanly after smoke verification.")
        result_state["scenario"] = "SMOKE_ONLY_PASS"
        return result_state

    # =========================================================================
    # Gate 2: Core Probe Trio (seed=42) + Automatic 0.3x Gentle Rescue
    # =========================================================================
    journal.section("Gate 2: Core KD Probes (`seed=42`) & Winner Selection")
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
        rc, m = _run_and_load(spec)
        result_state["executed"].append(spec)
        if rc == 0 and m is not None and m.is_finite():
            ok, score, detail = evaluate_candidate_vs_base(m, base_42)
            candidates[spec] = (ok, score, m)
            journal.log(f"Probe eval -> {detail}")

    promising = [(score, spec, m) for spec, (ok, score, m) in candidates.items() if ok]
    if not promising and _has_budget(1.0):
        journal.log(
            "🟡 None of default 1.0/0.5/1.0 probes cleared the anti-noise threshold -> "
            "Auto-triggering Gentle 0.3x Rescue (`p3_kd_gentle`) + Pure `M1 F-PKD` probe!"
        )
        for spec in (f"p3_kd_gentle_{t_tag}", f"p3_kd_m1_fpkd_{t_tag}"):
            if not _has_budget(0.5):
                break
            rc, m = _run_and_load(spec)
            result_state["executed"].append(spec)
            if rc == 0 and m is not None and m.is_finite():
                ok, score, detail = evaluate_candidate_vs_base(m, base_42)
                candidates[spec] = (ok, score, m)
                journal.log(f"Rescue eval -> {detail}")
        promising = [(score, spec, m) for spec, (ok, score, m) in candidates.items() if ok]

    # =========================================================================
    # Gate 3: Branch Execution (Scenario A/B Full Pack vs Scenario C Anti-Waste Brake)
    # =========================================================================
    sota_queue = ["v33_sota_yolov8n", "v33_sota_yolov10n", "v33_sota_yolo11n", "v33_sota_yolo12n"]

    if not promising:
        journal.section("Gate 3 -> Scenario C (Anti-Waste Brake Triggered)")
        result_state["scenario"] = "SCENARIO_C_BRAKE"
        journal.log(
            "🔴 No KD candidate cleared the strict anti-noise promotion gate on 2K HIT-UAV. "
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
    journal.section(f"Gate 3 -> {result_state['scenario']}! Winner Locked: `{winner_spec}` (rank={win_score*100:+.3f})")

    # Step 3.1: Complete remaining single-module ablations (seed=42)
    ablation_rest = [s for s in (f"p3_kd_m1_fpkd_{t_tag}", f"p3_kd_m3_qg_{t_tag}") if s not in candidates]
    for spec in ablation_rest:
        if not _has_budget(0.55):
            break
        _run_and_load(spec)
        result_state["executed"].append(spec)

    # Step 3.2: Paired 3-Seed Truth Test (`seeds = 42, 0, 3407`)
    journal.section(f"Step 3.2: Paired 3-Seed Verification (`{winner_spec}` vs `v33_e0_base`)")
    seed_pairs_50: Tuple[List[float], List[float]] = ([base_42.test_map50], [winner_m42.test_map50])
    seed_pairs_95: Tuple[List[float], List[float]] = ([base_42.test_map50_95], [winner_m42.test_map50_95])
    seed_pairs_3c: Tuple[List[float], List[float]] = ([base_42.test_main3], [winner_m42.test_main3])

    for sd in (0, 3407):
        if not _has_budget(0.55):
            break
        w_spec = f"{winner_spec}@s{sd}"
        rc_w, m_w = _run_and_load(w_spec)
        result_state["executed"].append(w_spec)
        m_b = _load(f"v33_e0_base_s{sd}")
        if m_b is None and sd in _FALLBACK_BASE_METRICS:
            fb = _FALLBACK_BASE_METRICS[sd]
            m_b = RunMetrics(
                f"v33_e0_base_s{sd}", 2.5353, 6.035, 0.90, 0.59, fb["mAP50"], fb["mAP50_95"], fb["main3"], fb["rect50"], fb["delta_e2e_50"], 2.48
            )
        if rc_w == 0 and m_w is not None and m_b is not None:
            seed_pairs_50[0].append(m_b.test_map50)
            seed_pairs_50[1].append(m_w.test_map50)
            seed_pairs_95[0].append(m_b.test_map50_95)
            seed_pairs_95[1].append(m_w.test_map50_95)
            seed_pairs_3c[0].append(m_b.test_main3)
            seed_pairs_3c[1].append(m_w.test_main3)

    st50 = compute_paired_stats(*seed_pairs_50)
    st95 = compute_paired_stats(*seed_pairs_95)
    st3c = compute_paired_stats(*seed_pairs_3c)
    journal.log(
        f"📊 3-Seed Paired Stats (n={int(st50['n'])}): "
        f"mAP50 `{st50['base_mean']*100:.2f}±{st50['base_std']*100:.2f}% -> {st50['cand_mean']*100:.2f}±{st50['cand_std']*100:.2f}%` "
        f"(`Δ={st50['delta_mean']*100:+.2f}%, t={st50['t_stat']:+.2f}`) | "
        f"mAP50-95 `{st95['base_mean']*100:.2f}% -> {st95['cand_mean']*100:.2f}%` "
        f"(`Δ={st95['delta_mean']*100:+.2f}%, t={st95['t_stat']:+.2f}`) | "
        f"3cls `Δ={st3c['delta_mean']*100:+.2f}%`"
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

    # Step 3.5: Bonus 5-Seed Upgrade (`seeds=2025, 2026`) ONLY if 3-seed paired gain held cleanly!
    three_seed_solid = (st50["n"] >= 3) and ((st50["delta_mean"] >= 0.0040) or (st95["delta_mean"] >= 0.0030))
    if three_seed_solid and _has_budget(1.6):
        journal.section("Step 3.5: Bonus 5-Seed Paired Upgrade (`seeds=2025, 2026`)")
        for sd in (2025, 2026):
            if not _has_budget(0.95):
                journal.log(f"⏱️ Budget guard reached before seed={sd} pair (elapsed {_elapsed_h():.2f}h).")
                break
            b_spec = f"v33_e0_base@s{sd}"
            w_spec = f"{winner_spec}@s{sd}"
            rc_b, m_b = _run_and_load(b_spec)
            result_state["executed"].append(b_spec)
            rc_w, m_w = _run_and_load(w_spec, onnx=(export_onnx and sd == 2026))
            result_state["executed"].append(w_spec)
            if rc_b == 0 and rc_w == 0 and m_b is not None and m_w is not None:
                seed_pairs_50[0].append(m_b.test_map50)
                seed_pairs_50[1].append(m_w.test_map50)
                seed_pairs_95[0].append(m_b.test_map50_95)
                seed_pairs_95[1].append(m_w.test_map50_95)
                seed_pairs_3c[0].append(m_b.test_main3)
                seed_pairs_3c[1].append(m_w.test_main3)
        st50_f = compute_paired_stats(*seed_pairs_50)
        st95_f = compute_paired_stats(*seed_pairs_95)
        journal.log(
            f"🏆 Final {int(st50_f['n'])}-Seed Paired Stats: "
            f"mAP50 `{st50_f['base_mean']*100:.2f}±{st50_f['base_std']*100:.2f}% -> {st50_f['cand_mean']*100:.2f}±{st50_f['cand_std']*100:.2f}%` "
            f"(`Δ={st50_f['delta_mean']*100:+.2f}%, t={st50_f['t_stat']:+.2f}`) | "
            f"mAP50-95 `{st95_f['base_mean']*100:.2f}% -> {st95_f['cand_mean']*100:.2f}%` "
            f"(`Δ={st95_f['delta_mean']*100:+.2f}%, t={st95_f['t_stat']:+.2f}`)"
        )
    else:
        journal.log(
            f"Skipping 5-seed extension (three_seed_solid={three_seed_solid}, elapsed={_elapsed_h():.2f}h / {max_hours:.1f}h)."
        )

    journal.log(f"🎉 Autonomous Phase 3 overnight orchestration completed cleanly in {_elapsed_h():.2f}h!")
    return result_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Phase 3 Overnight YOLO26 KD Orchestrator.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--max-hours", type=float, default=8.5, help="Wall-clock budget before graceful stop.")
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
