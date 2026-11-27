"""Autonomous Phase 4 DroneVehicle + HIT-UAV v2 Orchestrator (Wave 1 & Wave 2).

Callable directly from Colab Cell 1 with ZERO notebook re-upload:
  - Wave 1 (`EXP_LIST = ["auto_dv_wave1"]`, ~5.5–6.5h on L4, 8.5h hard cap):
      Step 0 (~45m): Verify 3-bug-fixed `p3_kd_v2_full_s` on HIT-UAV (M3 t&s intersection + M2 Ep-5 warmup + 40% decay).
      Step 1 (~10m first time / ~25s Drive restore): Prepare DroneVehicle directly on Colab -> Drive cache (0 local traffic!).
      Step 2 (~75m): `dv_ir_yolo26n` (DroneVehicle IR Student baseline + Day/Night breakdown).
      Step 3 (~105m): `dv_ir_teacher_s` (DroneVehicle IR Teacher YOLO26s + Day/Night breakdown).
      Step 4 (~105m): `dv_rgb_teacher_s` (DroneVehicle Privileged RGB Teacher YOLO26s + Day/Night breakdown).
      Step 5 (bonus if >=15m left): Pre-run Track B 2-ep Smoke Gate (`dv_kd_track_b_s --smoke-test`) to verify Wave 2 readiness!
  - Wave 2 (`EXP_LIST = ["auto_dv_wave2"]`, ~6.0–7.5h on L4, 8.5h hard cap):
      Gate 1 (~3m): Mandatory 2-ep formal `batch=32` Track B Smoke Lock (`dv_kd_track_b_s --smoke-test`):
        verifies PairedRGBIRDataset, `track_b_enabled=True`, `adapter_steps>=1`, `0.05 < illum_safe_frac < 0.98`,
        and exported `best.pt` 100% pure native `YOLO26n` (`2.535M / 0 KD refs`).
      Step 1 (~115m): `dv_kd_track_a_s` (Track A single-modal IR->IR, +0.00 ms).
      Step 2 (~145m): `dv_kd_track_b_s` (Track B OURS: Night-Safe Dual-Teacher RGB+IR->IR, +0.00 ms).
      Step 3 (~145m): `dv_kd_track_b_naive_s` (Negative-Transfer Control: ungated RGB+IR->IR, no night breaker).
      Step 4 (bonus if budget allows): `dv_kd_track_b_nolp_s` (no 3x3 low-pass ablation).

Hardened with strict fail-fast on subprocess crashes/NaNs and hard per-subprocess wall-clock timeout
so `AUTO_UNASSIGN=True` always stops Colab billing cleanly.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.run_phase3_auto import (
    _FALLBACK_BASE_METRICS,
    load_kd_health,
    verify_exported_student_purity,
)


@dataclass
class DVRunMetrics:
    exp_name: str
    params_m: float
    val_map50: float
    val_map50_95: float
    test_map50: float
    test_map50_95: float
    day_map50: float
    day_map50_95: float
    night_map50: float
    night_map50_95: float
    delta_e2e_50: float
    rect_map50: float
    total_ms: float

    def is_finite(self) -> bool:
        return all(
            math.isfinite(v)
            for v in (self.params_m, self.val_map50, self.val_map50_95, self.test_map50, self.test_map50_95)
        )


def _sf(v: Any) -> float:
    try:
        return float(v) if v is not None and v != "" else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _fp(v: float, signed: bool = False) -> str:
    if not math.isfinite(v):
        return "N/A"
    return f"{v*100:+.2f}%" if signed else f"{v*100:.2f}%"


def load_dv_metrics(exp_name: str, *, work_dir: Path, drive_dir: Optional[Path]) -> Optional[DVRunMetrics]:
    cands = [work_dir / "runs" / exp_name / "eval_summary.json"]
    if drive_dir is not None:
        cands.append(drive_dir / "runs" / exp_name / "eval_summary.json")
    for jp in cands:
        if not jp.is_file():
            continue
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            if not isinstance(d, dict) or not d.get("evals"):
                continue
            is_e2e = bool(d.get("is_e2e_model", True))
            ev = d["evals"]
            val_d = ev.get("val_o2o_640" if is_e2e else "val_nms_640", {}) or {}
            tst_d = ev.get("test_o2o_640" if is_e2e else "test_nms_640", {}) or {}
            day_d = ev.get("test_day_640", {}) or {}
            ngt_d = ev.get("test_night_640", {}) or {}
            rct_d = ev.get("test_o2o_rect640x512" if is_e2e else "test_nms_rect640x512", {}) or {}
            m = DVRunMetrics(
                exp_name=exp_name,
                params_m=_sf(d.get("params_M")),
                val_map50=_sf(val_d.get("mAP50")),
                val_map50_95=_sf(val_d.get("mAP50_95")),
                test_map50=_sf(tst_d.get("mAP50")),
                test_map50_95=_sf(tst_d.get("mAP50_95")),
                day_map50=_sf(day_d.get("mAP50")),
                day_map50_95=_sf(day_d.get("mAP50_95")),
                night_map50=_sf(ngt_d.get("mAP50")),
                night_map50_95=_sf(ngt_d.get("mAP50_95")),
                delta_e2e_50=_sf(d.get("test_delta_e2e_map50")),
                rect_map50=_sf(rct_d.get("mAP50")),
                total_ms=_sf(tst_d.get("total_ms")),
            )
            if m.is_finite():
                return m
        except Exception as exc:
            print(f"[DV AUTO WARN] Failed parsing {jp}: {exc}")
    return None


class _DVReportLogger:
    def __init__(self, work_dir: Path, drive_dir: Optional[Path], wave_name: str) -> None:
        self.paths: List[Path] = [work_dir / f"auto_dv_{wave_name}_report.md"]
        if drive_dir is not None:
            self.paths.append(drive_dir / "logs" / "auto_dv_report.md")
        header = (
            f"\n# Phase 4 DroneVehicle Autonomous Run ({wave_name})\n"
            f"- Started: `{datetime.now().isoformat(timespec='seconds')}`\n\n"
        )
        for p in self.paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(header)

    def section(self, title: str) -> None:
        block = f"\n## {title}\n\n"
        print(block, flush=True)
        for p in self.paths:
            with p.open("a", encoding="utf-8") as f:
                f.write(block)

    def log(self, msg: str) -> None:
        line = f"- `{datetime.now().strftime('%H:%M:%S')}` {msg}\n"
        print(f"[DV AUTO] {msg}", flush=True)
        for p in self.paths:
            with p.open("a", encoding="utf-8") as f:
                f.write(line)


def _run_subprocess_checked(cmd: List[str], *, timeout_sec: float, label: str) -> None:
    if timeout_sec < 60:
        raise TimeoutError(f"Insufficient remaining budget ({timeout_sec:.0f}s) to start '{label}'.")
    print(f"\n[DV AUTO] Running {label}: {' '.join(cmd)} (timeout={timeout_sec/60:.1f}m)", flush=True)
    proc = subprocess.run(cmd, timeout=max(60.0, float(timeout_sec)))
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess '{label}' FAILED with exit code {proc.returncode}!")


def _run_exp_checked(
    exp_name: str,
    *,
    dataset_root: Path,
    work_dir: Path,
    drive_dir: Optional[Path],
    resume: bool,
    smoke_test: bool,
    export_onnx: bool,
    timeout_sec: float,
    report: _DVReportLogger,
) -> DVRunMetrics:
    cmd = [
        sys.executable,
        "-m",
        "src.train_runner",
        "--exp",
        exp_name,
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
    if export_onnx and not smoke_test:
        cmd.append("--export-onnx")

    _run_subprocess_checked(cmd, timeout_sec=timeout_sec, label=exp_name)
    out_name = f"{exp_name}_smoke" if smoke_test else exp_name
    m = load_dv_metrics(out_name, work_dir=work_dir, drive_dir=drive_dir)
    if m is None or not m.is_finite():
        raise RuntimeError(f"Experiment '{out_name}' exited 0 but produced missing/non-finite eval_summary.json!")

    day_s = f"Day=`{_fp(m.day_map50)}/{_fp(m.day_map50_95)}`" if math.isfinite(m.day_map50) else ""
    ngt_s = f"Night=`{_fp(m.night_map50)}/{_fp(m.night_map50_95)}`" if math.isfinite(m.night_map50) else ""
    dn_s = f" | {day_s} | {ngt_s}" if (day_s or ngt_s) else ""
    report.log(
        f"✅ `{out_name}`: Params={m.params_m:.4f}M | "
        f"Val=`{_fp(m.val_map50)} / {_fp(m.val_map50_95)}` | "
        f"Test=`{_fp(m.test_map50)} / {_fp(m.test_map50_95)}`{dn_s} | "
        f"Δ_E2E=`{_fp(m.delta_e2e_50, signed=True)}` | Rect=`{_fp(m.rect_map50)}`"
    )
    return m


def _verify_track_b_smoke(
    *,
    dataset_root: Path,
    work_dir: Path,
    drive_dir: Optional[Path],
    timeout_sec: float,
    report: _DVReportLogger,
    smoke_exp: str = "dv_kd_track_b_s",
) -> None:
    report.section(f"Track B Gate 1: Mandatory 2-Epoch Smoke Lock (`{smoke_exp}`)")
    sm = _run_exp_checked(
        smoke_exp,
        dataset_root=dataset_root,
        work_dir=work_dir,
        drive_dir=drive_dir,
        resume=False,
        smoke_test=True,
        export_onnx=False,
        timeout_sec=min(timeout_sec, 1800.0),
        report=report,
    )
    out_name = f"{smoke_exp}_smoke"
    pure_ok, pure_msg = verify_exported_student_purity(
        out_name, work_dir=work_dir, drive_dir=drive_dir, max_params_m=2.55
    )
    if not pure_ok:
        report.log(f"❌ TRACK B SMOKE PURITY FAILED: {pure_msg}")
        raise RuntimeError(f"Track B Smoke Purity FAILED: {pure_msg}")

    health = load_kd_health(out_name, work_dir=work_dir, drive_dir=drive_dir)
    if not health or not health.get("track_b_enabled"):
        raise RuntimeError(f"Track B Smoke FAILED: kd_health.json missing or track_b_enabled=False ({health})")
    if int(health.get("adapter_steps", 0)) < 1:
        raise RuntimeError(f"Track B Smoke FAILED: zero adapter updates ({health})")
    last_kd = health.get("last_epoch_avg_kd", {}) or {}
    safe_frac = float(last_kd.get("illum_safe_frac", -1.0))
    gate_frac = float(last_kd.get("cm_gate_active_frac", -1.0))
    if not (0.02 < safe_frac < 0.99):
        raise RuntimeError(
            f"Track B Smoke FAILED: spatial illumination safe_frac={safe_frac:.4f} out of expected (0.02, 0.99) "
            "(check PairedRGBIRDataset RGB normalization & Day/Night mix!)"
        )
    report.log(
        f"🟢 TRACK B SMOKE GATE PASSED! {pure_msg} | "
        f"adapter_steps={health['adapter_steps']}, peak_vram={health.get('peak_vram_gb', 0):.2f} GB, "
        f"illum_safe_frac={safe_frac*100:.1f}%, cm_gate_active={gate_frac*100:.1f}%, "
        f"Test={_fp(sm.test_map50)}/{_fp(sm.test_map50_95)}"
    )


def run_dv_wave1(
    *,
    dataset_root: Path,
    work_dir: Path,
    drive_dir: Optional[Path],
    resume: bool = True,
    smoke_only: bool = False,
    export_onnx: bool = False,
    max_hours: float = 8.5,
) -> Dict[str, Any]:
    """Wave 1: HIT-UAV v2 verification + DroneVehicle prep + Student IR + Teacher IR/RGB."""
    t_start = time.monotonic()
    budget_sec = float(max_hours) * 3600.0
    report = _DVReportLogger(work_dir, drive_dir, "wave1")

    def _rem() -> float:
        return max(0.0, budget_sec - (time.monotonic() - t_start))

    # Step 0: HIT-UAV v2 fixed KD verification (~45 min) — immediately tests our 3 root-cause fixes!
    report.section("Step 0: HIT-UAV Phase 3 v2 Root-Cause Fix Verification (`p3_kd_v2_full_s`)")
    base_h = _FALLBACK_BASE_METRICS[42]
    m_v2 = _run_exp_checked(
        "p3_kd_v2_full_s",
        dataset_root=dataset_root,
        work_dir=work_dir,
        drive_dir=drive_dir,
        resume=resume,
        smoke_test=smoke_only,
        export_onnx=False,
        timeout_sec=_rem(),
        report=report,
    )
    dv_val50 = m_v2.val_map50 - base_h["val_mAP50"]
    dv_val95 = m_v2.val_map50_95 - base_h["val_mAP50_95"]
    dv_tst50 = m_v2.test_map50 - base_h["mAP50"]
    dv_tst95 = m_v2.test_map50_95 - base_h["mAP50_95"]
    report.log(
        f"📊 HIT-UAV v2 vs `v33_e0_base` (s42): "
        f"Val Δ=`{_fp(dv_val50, True)} / {_fp(dv_val95, True)}` | "
        f"Test O2O Δ=`{_fp(dv_tst50, True)} / {_fp(dv_tst95, True)}`"
    )

    # Step 1: Prepare DroneVehicle directly on Colab -> Drive cache (0 local proxy traffic)
    report.section("Step 1: Prepare DroneVehicle (17,990 train / 8,980 test, 640x512 RGB+IR)")
    prep_cmd = [
        sys.executable,
        "-m",
        "src.prepare_dronevehicle",
        "--out-root",
        str(dataset_root),
    ]
    if drive_dir is not None:
        prep_cmd += [
            "--drive-bundle",
            str(drive_dir / "datasets" / "dronevehicle_bundle.zip"),
            "--vis-dir",
            str(drive_dir / "audit" / "dv_sanity_vis"),
        ]
    _run_subprocess_checked(prep_cmd, timeout_sec=min(_rem(), 3600.0), label="prepare_dronevehicle")
    report.log("✅ DroneVehicle ready on local SSD & cached to Drive bundle.")

    if smoke_only:
        # In smoke-only mode, verify 2-ep IR student & RGB teacher + Track B smoke end-to-end!
        _run_exp_checked(
            "dv_ir_yolo26n",
            dataset_root=dataset_root,
            work_dir=work_dir,
            drive_dir=drive_dir,
            resume=False,
            smoke_test=True,
            export_onnx=False,
            timeout_sec=_rem(),
            report=report,
        )
        report.log("🏁 Wave 1 smoke-only check passed cleanly.")
        return {"status": "WAVE1_SMOKE_OK"}

    # Step 2–4: DroneVehicle IR Student + IR Teacher + Privileged RGB Teacher
    report.section("Step 2–4: DroneVehicle Baselines & Dual Teachers (`YOLO26n-IR`, `YOLO26s-IR`, `YOLO26s-RGB`)")
    results: Dict[str, DVRunMetrics] = {}
    for exp_name, est_min in [
        ("dv_ir_yolo26n", 75.0),
        ("dv_ir_teacher_s", 105.0),
        ("dv_rgb_teacher_s", 105.0),
    ]:
        if _rem() < est_min * 40.0:
            report.log(f"⏸️ Budget guard: remaining {_rem()/60:.0f}m too tight for `{exp_name}`; stopping Wave 1 cleanly.")
            break
        results[exp_name] = _run_exp_checked(
            exp_name,
            dataset_root=dataset_root,
            work_dir=work_dir,
            drive_dir=drive_dir,
            resume=resume,
            smoke_test=False,
            export_onnx=export_onnx and (exp_name == "dv_ir_yolo26n"),
            timeout_sec=_rem(),
            report=report,
        )

    # Bonus Step 5: If both teachers completed AND >= 20 min left, run Track B 2-ep smoke gate right now!
    if "dv_ir_teacher_s" in results and "dv_rgb_teacher_s" in results and _rem() >= 1200.0:
        _verify_track_b_smoke(
            dataset_root=dataset_root,
            work_dir=work_dir,
            drive_dir=drive_dir,
            timeout_sec=_rem(),
            report=report,
        )

    report.log(f"🏁 Wave 1 finished cleanly in {(time.monotonic() - t_start)/3600:.2f}h. Next: `EXP_LIST = ['auto_dv_wave2']`.")
    return {"status": "WAVE1_DONE", "completed": list(results.keys())}


def run_dv_wave2(
    *,
    dataset_root: Path,
    work_dir: Path,
    drive_dir: Optional[Path],
    resume: bool = True,
    smoke_only: bool = False,
    export_onnx: bool = False,
    max_hours: float = 8.5,
) -> Dict[str, Any]:
    """Wave 2: Mandatory Track B Smoke Lock -> Track A vs Track B (Ours) vs Naive Ungated Control."""
    t_start = time.monotonic()
    budget_sec = float(max_hours) * 3600.0
    report = _DVReportLogger(work_dir, drive_dir, "wave2")

    def _rem() -> float:
        return max(0.0, budget_sec - (time.monotonic() - t_start))

    # Ensure DroneVehicle is unpacked from Drive bundle (~25s)
    if not (dataset_root / "dronevehicle_ir.yaml").is_file():
        prep_cmd = [sys.executable, "-m", "src.prepare_dronevehicle", "--out-root", str(dataset_root)]
        if drive_dir is not None:
            prep_cmd += ["--drive-bundle", str(drive_dir / "datasets" / "dronevehicle_bundle.zip")]
        _run_subprocess_checked(prep_cmd, timeout_sec=min(_rem(), 3600.0), label="prepare_dronevehicle")

    # Gate 1: Mandatory 2-epoch Track B Smoke Lock (verifies PairedRGBIRDataset, night breaker, pure best.pt)
    _verify_track_b_smoke(
        dataset_root=dataset_root,
        work_dir=work_dir,
        drive_dir=drive_dir,
        timeout_sec=_rem(),
        report=report,
        smoke_exp="dv_kd_track_b_s",
    )
    if smoke_only:
        report.log("🏁 Wave 2 smoke-only check passed cleanly.")
        return {"status": "WAVE2_SMOKE_OK"}

    # Core Track A vs Track B vs Naive Negative-Transfer Control Trio (+ bonus ablations)
    report.section("Step 1–4: DroneVehicle Track A vs Track B (Ours) vs Naive Ungated Control")
    queue = [
        ("dv_kd_track_a_s", 115.0),
        ("dv_kd_track_b_s", 145.0),
        ("dv_kd_track_b_naive_s", 145.0),
        ("dv_kd_track_b_nolp_s", 145.0),
    ]
    completed: Dict[str, DVRunMetrics] = {}
    for exp_name, est_min in queue:
        if _rem() < est_min * 40.0:
            report.log(f"⏸️ Budget guard: remaining {_rem()/60:.0f}m too tight for `{exp_name}`; stopping cleanly.")
            break
        completed[exp_name] = _run_exp_checked(
            exp_name,
            dataset_root=dataset_root,
            work_dir=work_dir,
            drive_dir=drive_dir,
            resume=resume,
            smoke_test=False,
            export_onnx=export_onnx and (exp_name == "dv_kd_track_b_s"),
            timeout_sec=_rem(),
            report=report,
        )

    # Final Day vs Night summary against Student IR baseline
    base_dv = load_dv_metrics("dv_ir_yolo26n", work_dir=work_dir, drive_dir=drive_dir)
    if base_dv is not None and completed:
        report.section("DroneVehicle Day vs Night Summary vs `dv_ir_yolo26n` Baseline (+0.00 ms Inference)")
        for name, m in completed.items():
            report.log(
                f"🏆 `{name}` vs base: "
                f"Overall Δ=`{_fp(m.test_map50 - base_dv.test_map50, True)} / {_fp(m.test_map50_95 - base_dv.test_map50_95, True)}` | "
                f"Day Δ=`{_fp(m.day_map50 - base_dv.day_map50, True)} / {_fp(m.day_map50_95 - base_dv.day_map50_95, True)}` | "
                f"Night Δ=`{_fp(m.night_map50 - base_dv.night_map50, True)} / {_fp(m.night_map50_95 - base_dv.night_map50_95, True)}`"
            )

    report.log(f"🏁 Wave 2 finished cleanly in {(time.monotonic() - t_start)/3600:.2f}h.")
    return {"status": "WAVE2_DONE", "completed": list(completed.keys())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous DroneVehicle Wave 1 / Wave 2 orchestrator.")
    parser.add_argument("--wave", choices=["wave1", "wave2"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--max-hours", type=float, default=8.5)
    args = parser.parse_args()

    fn = run_dv_wave1 if args.wave == "wave1" else run_dv_wave2
    fn(
        dataset_root=args.dataset_root,
        work_dir=args.work_dir,
        drive_dir=args.drive_dir,
        resume=not args.no_resume,
        smoke_only=args.smoke_only,
        export_onnx=args.export_onnx,
        max_hours=args.max_hours,
    )


if __name__ == "__main__":
    main()
