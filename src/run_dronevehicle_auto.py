"""Autonomous Phase 4 DroneVehicle + HIT-UAV v2 Orchestrator (Wave 1 & Wave 2).

Hardened against all adversarial failure modes:
- Uses `Popen(..., start_new_session=True)` + process-group `SIGTERM -> SIGKILL` timeout so
  hung `curl` downloads or PyTorch DataLoader workers are cleanly terminated before Colab limits.
- Strictly validates ALL required metrics (Val, Test, O2M Δ_E2E, Rect, and Day+Night on DroneVehicle)
  and binds local/Drive lookup cleanly so stale JSONs cannot mask failures.
- Track B Gate 1 strictly verifies: `actual_batch_size == 32`, `oom_retries == 0`,
  `primary_adapter_steps > 0`, `aux_adapter_steps > 0`, `0.05 < illum_safe_frac < 0.95`,
  `cm_gate_active_frac > 0`, finite `cm_feat > 0` & `cm_cls`, AND exact `YOLO26n(nc=5)` state_dict purity!
- Budget check skips already-completed tasks FIRST, reserves `est_min * 60 + 300`, and returns
  `INCOMPLETE` (with non-zero CLI exit code) if any mandatory task could not complete!
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs.experiments import get_experiment_config
from src.prepare_dronevehicle import verify_prepared_dronevehicle
from src.run_phase3_auto import (
    _FALLBACK_BASE_METRICS,
    load_kd_health,
    verify_exported_student_purity,
)
from src.train_runner import _is_run_already_completed


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

    def is_complete(self, *, require_day_night: bool, require_e2e: bool, require_rect: bool = True) -> bool:
        req = [self.params_m, self.val_map50, self.val_map50_95, self.test_map50, self.test_map50_95]
        if require_day_night:
            req += [self.day_map50, self.day_map50_95, self.night_map50, self.night_map50_95]
        if require_e2e:
            req.append(self.delta_e2e_50)
        if require_rect:
            req.append(self.rect_map50)
        return all(math.isfinite(v) for v in req)


def _sf(v: Any) -> float:
    try:
        return float(v) if v is not None and v != "" else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _fp(v: float, signed: bool = False) -> str:
    if not math.isfinite(v):
        return "N/A"
    return f"{v*100:+.2f}%" if signed else f"{v*100:.2f}%"


def load_dv_metrics(
    exp_name: str,
    *,
    work_dir: Path,
    drive_dir: Optional[Path],
    require_day_night: bool = False,
    require_e2e: bool = True,
    require_rect: bool = True,
) -> Optional[DVRunMetrics]:
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
            if m.is_complete(
                require_day_night=require_day_night,
                require_e2e=(require_e2e and is_e2e),
                require_rect=require_rect,
            ):
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
    """Run child process in its own session and kill the ENTIRE process group on timeout."""
    if timeout_sec < 60:
        raise TimeoutError(f"Insufficient remaining budget ({timeout_sec:.0f}s) to start '{label}'.")
    print(f"\n[DV AUTO] Running {label}: {' '.join(cmd)} (timeout={timeout_sec/60:.1f}m)", flush=True)
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        rc = proc.wait(timeout=max(60.0, float(timeout_sec)))
    except subprocess.TimeoutExpired:
        print(f"[DV AUTO TIMEOUT] Terminating process group for '{label}' (pgid={proc.pid}) ...", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        # Always send SIGKILL to the process group so detached grandchildren cannot survive after leader exits
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise TimeoutError(f"Subprocess '{label}' exceeded wall-clock timeout ({timeout_sec/60:.1f}m) and was killed!")
    if rc != 0:
        raise RuntimeError(f"Subprocess '{label}' FAILED with exit code {rc}!")


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
    cfg = get_experiment_config(exp_name, smoke_test=smoke_test)
    require_dv = cfg.dataset_name == "dronevehicle"
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

    _run_subprocess_checked(cmd, timeout_sec=timeout_sec, label=cfg.exp_name)
    m = load_dv_metrics(
        cfg.exp_name,
        work_dir=work_dir,
        drive_dir=drive_dir,
        require_day_night=require_dv,
        require_e2e=cfg.is_e2e_model,
        require_rect=cfg.eval_rect_640x512,
    )
    if m is None:
        raise RuntimeError(
            f"Experiment '{cfg.exp_name}' exited 0 but failed complete metric verification "
            f"(require_day_night={require_dv}, require_e2e={cfg.is_e2e_model})!"
        )

    day_s = f"Day=`{_fp(m.day_map50)}/{_fp(m.day_map50_95)}`" if math.isfinite(m.day_map50) else ""
    ngt_s = f"Night=`{_fp(m.night_map50)}/{_fp(m.night_map50_95)}`" if math.isfinite(m.night_map50) else ""
    dn_s = f" | {day_s} | {ngt_s}" if (day_s or ngt_s) else ""
    report.log(
        f"✅ `{cfg.exp_name}`: Params={m.params_m:.4f}M | "
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
    report.section(f"Track B Gate 1: Mandatory 2-Epoch Smoke Lock (`{smoke_exp}`, formal batch=32, MuSGD, close_mosaic=1)")
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
        out_name, work_dir=work_dir, drive_dir=drive_dir, max_params_m=2.55, expected_nc=5
    )
    if not pure_ok:
        report.log(f"❌ TRACK B SMOKE PURITY FAILED: {pure_msg}")
        raise RuntimeError(f"Track B Smoke Purity FAILED: {pure_msg}")

    health = load_kd_health(out_name, work_dir=work_dir, drive_dir=drive_dir)
    if not isinstance(health, dict) or health.get("track_b_enabled") is not True:
        raise RuntimeError(f"Track B Smoke FAILED: kd_health.json missing or track_b_enabled!=True ({health})")

    req_top = (
        "actual_batch_size",
        "oom_retries",
        "optimizer_name",
        "primary_adapter_steps",
        "aux_adapter_steps",
        "last_epoch_avg_kd",
    )
    missing_top = [k for k in req_top if k not in health]
    if missing_top:
        raise RuntimeError(f"Track B Smoke FAILED: kd_health.json missing required fields {missing_top}")

    last_kd = health["last_epoch_avg_kd"]
    req_kd = ("illum_safe_frac", "cm_gate_active_frac", "cm_feat", "cm_cls")
    if not isinstance(last_kd, dict) or any(k not in last_kd for k in req_kd):
        raise RuntimeError(f"Track B Smoke FAILED: last_epoch_avg_kd missing required keys {req_kd}: {last_kd}")

    actual_bs = int(health["actual_batch_size"])
    oom_retries = int(health["oom_retries"])
    opt_name = str(health["optimizer_name"])
    primary_steps = int(health["primary_adapter_steps"])
    aux_steps = int(health["aux_adapter_steps"])
    safe_frac = float(last_kd["illum_safe_frac"])
    gate_frac = float(last_kd["cm_gate_active_frac"])
    cm_feat = float(last_kd["cm_feat"])
    cm_cls = float(last_kd["cm_cls"])

    failures: List[str] = []
    if opt_name != "MuSGD":
        failures.append(f"optimizer_name={opt_name!r} != 'MuSGD'")
    if actual_bs != 32:
        failures.append(f"actual_batch_size={actual_bs} != 32")
    if oom_retries != 0:
        failures.append(f"oom_retries={oom_retries} != 0")
    if primary_steps < 1:
        failures.append(f"primary_adapter_steps={primary_steps} < 1")
    if aux_steps < 1:
        failures.append(f"aux_adapter_steps={aux_steps} < 1 (aux RGB adapter never updated!)")
    if not (math.isfinite(safe_frac) and 0.05 < safe_frac < 0.95):
        failures.append(f"illum_safe_frac={safe_frac:.4f} outside (0.05, 0.95)")
    if not (math.isfinite(cm_feat) and cm_feat > 0.0):
        failures.append(f"cm_feat={cm_feat} not strictly positive & finite")
    if not (math.isfinite(gate_frac) and gate_frac > 0.0):
        failures.append(f"cm_gate_active_frac={gate_frac} not strictly positive & finite")
    if not (math.isfinite(cm_cls) and cm_cls > 0.0):
        failures.append(f"cm_cls={cm_cls} not strictly positive & finite")

    if failures:
        msg = "Track B Smoke Gate FAILED: " + "; ".join(failures)
        report.log(f"❌ {msg}")
        raise RuntimeError(msg)

    report.log(
        f"🟢 TRACK B SMOKE GATE PASSED! {pure_msg} | "
        f"batch={actual_bs} (oom={oom_retries}), opt={health.get('optimizer_name')}, "
        f"primary_steps={primary_steps}, aux_steps={aux_steps}, "
        f"peak_vram={health.get('peak_vram_gb', 0):.2f} GB, "
        f"illum_safe={safe_frac*100:.1f}%, cm_gate={gate_frac*100:.1f}%, "
        f"cm_feat={cm_feat:.4f}, cm_cls={cm_cls:.4f}, "
        f"Day={_fp(sm.day_map50)}, Night={_fp(sm.night_map50)}"
    )


def _is_already_done(exp_name: str, drive_dir: Optional[Path]) -> bool:
    if drive_dir is None:
        return False
    cfg = get_experiment_config(exp_name)
    return _is_run_already_completed(drive_dir / "runs" / cfg.exp_name, cfg)


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
    archive_reserve_sec = 300.0
    report = _DVReportLogger(work_dir, drive_dir, "wave1")

    def _rem() -> float:
        return max(0.0, budget_sec - (time.monotonic() - t_start))

    # Step 0: HIT-UAV v2 fixed KD verification (~45 min)
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
    report.log(
        f"📊 HIT-UAV v2 vs `v33_e0_base` (s42): "
        f"Val Δ=`{_fp(m_v2.val_map50 - base_h['val_mAP50'], True)} / {_fp(m_v2.val_map50_95 - base_h['val_mAP50_95'], True)}` | "
        f"Test O2O Δ=`{_fp(m_v2.test_map50 - base_h['mAP50'], True)} / {_fp(m_v2.test_map50_95 - base_h['mAP50_95'], True)}`"
    )

    # Step 1: Prepare & verify DroneVehicle directly on Colab -> Drive cache (0 local proxy traffic)
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
    ok_dv, msg_dv = verify_prepared_dronevehicle(dataset_root)
    if not ok_dv:
        raise RuntimeError(f"DroneVehicle verification failed after prepare: {msg_dv}")
    report.log(f"✅ {msg_dv}")

    if smoke_only:
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

    # Step 2–4 (Mandatory Wave 1 tasks): IR Student + IR Teacher + Privileged RGB Teacher
    report.section("Step 2–4: DroneVehicle Baselines & Dual Teachers (`YOLO26n-IR`, `YOLO26s-IR`, `YOLO26s-RGB`)")
    mandatory = [
        ("dv_ir_yolo26n", 75.0),
        ("dv_ir_teacher_s", 105.0),
        ("dv_rgb_teacher_s", 105.0),
    ]
    results: Dict[str, DVRunMetrics] = {}
    missing_mandatory: List[str] = []

    for exp_name, est_min in mandatory:
        already = _is_already_done(exp_name, drive_dir)
        need_sec = 120.0 if already else (est_min * 60.0 + archive_reserve_sec)
        if _rem() < need_sec:
            report.log(
                f"⏸️ Budget guard: remaining {_rem()/60:.0f}m < required {need_sec/60:.0f}m for mandatory `{exp_name}`."
            )
            missing_mandatory.append(exp_name)
            continue
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

    if missing_mandatory:
        report.log(
            f"⚠️ INCOMPLETE Wave 1: mandatory tasks not finished within budget: {missing_mandatory}. "
            "Re-run `auto_dv_wave1` to resume from completed checkpoints!"
        )
        return {"status": "INCOMPLETE_WAVE1", "completed": list(results.keys()), "missing": missing_mandatory}

    # Optional Bonus Step 5: Pre-run Track B Gate 1 Smoke Lock if >= 20m remain
    if _rem() >= 1200.0:
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
    archive_reserve_sec = 300.0
    report = _DVReportLogger(work_dir, drive_dir, "wave2")

    def _rem() -> float:
        return max(0.0, budget_sec - (time.monotonic() - t_start))

    # Verify full DroneVehicle manifest (never trust a lone YAML!)
    ok_dv, msg_dv = verify_prepared_dronevehicle(dataset_root)
    if not ok_dv:
        report.log(f"[DATA] DroneVehicle check ({msg_dv}) -> restoring/preparing ...")
        prep_cmd = [sys.executable, "-m", "src.prepare_dronevehicle", "--out-root", str(dataset_root)]
        if drive_dir is not None:
            prep_cmd += ["--drive-bundle", str(drive_dir / "datasets" / "dronevehicle_bundle.zip")]
        _run_subprocess_checked(prep_cmd, timeout_sec=min(_rem(), 3600.0), label="prepare_dronevehicle")
        ok_dv2, msg_dv2 = verify_prepared_dronevehicle(dataset_root)
        if not ok_dv2:
            raise RuntimeError(f"DroneVehicle verification failed: {msg_dv2}")

    # Gate 1: Mandatory 2-epoch Track B Smoke Lock
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

    report.section("Step 1–3 (Mandatory): Track A vs Track B (Ours) vs Naive Ungated Control")
    mandatory = [
        ("dv_kd_track_a_s", 115.0),
        ("dv_kd_track_b_s", 145.0),
        ("dv_kd_track_b_naive_s", 145.0),
    ]
    optional = [
        ("dv_kd_track_b_nolp_s", 145.0),
    ]

    completed: Dict[str, DVRunMetrics] = {}
    missing_mandatory: List[str] = []

    for exp_name, est_min in mandatory:
        already = _is_already_done(exp_name, drive_dir)
        need_sec = 120.0 if already else (est_min * 60.0 + archive_reserve_sec)
        if _rem() < need_sec:
            report.log(
                f"⏸️ Budget guard: remaining {_rem()/60:.0f}m < required {need_sec/60:.0f}m for mandatory `{exp_name}`."
            )
            missing_mandatory.append(exp_name)
            continue
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

    # Optional ablations only if mandatory trio finished and full budget remains
    if not missing_mandatory:
        for exp_name, est_min in optional:
            already = _is_already_done(exp_name, drive_dir)
            need_sec = 120.0 if already else (est_min * 60.0 + archive_reserve_sec)
            if _rem() < need_sec:
                report.log(f"⏸️ Skipping optional `{exp_name}` (remaining {_rem()/60:.0f}m < {need_sec/60:.0f}m).")
                continue
            completed[exp_name] = _run_exp_checked(
                exp_name,
                dataset_root=dataset_root,
                work_dir=work_dir,
                drive_dir=drive_dir,
                resume=resume,
                smoke_test=False,
                export_onnx=False,
                timeout_sec=_rem(),
                report=report,
            )

    # Day vs Night summary against Student IR baseline
    base_dv = load_dv_metrics(
        "dv_ir_yolo26n", work_dir=work_dir, drive_dir=drive_dir, require_day_night=True, require_e2e=True
    )
    if base_dv is not None and completed:
        report.section("DroneVehicle Day vs Night Summary vs `dv_ir_yolo26n` Baseline (+0.00 ms Inference)")
        for name, m in completed.items():
            report.log(
                f"🏆 `{name}` vs base: "
                f"Overall Δ=`{_fp(m.test_map50 - base_dv.test_map50, True)} / {_fp(m.test_map50_95 - base_dv.test_map50_95, True)}` | "
                f"Day Δ=`{_fp(m.day_map50 - base_dv.day_map50, True)} / {_fp(m.day_map50_95 - base_dv.day_map50_95, True)}` | "
                f"Night Δ=`{_fp(m.night_map50 - base_dv.night_map50, True)} / {_fp(m.night_map50_95 - base_dv.night_map50_95, True)}`"
            )

    if missing_mandatory:
        report.log(
            f"⚠️ INCOMPLETE Wave 2: mandatory tasks not finished within budget: {missing_mandatory}. "
            "Re-run `auto_dv_wave2` to continue seamlessly from Drive checkpoints!"
        )
        return {"status": "INCOMPLETE_WAVE2", "completed": list(completed.keys()), "missing": missing_mandatory}

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
    res = fn(
        dataset_root=args.dataset_root,
        work_dir=args.work_dir,
        drive_dir=args.drive_dir,
        resume=not args.no_resume,
        smoke_only=args.smoke_only,
        export_onnx=args.export_onnx,
        max_hours=args.max_hours,
    )
    if str(res.get("status", "")).startswith("INCOMPLETE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
