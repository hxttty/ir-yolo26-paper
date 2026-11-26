"""Master experiment runner for Colab Pro & local execution.

Features built for iterative research ("改了跑、跑了改") without wasting Colab CUs:
- Driven by named presets in `configs/experiments.py` + clean CLI overrides.
- Automatic Drive checkpoint syncing every `save_period` epochs so abrupt
  Colab disconnects never lose training progress.
- Automatic `--resume` detection from Drive `weights/last.pt`.
- Post-training dual evaluation (Val + Test, O2O + O2M Δ_E2E, native 640x512,
  optional ONNX CPU benchmark) appended directly to Drive `master_results.csv`.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from configs.experiments import ExperimentConfig, get_experiment_config, list_presets
from src.eval_utils import evaluate_checkpoint, save_qualitative_predictions

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class _TeeStream(io.TextIOBase):
    """Mirror stdout/stderr to console and log file while filtering noisy tqdm redraws."""

    def __init__(self, orig_stream, log_file):
        self._orig = orig_stream
        self._log = log_file

    def write(self, data: str) -> int:
        # Drop intermediate tqdm redraw frames (0%-99% contain '━' or '─' without '100%')
        # so Colab's browser DOM stays ultra-light across long multi-run overnight queues.
        if ("━" in data or "─" in data) and "100%" not in data:
            return len(data)
        n = self._orig.write(data)
        self._orig.flush()
        clean = _ANSI_RE.sub("", data).replace("\r", "\n")
        self._log.write(clean)
        self._log.flush()
        return n

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._orig, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._orig.fileno()


def _sync_tree(src: Path, dst: Path) -> None:
    """Safely mirror essential run artifacts from fast local SSD -> Google Drive."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _sync_weights_and_csv(local_run_dir: Path, drive_run_dir: Path) -> None:
    """Fast lightweight sync called periodically during training."""
    drive_weights = drive_run_dir / "weights"
    drive_weights.mkdir(parents=True, exist_ok=True)
    for fname in ("last.pt", "best.pt"):
        src_f = local_run_dir / "weights" / fname
        if src_f.exists():
            shutil.copy2(src_f, drive_weights / fname)
    for fname in ("results.csv", "args.yaml"):
        src_f = local_run_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, drive_run_dir / fname)


def run_experiment(
    cfg: ExperimentConfig,
    *,
    dataset_root: Path,
    local_work_dir: Path,
    drive_dir: Optional[Path],
    repo_root: Path,
) -> Path:
    dataset_yaml = dataset_root / f"hit_uav_{cfg.dataset_mode}.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Run `python -m src.prepare_hituav` first!"
        )

    local_project = local_work_dir / "runs"
    local_run_dir = local_project / cfg.exp_name
    drive_run_dir = (drive_dir / "runs" / cfg.exp_name) if drive_dir is not None else None
    master_csv = (drive_dir / "results" / "master_results.csv") if drive_dir is not None else (local_work_dir / "master_results.csv")

    print("\n" + "=" * 72)
    print(f"[RUN] Experiment: {cfg.exp_name} ({cfg.phase})")
    print(f"[RUN] Model: {cfg.model_cfg} | Pretrained: {cfg.pretrained_weights} | E2E: {cfg.is_e2e_model}")
    print(f"[RUN] Dataset: {dataset_yaml} | Epochs: {cfg.epochs} | Batch: {cfg.batch} | Opt: {cfg.optimizer} | Seed: {cfg.seed}")
    print(f"[RUN] Smoke test: {cfg.smoke_test} | Resume: {cfg.resume} | Save period: {cfg.save_period}")
    print("=" * 72 + "\n")

    # Skip cleanly if already completed on Drive (unless smoke_test)
    if drive_run_dir is not None and (drive_run_dir / "_COMPLETED.flag").exists() and not cfg.smoke_test:
        print(f"[SKIP] Found {drive_run_dir / '_COMPLETED.flag'}. Delete it on Drive if you want to re-run.")
        return drive_run_dir

    # Check if we should resume from Drive checkpoint
    resume_ckpt: Optional[Path] = None
    if cfg.resume and not cfg.smoke_test:
        candidates = [local_run_dir / "weights" / "last.pt"]
        if drive_run_dir is not None:
            candidates.append(drive_run_dir / "weights" / "last.pt")
        for cand in candidates:
            if cand.is_file():
                # Copy Drive checkpoint to local SSD before resuming
                if drive_run_dir is not None and cand.is_relative_to(drive_run_dir):
                    local_weights = local_run_dir / "weights"
                    local_weights.mkdir(parents=True, exist_ok=True)
                    dst_last = local_weights / "last.pt"
                    shutil.copy2(cand, dst_last)
                    resume_ckpt = dst_last
                else:
                    resume_ckpt = cand
                print(f"[RESUME] Resuming interrupted run from: {resume_ckpt}")
                break

    from src.modules import attach_phase1_callbacks, build_model_with_switches

    # Auto-resolve frozen KD teacher checkpoint from local SSD or persistent Drive archive
    if cfg.use_kd:
        resolved_teacher: Optional[Path] = None
        if cfg.kd_teacher_weights and Path(cfg.kd_teacher_weights).is_file():
            resolved_teacher = Path(cfg.kd_teacher_weights)
        elif cfg.kd_teacher_run:
            search_candidates = [
                local_project / cfg.kd_teacher_run / "weights" / "best.pt",
            ]
            if drive_dir is not None:
                search_candidates.append(drive_dir / "runs" / cfg.kd_teacher_run / "weights" / "best.pt")
            for cand in search_candidates:
                if cand.is_file():
                    if drive_dir is not None and cand.is_relative_to(drive_dir):
                        cached_dir = local_work_dir / "teacher_cache" / cfg.kd_teacher_run
                        cached_dir.mkdir(parents=True, exist_ok=True)
                        cached_pt = cached_dir / "best.pt"
                        if not cached_pt.is_file():
                            shutil.copy2(cand, cached_pt)
                        resolved_teacher = cached_pt
                    else:
                        resolved_teacher = cand
                    break
        if resolved_teacher is None:
            raise FileNotFoundError(
                f"KD teacher weights not found! Looked for weights='{cfg.kd_teacher_weights}' "
                f"and run='{cfg.kd_teacher_run}' under {local_project} / {drive_dir}."
            )
        cfg.kd_teacher_weights = str(resolved_teacher)
        print(f"[PHASE 3 KD] Resolved frozen teacher checkpoint: {resolved_teacher}")

    if resume_ckpt is not None:
        from ultralytics import YOLO

        model = YOLO(str(resume_ckpt))
        # Re-attach training-only hooks (e.g. Mod C1/C2 NWD criterion patch) on resume!
        attach_phase1_callbacks(model, cfg)
    else:
        model = build_model_with_switches(cfg, repo_root=repo_root)

    # Register periodic Drive sync callback so disconnects lose minimal epochs
    if drive_run_dir is not None and cfg.save_period > 0:
        def _on_fit_epoch_end(trainer) -> None:
            epoch_idx = getattr(trainer, "epoch", 0) + 1
            if epoch_idx % cfg.save_period == 0:
                try:
                    _sync_weights_and_csv(Path(trainer.save_dir), drive_run_dir)
                    print(f"[DRIVE SYNC] Backed up epoch {epoch_idx} checkpoints -> {drive_run_dir}")
                except Exception as exc:
                    print(f"[WARN] Periodic Drive sync failed at epoch {epoch_idx}: {exc}")

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    t_start = time.perf_counter()
    if resume_ckpt is not None:
        model.train(resume=True)
    else:
        train_kwargs = {
            "data": str(dataset_yaml),
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "batch": cfg.batch,
            "imgsz": cfg.imgsz,
            "optimizer": cfg.optimizer,
            "lr0": cfg.lr0,
            "lrf": cfg.lrf,
            "weight_decay": cfg.weight_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "close_mosaic": cfg.close_mosaic,
            "workers": cfg.workers,
            "cache": cfg.cache,
            "amp": cfg.amp,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "project": str(local_project),
            "name": cfg.exp_name,
            "exist_ok": True,
            "save": True,
            "save_period": -1,  # Keep disk clean; our callback backs up last.pt + best.pt to Drive
            "plots": True,
        }
        # Apply optional IR-Clean augmentation overrides when explicitly specified in preset
        for opt_key in ("hsv_h", "hsv_s", "hsv_v", "erasing"):
            val = getattr(cfg, opt_key, None)
            if val is not None:
                train_kwargs[opt_key] = val
        train_kwargs.update(cfg.extra_train_kwargs)
        model.train(**train_kwargs)

    train_minutes = (time.perf_counter() - t_start) / 60.0
    print(f"\n[DONE] Training finished in {train_minutes:.2f} minutes.")

    # Save exact experiment config snapshot
    local_run_dir.mkdir(parents=True, exist_ok=True)
    snap = cfg.to_dict()
    snap["train_wall_minutes"] = round(train_minutes, 2)
    (local_run_dir / "config_snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Evaluate best checkpoint (fallback to last.pt)
    best_pt = local_run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = local_run_dir / "weights" / "last.pt"

    if best_pt.is_file():
        evaluate_checkpoint(
            weights_path=best_pt,
            dataset_yaml=dataset_yaml,
            exp_name=cfg.exp_name,
            is_e2e_model=cfg.is_e2e_model,
            imgsz=cfg.imgsz,
            batch=min(cfg.batch, 16),
            eval_rect_640x512=cfg.eval_rect_640x512,
            export_onnx_bench=cfg.export_onnx_bench,
            seed=cfg.seed,
            dataset_mode=cfg.dataset_mode,
            notes=cfg.notes,
            summary_json_path=local_run_dir / "eval_summary.json",
            master_csv_path=None if cfg.smoke_test else master_csv,
        )
        if cfg.save_failure_vis:
            save_qualitative_predictions(
                weights_path=best_pt,
                dataset_yaml=dataset_yaml,
                out_dir=local_run_dir / "vis_preds",
                is_e2e_model=cfg.is_e2e_model,
                imgsz=cfg.imgsz,
            )

    # Final full sync to Google Drive
    if drive_run_dir is not None:
        print(f"[DRIVE SYNC] Uploading final run artifacts -> {drive_run_dir} ...")
        _sync_tree(local_run_dir, drive_run_dir)
        if not cfg.smoke_test:
            (drive_run_dir / "_COMPLETED.flag").write_text(
                f"completed_minutes={train_minutes:.2f}\n", encoding="utf-8"
            )
        print(f"[OK] Run safely archived on Google Drive: {drive_run_dir}")
        return drive_run_dir

    return local_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HIT-UAV YOLO26 experiment preset.")
    parser.add_argument("--list", action="store_true", help="List all available experiment presets and exit.")
    parser.add_argument("--exp", type=str, default="p0_yolo26n_musgd", help="Experiment preset name.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    parser.add_argument("--drive-dir", type=Path, default=None, help="Persistent Google Drive root.")
    parser.add_argument("--smoke-test", action="store_true", help="2-epoch fast sanity run (~60-90s).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--dataset-mode", type=str, choices=["4cls", "5cls"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(list_presets())
        return

    # Magic meta-preset: lets Colab Cell 1 run the full autonomous Phase 3 overnight
    # decision tree via `EXP_LIST = ["auto_phase3_overnight"]` with ZERO notebook edits!
    if args.exp == "auto_phase3_overnight":
        from src.run_phase3_auto import run_overnight_orchestrator

        run_overnight_orchestrator(
            dataset_root=args.dataset_root,
            work_dir=args.work_dir,
            drive_dir=args.drive_dir,
            resume=not args.no_resume,
            smoke_only=bool(args.smoke_test),
            export_onnx=bool(args.export_onnx),
        )
        return

    # Support inline per-item seed syntax in Colab EXP_LIST: e.g. "v33_e0_base@s0"
    exp_arg = args.exp
    seed_override = args.seed
    if "@s" in exp_arg:
        base_exp, seed_str = exp_arg.rsplit("@s", 1)
        if seed_str.isdigit():
            exp_arg = base_exp
            seed_override = int(seed_str)
            print(f"[SEED OVERRIDE] Parsed '{args.exp}' -> preset='{exp_arg}', seed={seed_override}")

    cfg = get_experiment_config(
        exp_arg,
        smoke_test=True if args.smoke_test else None,
        seed=seed_override,
        optimizer=args.optimizer,
        dataset_mode=args.dataset_mode,
        epochs=args.epochs,
        batch=args.batch,
        resume=False if args.no_resume else None,
        export_onnx_bench=True if args.export_onnx else None,
    )

    repo_root = Path(__file__).resolve().parent.parent
    local_log_dir = args.work_dir / "logs"
    local_log_dir.mkdir(parents=True, exist_ok=True)
    local_log_path = local_log_dir / f"{cfg.exp_name}.log"

    orig_out, orig_err = sys.stdout, sys.stderr
    with local_log_path.open("w", encoding="utf-8") as log_f:
        sys.stdout = _TeeStream(orig_out, log_f)
        sys.stderr = _TeeStream(orig_err, log_f)
        try:
            run_experiment(
                cfg,
                dataset_root=args.dataset_root,
                local_work_dir=args.work_dir,
                drive_dir=args.drive_dir,
                repo_root=repo_root,
            )
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
            if args.drive_dir is not None:
                try:
                    drive_logs = args.drive_dir / "logs"
                    drive_logs.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_log_path, drive_logs / f"{cfg.exp_name}.log")
                    shutil.copy2(local_log_path, drive_logs / "latest_run.log")
                    print(f"[LOG SYNC] Full log synced to Drive: {drive_logs / 'latest_run.log'}")
                except Exception as exc:
                    print(f"[WARN] Could not sync run log to Drive: {exc}")


if __name__ == "__main__":
    main()
