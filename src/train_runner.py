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
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from configs.experiments import ExperimentConfig, get_experiment_config, list_presets
from src.eval_utils import evaluate_checkpoint, save_qualitative_predictions


class _TeeStream(io.TextIOBase):
    """Mirror stdout/stderr to both console and a fast local log file."""

    def __init__(self, orig_stream, log_file):
        self._orig = orig_stream
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._orig.write(data)
        self._orig.flush()
        # Keep log clean and readable even with tqdm carriage returns
        self._log.write(data.replace("\r", "\n"))
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
            **cfg.extra_train_kwargs,
        }
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

    cfg = get_experiment_config(
        args.exp,
        smoke_test=True if args.smoke_test else None,
        seed=args.seed,
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
