"""Master experiment runner for Colab Pro & local execution (HIT-UAV & DroneVehicle).

Hardened against partial checkpoints, mixed-generation Drive syncs, last-epoch crashes,
stale teacher caches, and missing post-evaluation metrics:
- Atomic generation-directory Drive checkpoint sync (`weights/gen_ep{N}` + `current_gen.json`)
  so student `last.pt`, `best.pt`, and KD `kd_state.pt` ALWAYS restore from the exact same epoch.
- SHA-256 content-addressed teacher cache (`teacher_cache/<run>_<sha12>.pt`).
- Detects ALL completed-training checkpoints (`epoch == -1`, stripped optimizer, OR
  `ckpt_epoch + 1 >= planned_epochs`) and jumps straight to pure-IR evaluation without requiring
  teacher weights.
- Verifies `best.pt` existence, required eval splits (incl. DroneVehicle Day/Night & E2E O2M),
  and writes config/weights-bound `_COMPLETED.flag` ONLY after evaluation and Drive archive pass.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from configs.experiments import ExperimentConfig, get_experiment_config, list_presets
from src.eval_utils import evaluate_checkpoint, save_qualitative_predictions

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_sha256(cfg: ExperimentConfig) -> str:
    d = cfg.to_dict()
    for transient in ("resume", "save_period", "save_failure_vis", "kd_teacher_weights", "kd_aux_teacher_weights", "kd_aux_rgb_root"):
        d.pop(transient, None)
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class _TeeStream(io.TextIOBase):
    """Mirror stdout/stderr to console and log file while filtering noisy tqdm redraws."""

    def __init__(self, orig_stream, log_file):
        self._orig = orig_stream
        self._log = log_file

    def write(self, data: str) -> int:
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


def _atomic_copy2(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _sync_tree(src: Path, dst: Path) -> None:
    """Safely mirror essential run artifacts from fast local SSD -> Google Drive."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_dir() or item.name.endswith(".tmp"):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _sync_weights_and_csv(
    local_run_dir: Path,
    drive_run_dir: Path,
    *,
    epoch_idx: int,
    require_kd_state: bool = False,
) -> None:
    """Atomic unique-generation Drive checkpoint sync with per-file SHA-256 (zero overwrites/mixed epochs!)."""
    drive_weights = drive_run_dir / "weights"
    gen_name = f"gen_ep{epoch_idx}_{uuid.uuid4().hex[:8]}"
    gen_dir = drive_weights / gen_name
    gen_dir.mkdir(parents=True, exist_ok=True)

    last_src = local_run_dir / "weights" / "last.pt"
    if not last_src.is_file():
        raise FileNotFoundError(f"Periodic Drive sync expected {last_src}, but it is missing!")

    kd_src = local_run_dir / "weights" / "kd_state.pt"
    if require_kd_state and not kd_src.is_file():
        raise FileNotFoundError(f"KD Drive sync expected {kd_src}, but it is missing!")

    synced_files: Dict[str, str] = {}
    for fname in ("last.pt", "best.pt", "kd_state.pt"):
        src_f = local_run_dir / "weights" / fname
        if src_f.is_file():
            dst_f = gen_dir / fname
            _atomic_copy2(src_f, dst_f)
            synced_files[fname] = _file_sha256(src_f)

    # Atomically point `current_gen.json` to this complete verified generation ONLY after all files succeed!
    manifest = {
        "generation": gen_name,
        "epoch": int(epoch_idx),
        "files": list(synced_files.keys()),
        "sha256": synced_files,
    }
    tmp_m = drive_weights / ".current_gen.json.tmp"
    tmp_m.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_m, drive_weights / "current_gen.json")

    for fname in synced_files:
        _atomic_copy2(local_run_dir / "weights" / fname, drive_weights / fname)

    for fname in ("results.csv", "args.yaml"):
        src_f = local_run_dir / fname
        if src_f.is_file():
            _atomic_copy2(src_f, drive_run_dir / fname)


def _is_eval_summary_complete(eval_path: Path, cfg: ExperimentConfig) -> bool:
    if not eval_path.is_file():
        return False
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        evals = data.get("evals", {}) or {}
        req_tags = ["val_o2o_640", "test_o2o_640"] if cfg.is_e2e_model else ["val_nms_640", "test_nms_640"]
        if cfg.is_e2e_model:
            req_tags += ["val_o2m_640", "test_o2m_640"]
            de2e = data.get("test_delta_e2e_map50")
            if de2e is None or not math.isfinite(float(de2e)):
                return False
        if cfg.eval_rect_640x512:
            req_tags.append("test_o2o_rect640x512" if cfg.is_e2e_model else "test_nms_rect640x512")
        if cfg.dataset_name == "dronevehicle":
            req_tags += ["test_day_640", "test_night_640"]
        for tag in req_tags:
            sub = evals.get(tag)
            if not isinstance(sub, dict):
                return False
            for k in ("mAP50", "mAP50_95"):
                v = sub.get(k)
                if v is None or not math.isfinite(float(v)):
                    return False
        if cfg.export_onnx_bench:
            ob = data.get("onnx_bench", {}) or {}
            for sz_k in ("640x640", "512x640"):
                sub_o = ob.get(sz_k)
                if not isinstance(sub_o, dict):
                    return False
                lat = sub_o.get("cpu_latency_ms")
                if lat is None or not math.isfinite(float(lat)):
                    return False
        return True
    except Exception:
        return False


def _is_run_already_completed(drive_run_dir: Optional[Path], cfg: ExperimentConfig) -> bool:
    if drive_run_dir is None or cfg.smoke_test:
        return False
    flag_p = drive_run_dir / "_COMPLETED.flag"
    if not flag_p.is_file():
        return False
    best_p = drive_run_dir / "weights" / "best.pt"
    if not best_p.is_file():
        print(f"[WARN] Found {flag_p}, but {best_p} is MISSING -> refusing to skip!")
        return False
    eval_p = drive_run_dir / "eval_summary.json"
    if not _is_eval_summary_complete(eval_p, cfg):
        print(f"[WARN] Found {flag_p}, but {eval_p} is missing or incomplete -> will re-run/repair!")
        return False
    actual_sha = _file_sha256(best_p)
    eval_data = json.loads(eval_p.read_text(encoding="utf-8"))
    eval_sha = str(eval_data.get("weights_sha256", "") or "")
    if eval_sha and eval_sha != actual_sha:
        print(f"[WARN] {eval_p} weights_sha256 disagrees with {best_p} -> refusing to skip!")
        return False

    raw = flag_p.read_text(encoding="utf-8").strip()
    if raw.startswith("{"):
        try:
            meta = json.loads(raw)
            saved_cfg_sha = str(meta.get("config_sha256", "") or "")
            saved_w_sha = str(meta.get("weights_sha256", "") or "")
            cur_cfg_sha = _config_sha256(cfg)
            if not saved_cfg_sha or saved_cfg_sha != cur_cfg_sha:
                print(
                    f"[WARN] Found {flag_p}, but config digest changed "
                    f"(saved={saved_cfg_sha[:8]} vs current={cur_cfg_sha[:8]}) -> re-running!"
                )
                return False
            if not saved_w_sha or saved_w_sha != actual_sha:
                print(f"[WARN] Found {flag_p}, but weights_sha256 disagrees with {best_p} -> re-running!")
                return False
            if cfg.dataset_name == "dronevehicle" and eval_sha != actual_sha:
                print(f"[WARN] DroneVehicle eval_summary missing matching weights_sha256 -> re-running!")
                return False
            return True
        except Exception:
            return False

    # Legacy plain-text `_COMPLETED.flag`: NEVER accept for DroneVehicle; on HIT-UAV require matching snapshot + best.pt
    if cfg.dataset_name == "dronevehicle":
        print(f"[WARN] Legacy plain-text {flag_p} rejected for DroneVehicle (requires structured SHA flag) -> re-running!")
        return False
    snap_p = drive_run_dir / "config_snapshot.json"
    if snap_p.is_file():
        try:
            snap = json.loads(snap_p.read_text(encoding="utf-8"))
            if int(snap.get("epochs", -1)) != int(cfg.epochs) or str(snap.get("model_cfg", "")) != str(cfg.model_cfg):
                print(f"[WARN] Legacy {flag_p} snapshot disagrees with current config -> re-running!")
                return False
        except Exception:
            return False
    return True


def _resolve_teacher_ckpt(
    *,
    explicit_weights: Optional[str],
    run_name: Optional[str],
    local_project: Path,
    local_work_dir: Path,
    drive_dir: Optional[Path],
    tag: str,
) -> Path:
    resolved: Optional[Path] = None
    if explicit_weights and Path(explicit_weights).is_file():
        resolved = Path(explicit_weights)
    elif run_name:
        search_candidates = [local_project / run_name / "weights" / "best.pt"]
        if drive_dir is not None:
            search_candidates.append(drive_dir / "runs" / run_name / "weights" / "best.pt")
        for cand in search_candidates:
            if cand.is_file():
                if drive_dir is not None and cand.is_relative_to(drive_dir):
                    sha12 = _file_sha256(cand)[:12]
                    cached_dir = local_work_dir / "teacher_cache"
                    cached_dir.mkdir(parents=True, exist_ok=True)
                    cached_pt = cached_dir / f"{run_name}_{sha12}.pt"
                    if not cached_pt.is_file():
                        _atomic_copy2(cand, cached_pt)
                    resolved = cached_pt
                else:
                    resolved = cand
                break
    if resolved is None:
        raise FileNotFoundError(
            f"KD {tag} teacher weights not found! Looked for weights='{explicit_weights}' "
            f"and run='{run_name}' under {local_project} / {drive_dir}."
        )
    print(f"[PHASE 3 KD] Resolved frozen {tag} teacher checkpoint: {resolved}")
    return resolved


def run_experiment(
    cfg: ExperimentConfig,
    *,
    dataset_root: Path,
    local_work_dir: Path,
    drive_dir: Optional[Path],
    repo_root: Path,
) -> Path:
    if cfg.dataset_name == "dronevehicle":
        from src.prepare_dronevehicle import prepare_dronevehicle, verify_prepared_dronevehicle

        ok_dv, msg_dv = verify_prepared_dronevehicle(dataset_root)
        if not ok_dv:
            print(f"[AUTO-DATA] DroneVehicle check ({msg_dv}) -> preparing directly on Colab SSD ...")
            prepare_dronevehicle(
                out_root=dataset_root,
                drive_bundle=(drive_dir / "datasets" / "dronevehicle_bundle.zip") if drive_dir is not None else None,
                vis_dir=(drive_dir / "audit" / "dv_sanity_vis") if drive_dir is not None else None,
            )
        dataset_yaml = dataset_root / f"dronevehicle_{cfg.dataset_mode}.yaml"
        if not dataset_yaml.is_file():
            raise FileNotFoundError(f"DroneVehicle YAML missing even after preparation: {dataset_yaml}")
        master_csv_name = "master_results_dronevehicle.csv"
    else:
        dataset_yaml = dataset_root / f"hit_uav_{cfg.dataset_mode}.yaml"
        if not dataset_yaml.is_file():
            raise FileNotFoundError(
                f"Dataset YAML not found: {dataset_yaml}\n"
                "Run `python -m src.prepare_hituav` first!"
            )
        master_csv_name = "master_results.csv"

    local_project = local_work_dir / "runs"
    local_run_dir = local_project / cfg.exp_name
    drive_run_dir = (drive_dir / "runs" / cfg.exp_name) if drive_dir is not None else None
    master_csv = (
        (drive_dir / "results" / master_csv_name)
        if drive_dir is not None
        else (local_work_dir / master_csv_name)
    )

    print("\n" + "=" * 72)
    print(f"[RUN] Experiment: {cfg.exp_name} ({cfg.phase})")
    print(f"[RUN] Model: {cfg.model_cfg} | Pretrained: {cfg.pretrained_weights} | E2E: {cfg.is_e2e_model}")
    print(f"[RUN] Dataset: {dataset_yaml} | Epochs: {cfg.epochs} | Batch: {cfg.batch} | Opt: {cfg.optimizer} | Seed: {cfg.seed}")
    print(f"[RUN] Smoke test: {cfg.smoke_test} | Resume: {cfg.resume} | Save period: {cfg.save_period}")
    print("=" * 72 + "\n")

    # Skip cleanly ONLY if `_COMPLETED.flag` matches config digest AND `eval_summary.json` is complete & finite!
    if _is_run_already_completed(drive_run_dir, cfg):
        assert drive_run_dir is not None
        print(f"[SKIP] Verified completed run at {drive_run_dir}. Delete `_COMPLETED.flag` on Drive to force re-run.")
        return drive_run_dir

    # Check if we should resume from local or atomic Drive generation checkpoint
    # FIRST check whether Drive has a config-mismatch marker (refuse resuming old weights under changed config!)
    config_changed = False
    if drive_run_dir is not None and not cfg.smoke_test:
        flag_p = drive_run_dir / "_COMPLETED.flag"
        if flag_p.is_file():
            raw_f = flag_p.read_text(encoding="utf-8").strip()
            if raw_f.startswith("{"):
                try:
                    f_meta = json.loads(raw_f)
                    if f_meta.get("config_sha256") and f_meta.get("config_sha256") != _config_sha256(cfg):
                        config_changed = True
                except Exception:
                    pass
        snap_p = drive_run_dir / "config_snapshot.json"
        if not config_changed and snap_p.is_file():
            try:
                s_meta = json.loads(snap_p.read_text(encoding="utf-8"))
                if s_meta.get("config_sha256") and s_meta.get("config_sha256") != _config_sha256(cfg):
                    # Allow reusing completed weights ONLY if training hyperparams match and only eval flags changed
                    train_keys = ("model_cfg", "epochs", "batch", "optimizer", "lr0", "seed", "dataset_name", "use_kd")
                    if any(s_meta.get(k) != cfg.to_dict().get(k) for k in train_keys):
                        config_changed = True
            except Exception:
                pass

    resume_ckpt: Optional[Path] = None
    skip_train_eval_only: bool = False
    if config_changed:
        print(
            f"[FRESH] Training config changed for `{cfg.exp_name}` -> refusing to resume stale Drive/local checkpoint; "
            f"starting clean run!"
        )
        if local_run_dir.exists():
            shutil.rmtree(local_run_dir)
    elif cfg.resume and not cfg.smoke_test:
        local_last = local_run_dir / "weights" / "last.pt"
        drive_src_dir: Optional[Path] = None
        gen_sha_map: Dict[str, str] = {}
        if not local_last.is_file() and drive_run_dir is not None:
            gen_manifest = drive_run_dir / "weights" / "current_gen.json"
            if gen_manifest.is_file():
                gm = json.loads(gen_manifest.read_text(encoding="utf-8"))
                cand_gen = drive_run_dir / "weights" / str(gm.get("generation", ""))
                if (cand_gen / "last.pt").is_file():
                    drive_src_dir = cand_gen
                    gen_sha_map = dict(gm.get("sha256", {}) or {})
            if drive_src_dir is None and (drive_run_dir / "weights" / "last.pt").is_file():
                drive_src_dir = drive_run_dir / "weights"

        if drive_src_dir is not None:
            # Stage restore into isolated temp dir & verify ALL files/SHAs BEFORE publishing into local_weights!
            local_weights = local_run_dir / "weights"
            stage_w = local_run_dir / ".weights_restore_tmp"
            if stage_w.exists():
                shutil.rmtree(stage_w)
            stage_w.mkdir(parents=True, exist_ok=True)
            copied_names = []
            for fname in ("last.pt", "best.pt", "kd_state.pt"):
                src_f = drive_src_dir / fname
                if src_f.is_file():
                    dst_f = stage_w / fname
                    _atomic_copy2(src_f, dst_f)
                    exp_sha = gen_sha_map.get(fname)
                    if exp_sha and _file_sha256(dst_f) != exp_sha:
                        shutil.rmtree(stage_w, ignore_errors=True)
                        raise RuntimeError(
                            f"Drive checkpoint SHA-256 mismatch while restoring {src_f}!"
                        )
                    copied_names.append(fname)
            if "last.pt" not in copied_names:
                shutil.rmtree(stage_w, ignore_errors=True)
                raise FileNotFoundError(f"Drive generation {drive_src_dir} missing last.pt!")
            local_weights.mkdir(parents=True, exist_ok=True)
            for fname in ("last.pt", "best.pt", "kd_state.pt"):
                dst_f = local_weights / fname
                stg_f = stage_w / fname
                if stg_f.is_file():
                    os.replace(stg_f, dst_f)
                elif dst_f.is_file():
                    dst_f.unlink()
            shutil.rmtree(stage_w, ignore_errors=True)
            # Also restore backed-up results.csv so resumed runs preserve early-epoch history!
            drive_csv = drive_run_dir / "results.csv" if drive_run_dir is not None else None
            if drive_csv is not None and drive_csv.is_file():
                _atomic_copy2(drive_csv, local_run_dir / "results.csv")
            resume_ckpt = local_weights / "last.pt"
        elif local_last.is_file():
            resume_ckpt = local_last

        if resume_ckpt is not None:
            import torch

            meta = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
            ckpt_epoch = int(meta.get("epoch", 0)) if isinstance(meta, dict) else 0
            has_opt = bool(isinstance(meta, dict) and meta.get("optimizer") is not None)
            train_args = meta.get("train_args", {}) if isinstance(meta, dict) else {}
            ckpt_epochs = int((train_args or {}).get("epochs", 0) or 0)
            if ckpt_epochs > 0 and ckpt_epochs != int(cfg.epochs):
                print(
                    f"[FRESH] Checkpoint {resume_ckpt} trained for epochs={ckpt_epochs}, "
                    f"but current config requires epochs={cfg.epochs} -> refusing stale resume!"
                )
                resume_ckpt = None
                if local_run_dir.exists():
                    shutil.rmtree(local_run_dir)
            else:
                planned_epochs = ckpt_epochs if ckpt_epochs > 0 else int(cfg.epochs)
                completed_by_epoch = (ckpt_epoch == -1) or (ckpt_epoch + 1 >= planned_epochs)
                if not completed_by_epoch and not has_opt:
                    raise RuntimeError(
                        f"Corrupt mid-run checkpoint {resume_ckpt}: epoch={ckpt_epoch}/{planned_epochs} "
                        f"has optimizer=None (refusing to treat truncated checkpoint as finished)!"
                    )
                if completed_by_epoch:
                    best_cand = local_run_dir / "weights" / "best.pt"
                    if not best_cand.is_file():
                        raise FileNotFoundError(
                            f"Checkpoint {resume_ckpt} finished training (epoch={ckpt_epoch}/{planned_epochs}), "
                            f"but matching best.pt is missing at {best_cand}!"
                        )
                    skip_train_eval_only = True
                    print(
                        f"[RESUME] Detected completed training checkpoint (epoch={ckpt_epoch}/{planned_epochs}, "
                        f"optimizer={has_opt}); skipping re-train and running post-eval + Drive archive!"
                    )
                else:
                    print(
                        f"[RESUME] Resuming interrupted run from: {resume_ckpt} "
                        f"(saved epoch={ckpt_epoch}/{planned_epochs})"
                    )

    if skip_train_eval_only:
        train_minutes = 0.0
    else:
        from src.modules import attach_phase1_callbacks, build_model_with_switches

        # Resolve frozen KD teacher(s) ONLY when actually training (never blocks pure-IR post-eval recovery!)
        if cfg.use_kd:
            cfg.kd_teacher_weights = str(
                _resolve_teacher_ckpt(
                    explicit_weights=cfg.kd_teacher_weights,
                    run_name=cfg.kd_teacher_run,
                    local_project=local_project,
                    local_work_dir=local_work_dir,
                    drive_dir=drive_dir,
                    tag="primary-IR",
                )
            )
            if cfg.kd_aux_teacher_weights or cfg.kd_aux_teacher_run:
                cfg.kd_aux_teacher_weights = str(
                    _resolve_teacher_ckpt(
                        explicit_weights=cfg.kd_aux_teacher_weights,
                        run_name=cfg.kd_aux_teacher_run,
                        local_project=local_project,
                        local_work_dir=local_work_dir,
                        drive_dir=drive_dir,
                        tag="aux-RGB",
                    )
                )
                if not cfg.kd_aux_rgb_root:
                    import yaml

                    y_cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
                    rgb_root = y_cfg.get("rgb_root") or str(dataset_root / "dronevehicle_rgb")
                    cfg.kd_aux_rgb_root = str(rgb_root)
                    print(f"[TRACK B] Resolved paired RGB dataset root: {cfg.kd_aux_rgb_root}")

        if resume_ckpt is not None:
            from ultralytics import YOLO

            model = YOLO(str(resume_ckpt))
            attach_phase1_callbacks(model, cfg)
        else:
            model = build_model_with_switches(cfg, repo_root=repo_root)

        # Register atomic generation Drive sync inside `on_model_save` (runs right after last.pt + kd_state.pt!)
        if drive_run_dir is not None and cfg.save_period > 0:
            def _on_model_save_sync(trainer) -> None:
                epoch_idx = int(getattr(trainer, "epoch", 0)) + 1
                total_ep = int(getattr(trainer, "epochs", cfg.epochs) or cfg.epochs)
                if epoch_idx % cfg.save_period == 0 or epoch_idx == total_ep:
                    _sync_weights_and_csv(
                        Path(trainer.save_dir),
                        drive_run_dir,
                        epoch_idx=epoch_idx,
                        require_kd_state=cfg.use_kd,
                    )
                    print(f"[DRIVE SYNC] Atomically backed up epoch {epoch_idx} generation -> {drive_run_dir}")

            model.add_callback("on_model_save", _on_model_save_sync)

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
                "save_period": -1,
                "plots": True,
            }
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
    snap["config_sha256"] = _config_sha256(cfg)
    (local_run_dir / "config_snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # STRICT fail-fast: formal runs MUST produce `best.pt` (smoke runs may fall back to `last.pt` if 0 fitness)
    best_pt = local_run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        if cfg.smoke_test and (local_run_dir / "weights" / "last.pt").is_file():
            best_pt = local_run_dir / "weights" / "last.pt"
        else:
            raise FileNotFoundError(
                f"Formal training run `{cfg.exp_name}` failed to produce required `best.pt` under {local_run_dir / 'weights'}!"
            )

    eval_summary = evaluate_checkpoint(
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
    if not _is_eval_summary_complete(local_run_dir / "eval_summary.json", cfg):
        raise RuntimeError(f"Post-evaluation summary failed completeness validation for {cfg.exp_name}!")

    if cfg.save_failure_vis:
        save_qualitative_predictions(
            weights_path=best_pt,
            dataset_yaml=dataset_yaml,
            out_dir=local_run_dir / "vis_preds",
            is_e2e_model=cfg.is_e2e_model,
            imgsz=cfg.imgsz,
        )

    # Final full sync to Google Drive & config/weights-bound `_COMPLETED.flag`
    if drive_run_dir is not None:
        print(f"[DRIVE SYNC] Uploading final run artifacts -> {drive_run_dir} ...")
        _sync_tree(local_run_dir, drive_run_dir)
        if not cfg.smoke_test:
            flag_payload = {
                "exp_name": cfg.exp_name,
                "config_sha256": _config_sha256(cfg),
                "weights_sha256": eval_summary.get("weights_sha256", ""),
                "completed_minutes": round(train_minutes, 2),
            }
            tmp_flag = drive_run_dir / "._COMPLETED.flag.tmp"
            tmp_flag.write_text(json.dumps(flag_payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_flag, drive_run_dir / "_COMPLETED.flag")
        print(f"[OK] Run safely archived on Google Drive: {drive_run_dir}")
        return drive_run_dir

    return local_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HIT-UAV / DroneVehicle YOLO26 experiment preset.")
    parser.add_argument("--list", action="store_true", help="List all available experiment presets and exit.")
    parser.add_argument("--exp", type=str, default="p0_yolo26n_musgd", help="Experiment preset name.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    parser.add_argument("--drive-dir", type=Path, default=None, help="Persistent Google Drive root.")
    parser.add_argument("--smoke-test", action="store_true", help="2-epoch fast sanity run.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--dataset-mode", type=str, choices=["4cls", "5cls", "ir", "rgb"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(list_presets())
        return

    # Magic meta-presets: guard against silently ignoring non-default `--seed`!
    if args.exp in {"auto_phase3_overnight", "auto_dv_wave1", "auto_dv_wave2"}:
        if args.seed not in (None, 42):
            parser.error(
                f"Meta-preset '{args.exp}' manages its own seed schedule and does not accept --seed={args.seed} "
                "(use default seed=42 or run individual presets with `@s<seed>`)."
            )
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
        from src.run_dronevehicle_auto import run_dv_wave1, run_dv_wave2

        fn = run_dv_wave1 if args.exp == "auto_dv_wave1" else run_dv_wave2
        res = fn(
            dataset_root=args.dataset_root,
            work_dir=args.work_dir,
            drive_dir=args.drive_dir,
            resume=not args.no_resume,
            smoke_only=bool(args.smoke_test),
            export_onnx=bool(args.export_onnx),
        )
        if str(res.get("status", "")).startswith("INCOMPLETE"):
            raise SystemExit(2)
        return

    # Support inline per-item seed syntax in Colab EXP_LIST: e.g. "v33_e0_base@s0"
    exp_arg = args.exp
    seed_override = args.seed
    if "@s" in exp_arg:
        base_part, seed_part = exp_arg.rsplit("@s", 1)
        if seed_part.isdigit():
            exp_arg = base_part
            seed_override = int(seed_part)

    cfg = get_experiment_config(
        exp_arg,
        smoke_test=args.smoke_test,
        seed=seed_override,
        optimizer=args.optimizer,
        dataset_mode=args.dataset_mode,
        epochs=args.epochs,
        batch=args.batch,
        resume=not args.no_resume,
        export_onnx_bench=args.export_onnx if args.export_onnx else None,
    )

    repo_root = Path(__file__).resolve().parent.parent

    log_files = []
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        if args.drive_dir is not None:
            logs_dir = args.drive_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            for filename in (f"{cfg.exp_name}.log", "latest_run.log"):
                log_files.append((logs_dir / filename).open("w", encoding="utf-8"))

            class _MultiLog(io.TextIOBase):
                def write(self, s: str) -> int:
                    for lf in log_files:
                        if not lf.closed:
                            lf.write(s)
                            lf.flush()
                    return len(s)

                def flush(self) -> None:
                    for lf in log_files:
                        if not lf.closed:
                            lf.flush()

            sink = _MultiLog()
            sys.stdout = _TeeStream(orig_out, sink)
            sys.stderr = _TeeStream(orig_err, sink)

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
        for lf in log_files:
            try:
                lf.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
