"""Deterministic DroneVehicle (RGB + IR) Colab-direct downloader, cropper, HBB converter & Day/Night splitter.

Designed specifically for zero local proxy traffic (runs directly on Colab -> Google Drive cache):
1. Downloads `val.zip`, `test.zip`, `train.zip` from Hugging Face `McCheng/DroneVehicle`
   (or restores in ~25s from a cached cropped `dronevehicle_bundle.zip` on Google Drive).
2. Removes the official 100 px white annotation border (`840x712 -> 640x512`) from BOTH
   paired RGB (`*img`) and LWIR Infrared (`*imgr`) images while verifying strict 1:1 pairing.
3. Parses DroneVehicle VOC XML `<polygon>` (and `<bndbox>`) annotations, normalizes known
   dataset typos (`feright car` / `feright_car` -> `freight_car`), converts OBB -> HBB,
   shifts by `-100 px`, clamps to `[0, 640] x [0, 512]`, and drops outside/degenerate boxes.
4. Generates Ultralytics-native directory structure (zero `img2label_paths` hacks):
   - `<out_root>/dronevehicle_ir/{images,labels}/{train,val,test}`  (canonical IR `*imgr` + `*labelr`)
   - `<out_root>/dronevehicle_rgb/{images,labels}/{train,val,test}` (paired RGB `*img` + `*label`)
   - `<out_root>/dronevehicle_ir.yaml`          (primary IR benchmark & Track B student target)
   - `<out_root>/dronevehicle_rgb.yaml`         (auxiliary RGB teacher benchmark)
   - `<out_root>/dronevehicle_ir_day.yaml`      (Daytime subset evaluation: RGB luminance >= thresh)
   - `<out_root>/dronevehicle_ir_night.yaml`    (Nighttime/Dark subset evaluation: RGB luminance < thresh)
5. Saves side-by-side `(RGB | IR)` sanity visualizations (3 daytime + 3 nighttime) and full
   illumination & class statistics (`illumination_stats.json`) for paper tables.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

HF_REPO_ID = "McCheng/DroneVehicle"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

CLASS_NAMES = ["car", "truck", "bus", "van", "freight_car"]
CLASS_MAP: Dict[str, int] = {
    "car": 0,
    "cars": 0,
    "truck": 1,
    "trucks": 1,
    "truvk": 1,
    "bus": 2,
    "buses": 2,
    "van": 3,
    "vans": 3,
    "freight car": 4,
    "freight_car": 4,
    "freight-car": 4,
    "freightcar": 4,
    "feright car": 4,
    "feright_car": 4,
    "feright-car": 4,
    "ferightcar": 4,
}
IGNORE_LABELS = {"*", "ignore", "ignored", "dontcare", "dont_care", ""}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _download_hf_file(filename: str, dst_path: Path) -> Path:
    """Download a split zip directly inside Colab using huggingface_hub or resumable curl."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.is_file() and dst_path.stat().st_size > 10_000_000:
        print(f"[CACHE] Reusing existing local archive: {dst_path} ({dst_path.stat().st_size / 1e9:.2f} GB)")
        return dst_path

    url = f"{HF_BASE_URL}/{filename}"
    print(f"[DOWNLOAD] Fetching {filename} from {url} ...")

    # Prefer curl with resume (`-C -`) and retries; super fast and reliable on Colab
    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [
            curl_bin,
            "-L",
            "-C",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "-o",
            str(dst_path),
            url,
        ]
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return dst_path

    # Fallback: urllib streaming download
    with urllib.request.urlopen(url, timeout=60) as resp, dst_path.open("wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=4 * 1024 * 1024)
    return dst_path


def _parse_dv_xml_to_hbb(
    xml_path: Path,
    *,
    pad: int,
    out_w: int,
    out_h: int,
    unknown_counter: Counter,
    class_counter: Counter,
) -> List[Tuple[int, float, float, float, float]]:
    """Parse DroneVehicle VOC XML (polygon or bndbox), shift by `-pad`, convert to normalized YOLO HBB."""
    if not xml_path.is_file():
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    boxes: List[Tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            continue
        raw_name = " ".join(name_el.text.strip().lower().split())
        if raw_name in IGNORE_LABELS:
            continue
        cls_id = CLASS_MAP.get(raw_name)
        if cls_id is None:
            unknown_counter[raw_name] += 1
            continue

        xs: List[float] = []
        ys: List[float] = []
        poly = obj.find("polygon")
        if poly is not None:
            for k in ("1", "2", "3", "4"):
                xe = poly.find(f"x{k}")
                ye = poly.find(f"y{k}")
                if xe is not None and ye is not None and xe.text and ye.text:
                    try:
                        xs.append(float(xe.text) - pad)
                        ys.append(float(ye.text) - pad)
                    except ValueError:
                        pass
        if len(xs) < 2:
            bnd = obj.find("bndbox")
            if bnd is not None:
                try:
                    xs = [float(bnd.findtext("xmin", "0")) - pad, float(bnd.findtext("xmax", "0")) - pad]
                    ys = [float(bnd.findtext("ymin", "0")) - pad, float(bnd.findtext("ymax", "0")) - pad]
                except ValueError:
                    continue
        if len(xs) < 2 or len(ys) < 2:
            continue

        raw_x1, raw_x2 = min(xs), max(xs)
        raw_y1, raw_y2 = min(ys), max(ys)
        raw_area = max((raw_x2 - raw_x1) * (raw_y2 - raw_y1), 1e-6)

        x1 = max(0.0, min(float(out_w), raw_x1))
        y1 = max(0.0, min(float(out_h), raw_y1))
        x2 = max(0.0, min(float(out_w), raw_x2))
        y2 = max(0.0, min(float(out_h), raw_y2))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1.5 or bh <= 1.5:
            continue
        # Drop objects almost entirely located inside the cropped 100 px white border
        if (bw * bh) / raw_area < 0.10:
            continue

        xc = (x1 + x2) * 0.5 / float(out_w)
        yc = (y1 + y2) * 0.5 / float(out_h)
        boxes.append((cls_id, xc, yc, bw / float(out_w), bh / float(out_h)))
        class_counter[CLASS_NAMES[cls_id]] += 1
    return boxes


def _write_yolo_txt(path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _find_split_subdirs(extract_dir: Path, split: str) -> Tuple[Path, Path, Path, Path]:
    """Locate (rgb_img_dir, ir_img_dir, rgb_lbl_dir, ir_lbl_dir) inside extracted split archive."""
    all_dirs = [p for p in extract_dir.rglob("*") if p.is_dir()]
    name_map = {p.name.lower(): p for p in all_dirs}

    def _pick(candidates: List[str]) -> Optional[Path]:
        for c in candidates:
            if c in name_map:
                return name_map[c]
        return None

    rgb_img = _pick([f"{split}img", "rgb", "images_rgb"])
    ir_img = _pick([f"{split}imgr", "ir", "infrared", "images_ir"])
    rgb_lbl = _pick([f"{split}label", "labels_rgb", "label"])
    ir_lbl = _pick([f"{split}labelr", "labels_ir", "labelr"])

    if not all([rgb_img, ir_img, ir_lbl]):
        found = sorted(str(p.relative_to(extract_dir)) for p in all_dirs)
        raise FileNotFoundError(
            f"Could not locate DroneVehicle '{split}' subdirs in {extract_dir}. "
            f"Found: {found[:30]}"
        )
    if rgb_lbl is None:
        rgb_lbl = ir_lbl
    return rgb_img, ir_img, rgb_lbl, ir_lbl  # type: ignore[return-value]


def _crop_border_if_needed(im: np.ndarray, border: int = 100) -> Tuple[np.ndarray, int]:
    """Crop official 100 px white border (`840x712 -> 640x512`); pass through if already `640x512`."""
    h, w = im.shape[:2]
    if w == 840 and h == 712 and border == 100:
        return im[border : h - border, border : w - border], border
    if w == 640 and h == 512:
        return im, 0
    if w > 2 * border and h > 2 * border and (w - 2 * border == 640 and h - 2 * border == 512):
        return im[border : h - border, border : w - border], border
    return im, 0


def _process_single_pair(
    stem: str,
    ir_src: Path,
    rgb_src: Path,
    ir_xml: Path,
    rgb_xml: Path,
    out_ir_img: Path,
    out_rgb_img: Path,
    out_ir_lbl: Path,
    out_rgb_lbl: Path,
    jpeg_quality: int,
) -> Optional[Dict[str, object]]:
    ir_im = cv2.imread(str(ir_src), cv2.IMREAD_COLOR)
    rgb_im = cv2.imread(str(rgb_src), cv2.IMREAD_COLOR)
    if ir_im is None or rgb_im is None:
        return None

    ir_crop, pad_ir = _crop_border_if_needed(ir_im, border=100)
    rgb_crop, pad_rgb = _crop_border_if_needed(rgb_im, border=100)

    # Guarantee identical HxW between paired IR & RGB (handles rare edge cases cleanly)
    h_ir, w_ir = ir_crop.shape[:2]
    if rgb_crop.shape[:2] != (h_ir, w_ir):
        rgb_crop = cv2.resize(rgb_crop, (w_ir, h_ir), interpolation=cv2.INTER_LINEAR)

    # Compute exact RGB illumination stats BEFORE JPEG re-save (normalized [0, 1])
    rgb_f = rgb_crop.astype(np.float32) * (1.0 / 255.0)
    lum_mean = float(rgb_f.mean())
    lum_std = float(rgb_f.std())

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    cv2.imwrite(str(out_ir_img), ir_crop, encode_params)
    cv2.imwrite(str(out_rgb_img), rgb_crop, encode_params)

    unk_ir: Counter = Counter()
    cls_ir: Counter = Counter()
    unk_rgb: Counter = Counter()
    cls_rgb: Counter = Counter()

    ir_boxes = _parse_dv_xml_to_hbb(
        ir_xml, pad=pad_ir, out_w=w_ir, out_h=h_ir, unknown_counter=unk_ir, class_counter=cls_ir
    )
    rgb_boxes = _parse_dv_xml_to_hbb(
        rgb_xml, pad=pad_rgb, out_w=w_ir, out_h=h_ir, unknown_counter=unk_rgb, class_counter=cls_rgb
    )
    _write_yolo_txt(out_ir_lbl, ir_boxes)
    _write_yolo_txt(out_rgb_lbl, rgb_boxes)

    return {
        "stem": stem,
        "w": w_ir,
        "h": h_ir,
        "lum_mean": lum_mean,
        "lum_std": lum_std,
        "n_ir": len(ir_boxes),
        "n_rgb": len(rgb_boxes),
        "cls_ir": dict(cls_ir),
        "cls_rgb": dict(cls_rgb),
        "unk": dict(unk_ir + unk_rgb),
    }


def _draw_boxes(img: np.ndarray, lbl_path: Path, color: Tuple[int, int, int]) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    if not lbl_path.is_file():
        return vis
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int(round((xc - bw / 2) * w))
        y1 = int(round((yc - bh / 2) * h))
        x2 = int(round((xc + bw / 2) * w))
        y2 = int(round((yc + bh / 2) * h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        label_s = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
        cv2.putText(vis, label_s, (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return vis


def _save_sanity_visuals(
    ir_root: Path,
    rgb_root: Path,
    val_records: List[Dict[str, object]],
    vis_dir: Path,
    day_thresh: float,
) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)
    day_recs = [r for r in val_records if float(r["lum_mean"]) >= day_thresh and int(r["n_ir"]) > 0]
    night_recs = [r for r in val_records if float(r["lum_mean"]) < day_thresh and int(r["n_ir"]) > 0]
    chosen = [("day", r) for r in day_recs[:3]] + [("night", r) for r in night_recs[:3]]

    for tag, rec in chosen:
        stem = str(rec["stem"])
        ir_im = cv2.imread(str(ir_root / "images" / "val" / f"{stem}.jpg"))
        rgb_im = cv2.imread(str(rgb_root / "images" / "val" / f"{stem}.jpg"))
        if ir_im is None or rgb_im is None:
            continue
        ir_v = _draw_boxes(ir_im, ir_root / "labels" / "val" / f"{stem}.txt", (0, 255, 255))
        rgb_v = _draw_boxes(rgb_im, rgb_root / "labels" / "val" / f"{stem}.txt", (0, 255, 0))
        cv2.putText(
            rgb_v,
            f"RGB ({tag}, mu={float(rec['lum_mean']):.3f}, std={float(rec['lum_std']):.3f})",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            ir_v,
            f"IR (n_ir={rec['n_ir']}, n_rgb={rec['n_rgb']})",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        pair = np.concatenate([rgb_v, ir_v], axis=1)
        cv2.imwrite(str(vis_dir / f"dv_{tag}_{stem}.jpg"), pair)
    print(f"[OK] Saved {len(chosen)} RGB|IR DroneVehicle sanity visuals -> {vis_dir}")


def _write_yamls_and_split_lists(
    out_root: Path,
    split_records: Dict[str, List[Dict[str, object]]],
    *,
    day_lum_thresh: float = 0.15,
) -> Dict[str, object]:
    ir_root = out_root / "dronevehicle_ir"
    rgb_root = out_root / "dronevehicle_rgb"

    stats: Dict[str, object] = {
        "day_lum_thresh": day_lum_thresh,
        "splits": {},
    }

    for split, recs in split_records.items():
        if not recs:
            continue
        day_ir: List[str] = []
        night_ir: List[str] = []
        day_rgb: List[str] = []
        night_rgb: List[str] = []

        lum_vals = [float(r["lum_mean"]) for r in recs]
        for r in recs:
            stem = str(r["stem"])
            ir_rel = f"./images/{split}/{stem}.jpg"
            rgb_rel = f"./images/{split}/{stem}.jpg"
            if float(r["lum_mean"]) >= day_lum_thresh:
                day_ir.append(ir_rel)
                day_rgb.append(rgb_rel)
            else:
                night_ir.append(ir_rel)
                night_rgb.append(rgb_rel)

        (ir_root / f"{split}_day.txt").write_text("\n".join(day_ir) + ("\n" if day_ir else ""), encoding="utf-8")
        (ir_root / f"{split}_night.txt").write_text("\n".join(night_ir) + ("\n" if night_ir else ""), encoding="utf-8")
        (rgb_root / f"{split}_day.txt").write_text("\n".join(day_rgb) + ("\n" if day_rgb else ""), encoding="utf-8")
        (rgb_root / f"{split}_night.txt").write_text("\n".join(night_rgb) + ("\n" if night_rgb else ""), encoding="utf-8")

        cls_ir_tot: Counter = Counter()
        cls_rgb_tot: Counter = Counter()
        unk_tot: Counter = Counter()
        for r in recs:
            cls_ir_tot.update(r["cls_ir"])   # type: ignore[arg-type]
            cls_rgb_tot.update(r["cls_rgb"]) # type: ignore[arg-type]
            unk_tot.update(r["unk"])         # type: ignore[arg-type]

        stats["splits"][split] = {
            "pairs": len(recs),
            "day_pairs": len(day_ir),
            "night_pairs": len(night_ir),
            "frac_below_008": round(sum(1 for v in lum_vals if v < 0.08) / max(len(lum_vals), 1), 4),
            "frac_below_012": round(sum(1 for v in lum_vals if v < 0.12) / max(len(lum_vals), 1), 4),
            "frac_below_015": round(sum(1 for v in lum_vals if v < 0.15) / max(len(lum_vals), 1), 4),
            "frac_below_020": round(sum(1 for v in lum_vals if v < 0.20) / max(len(lum_vals), 1), 4),
            "ir_boxes_total": int(sum(cls_ir_tot.values())),
            "rgb_boxes_total": int(sum(cls_rgb_tot.values())),
            "ir_per_class": dict(cls_ir_tot),
            "rgb_per_class": dict(cls_rgb_tot),
            "unknown_skipped": dict(unk_tot),
        }

    names_dict = {i: n for i, n in enumerate(CLASS_NAMES)}

    def _dump_yaml(path: Path, payload: Dict[str, object]) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # 1. Primary IR dataset YAML (with paired `rgb_root` metadata for Track B!)
    _dump_yaml(
        out_root / "dronevehicle_ir.yaml",
        {
            "path": str(ir_root.resolve()),
            "rgb_root": str(rgb_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(CLASS_NAMES),
            "names": names_dict,
        },
    )
    # 2. Auxiliary RGB Teacher dataset YAML
    _dump_yaml(
        out_root / "dronevehicle_rgb.yaml",
        {
            "path": str(rgb_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(CLASS_NAMES),
            "names": names_dict,
        },
    )
    # 3. Daytime & Nighttime IR evaluation YAMLs (zero image duplication!)
    for tag in ("day", "night"):
        _dump_yaml(
            out_root / f"dronevehicle_ir_{tag}.yaml",
            {
                "path": str(ir_root.resolve()),
                "train": "images/train",
                "val": f"val_{tag}.txt",
                "test": f"test_{tag}.txt",
                "nc": len(CLASS_NAMES),
                "names": names_dict,
            },
        )
        _dump_yaml(
            out_root / f"dronevehicle_rgb_{tag}.yaml",
            {
                "path": str(rgb_root.resolve()),
                "train": "images/train",
                "val": f"val_{tag}.txt",
                "test": f"test_{tag}.txt",
                "nc": len(CLASS_NAMES),
                "names": names_dict,
            },
        )

    stats_path = ir_root / "illumination_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote DroneVehicle YAMLs + Day/Night splits + stats -> {stats_path}")
    return stats


def _bundle_to_drive(out_root: Path, bundle_path: Path) -> None:
    """Create a fast uncompressed/low-compression zip bundle on Google Drive for rapid Colab restores."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_root / "_dv_bundle_tmp.zip"
    print(f"[BUNDLE] Packing cropped DroneVehicle -> {bundle_path} (STORED mode for max speed) ...")
    t0 = time.perf_counter()
    # JPEGs are already compressed; ZIP_STORED is 10x faster to pack & unpack!
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for sub in ("dronevehicle_ir", "dronevehicle_rgb"):
            root_sub = out_root / sub
            if not root_sub.is_dir():
                continue
            for p in root_sub.rglob("*"):
                if p.is_file() and not p.name.endswith(".cache"):
                    zf.write(p, arcname=str(p.relative_to(out_root)))
        for y in out_root.glob("dronevehicle_*.yaml"):
            zf.write(y, arcname=y.name)
    shutil.move(str(tmp_zip), str(bundle_path))
    print(
        f"[OK] Saved DroneVehicle bundle ({bundle_path.stat().st_size / 1e9:.2f} GB) "
        f"in {time.perf_counter() - t0:.1f}s -> {bundle_path}"
    )


def _restore_from_bundle(out_root: Path, bundle_path: Path) -> bool:
    if not bundle_path.is_file():
        return False
    ir_yaml = out_root / "dronevehicle_ir.yaml"
    rgb_yaml = out_root / "dronevehicle_rgb.yaml"
    if ir_yaml.is_file() and rgb_yaml.is_file() and (out_root / "dronevehicle_ir" / "images" / "train").is_dir():
        n_tr = len(list((out_root / "dronevehicle_ir" / "images" / "train").glob("*.jpg")))
        if n_tr > 1000:
            print(f"[FAST] DroneVehicle already unpacked at {out_root} ({n_tr} train IR images).")
            return True

    print(f"[RESTORE] Unpacking cached DroneVehicle bundle from Drive: {bundle_path} ...")
    t0 = time.perf_counter()
    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        zf.extractall(out_root)

    # Rewrite YAML `path` fields so they point cleanly to `out_root.resolve()`
    ir_root = out_root / "dronevehicle_ir"
    rgb_root = out_root / "dronevehicle_rgb"
    for y_path in out_root.glob("dronevehicle_*.yaml"):
        cfg = yaml.safe_load(y_path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict):
            if "_rgb" in y_path.name:
                cfg["path"] = str(rgb_root.resolve())
            else:
                cfg["path"] = str(ir_root.resolve())
                if "rgb_root" in cfg:
                    cfg["rgb_root"] = str(rgb_root.resolve())
            y_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"[OK] Restored DroneVehicle in {time.perf_counter() - t0:.1f}s -> {out_root}")
    return True


def prepare_dronevehicle(
    *,
    out_root: Path,
    drive_bundle: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    vis_dir: Optional[Path] = None,
    splits: Tuple[str, ...] = ("val", "test", "train"),
    jpeg_quality: int = 95,
    day_lum_thresh: float = 0.15,
    workers: int = 8,
    keep_raw: bool = False,
) -> Dict[str, object]:
    out_root.mkdir(parents=True, exist_ok=True)
    if drive_bundle is not None and _restore_from_bundle(out_root, drive_bundle):
        stats_p = out_root / "dronevehicle_ir" / "illumination_stats.json"
        return json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.is_file() else {}

    work_raw = raw_dir if raw_dir is not None else (out_root / "_dv_raw_tmp")
    work_raw.mkdir(parents=True, exist_ok=True)

    ir_root = out_root / "dronevehicle_ir"
    rgb_root = out_root / "dronevehicle_rgb"
    split_records: Dict[str, List[Dict[str, object]]] = {}

    for split in splits:
        t0 = time.perf_counter()
        print("\n" + "=" * 72)
        print(f"[DRONEVEHICLE] Preparing split: {split}")
        print("=" * 72)

        extract_dir = work_raw / f"extracted_{split}"
        zip_path = work_raw / f"{split}.zip"
        if not extract_dir.is_dir() or not any(extract_dir.iterdir()):
            _download_hf_file(f"{split}.zip", zip_path)
            print(f"[UNZIP] Extracting {zip_path.name} -> {extract_dir} ...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            if not keep_raw and raw_dir is None:
                zip_path.unlink(missing_ok=True)

        rgb_img_dir, ir_img_dir, rgb_lbl_dir, ir_lbl_dir = _find_split_subdirs(extract_dir, split)
        ir_files = {p.stem: p for p in ir_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        rgb_files = {p.stem: p for p in rgb_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        common_stems = sorted(set(ir_files.keys()) & set(rgb_files.keys()))
        print(
            f"[INFO] {split}: found {len(ir_files)} IR, {len(rgb_files)} RGB -> "
            f"{len(common_stems)} strict 1:1 paired frames."
        )
        if not common_stems:
            raise RuntimeError(f"Zero matched RGB-IR pairs found for DroneVehicle split '{split}'!")

        out_ir_img_dir = ir_root / "images" / split
        out_ir_lbl_dir = ir_root / "labels" / split
        out_rgb_img_dir = rgb_root / "images" / split
        out_rgb_lbl_dir = rgb_root / "labels" / split
        for d in (out_ir_img_dir, out_ir_lbl_dir, out_rgb_img_dir, out_rgb_lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        def _worker(st: str) -> Optional[Dict[str, object]]:
            return _process_single_pair(
                st,
                ir_files[st],
                rgb_files[st],
                ir_lbl_dir / f"{st}.xml",
                rgb_lbl_dir / f"{st}.xml",
                out_ir_img_dir / f"{st}.jpg",
                out_rgb_img_dir / f"{st}.jpg",
                out_ir_lbl_dir / f"{st}.txt",
                out_rgb_lbl_dir / f"{st}.txt",
                jpeg_quality=jpeg_quality,
            )

        recs: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for idx, res in enumerate(ex.map(_worker, common_stems), 1):
                if res is not None:
                    recs.append(res)
                if idx % 2000 == 0 or idx == len(common_stems):
                    print(f"  [{split}] processed {idx}/{len(common_stems)} pairs ({time.perf_counter() - t0:.1f}s)")

        split_records[split] = recs

        # Free extracted raw files immediately after converting this split to save Colab SSD space!
        if not keep_raw and raw_dir is None:
            shutil.rmtree(extract_dir, ignore_errors=True)

    if not keep_raw and raw_dir is None:
        shutil.rmtree(work_raw, ignore_errors=True)

    stats = _write_yamls_and_split_lists(out_root, split_records, day_lum_thresh=day_lum_thresh)
    if vis_dir is not None and split_records.get("val"):
        _save_sanity_visuals(ir_root, rgb_root, split_records["val"], vis_dir, day_lum_thresh)

    if drive_bundle is not None:
        _bundle_to_drive(out_root, drive_bundle)

    print("\n[SUMMARY] DroneVehicle preparation complete:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DroneVehicle (640x512 RGB+IR) directly on Colab.")
    parser.add_argument("--out-root", type=Path, default=Path("/content/datasets"))
    parser.add_argument("--drive-bundle", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional pre-extracted raw directory.")
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--day-thresh", type=float, default=0.15)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    prepare_dronevehicle(
        out_root=args.out_root,
        drive_bundle=args.drive_bundle,
        raw_dir=args.raw_dir,
        vis_dir=args.vis_dir,
        jpeg_quality=args.jpeg_quality,
        day_lum_thresh=args.day_thresh,
        workers=args.workers,
        keep_raw=args.keep_raw,
    )


if __name__ == "__main__":
    main()
