"""Deterministic HIT-UAV dataset downloader, converter, validator, and Drive bundler.

Key guarantees:
1. Preserves the official HIT-UAV Hold-out split (Train 2,029 / Val 290 / Test 579).
2. Auto-detects official VOC XML (`normal_xml`, preferred unambiguous format),
   COCO JSON (`normal_json`), or pre-split YOLO directories (e.g., Kaggle mirror).
3. Generates BOTH:
   - `hit_uav_4cls.yaml` (Primary: Person, Car, Bicycle, OtherVehicle; drops DontCare)
   - `hit_uav_5cls.yaml` (Appendix: includes DontCare)
   sharing a single copy of images via lightweight symlinks.
4. Prints full scale/class statistics and saves 4 annotated sanity-check images so
   bounding-box alignment can be visually verified before burning GPU compute.
5. Caches a single `hit_uav_bundle.zip` to Google Drive for ~3s restores on future Colab sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

OFFICIAL_REPO_URL = "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git"

CLASS_MAP_5 = {
    "person": 0,
    "car": 1,
    "bicycle": 2,
    "othervehicle": 3,
    "other_vehicle": 3,
    "dontcare": 4,
    "dont_care": 4,
}
CLASS_NAMES_5 = ["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"]
CLASS_NAMES_4 = ["Person", "Car", "Bicycle", "OtherVehicle"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _run(cmd: List[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _index_images(root: Path) -> Dict[str, Path]:
    """Recursively index all image files by stem (and filename)."""
    by_stem: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            # Prefer non-hidden files; ignore git metadata
            if ".git" in p.parts:
                continue
            by_stem.setdefault(p.stem, p)
    return by_stem


def _read_image_size(img_path: Path) -> Tuple[int, int]:
    """Read (width, height) using Pillow (or OpenCV fallback)."""
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            return im.size  # (W, H)
    except Exception:
        import cv2

        arr = cv2.imread(str(img_path))
        if arr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = arr.shape[:2]
        return w, h


def _clamp_box(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.5 or bh <= 0.5:
        return None
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    return xc, yc, bw / w, bh / h


def _parse_voc_xml_dir(
    raw_dir: Path,
) -> Optional[Dict[str, List[Tuple[Path, int, int, List[Tuple[int, float, float, float, float]]]]]]:
    """Parse official `normal_xml` directory if present."""
    # Locate ImageSets/Main/{train,val,test}.txt and Annotations/
    split_txts: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidates = [
            p
            for p in raw_dir.rglob(f"{split}.txt")
            if "rotate" not in str(p).lower() and ".git" not in p.parts
        ]
        if candidates:
            # Prefer paths mentioning normal_xml or ImageSets
            candidates.sort(
                key=lambda p: (
                    0 if "normal_xml" in str(p).lower() else 1,
                    0 if "imagesets" in str(p).lower() else 1,
                    len(str(p)),
                )
            )
            split_txts[split] = candidates[0]

    ann_dirs = [
        p
        for p in raw_dir.rglob("Annotations")
        if p.is_dir() and "rotate" not in str(p).lower()
    ]
    if not ann_dirs or len(split_txts) < 3:
        return None

    ann_dirs.sort(key=lambda p: (0 if "normal_xml" in str(p).lower() else 1, len(str(p))))
    ann_dir = ann_dirs[0]
    img_index = _index_images(raw_dir)
    if not img_index:
        return None

    print(f"[INFO] Detected VOC XML layout: annotations={ann_dir}, splits={split_txts}")
    parsed: Dict[str, List[Tuple[Path, int, int, List[Tuple[int, float, float, float, float]]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    missing_imgs = 0
    for split in ("train", "val", "test"):
        stems = [
            line.strip()
            for line in split_txts[split].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for stem in stems:
            stem_clean = Path(stem).stem
            img_path = img_index.get(stem_clean)
            if img_path is None:
                missing_imgs += 1
                continue
            w, h = _read_image_size(img_path)
            xml_path = ann_dir / f"{stem_clean}.xml"
            boxes: List[Tuple[int, float, float, float, float]] = []
            if xml_path.exists():
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name_el = obj.find("name")
                    bnd = obj.find("bndbox")
                    if name_el is None or bnd is None or not name_el.text:
                        continue
                    cname = name_el.text.strip().lower().replace(" ", "")
                    if cname not in CLASS_MAP_5:
                        raise ValueError(f"Unexpected class name '{name_el.text}' in {xml_path}")
                    cid = CLASS_MAP_5[cname]
                    x1 = float(bnd.findtext("xmin", "0"))
                    y1 = float(bnd.findtext("ymin", "0"))
                    x2 = float(bnd.findtext("xmax", "0"))
                    y2 = float(bnd.findtext("ymax", "0"))
                    norm = _clamp_box(x1, y1, x2, y2, w, h)
                    if norm is not None:
                        boxes.append((cid, *norm))
            parsed[split].append((img_path, w, h, boxes))

    if missing_imgs > 0:
        print(f"[WARN] Skipped {missing_imgs} missing image entries in VOC split lists.")
    total = sum(len(v) for v in parsed.values())
    if total == 0:
        return None
    return parsed


def _parse_coco_json_dir(
    raw_dir: Path,
) -> Optional[Dict[str, List[Tuple[Path, int, int, List[Tuple[int, float, float, float, float]]]]]]:
    """Parse official `normal_json` COCO-style train/val/test JSON files if present."""
    json_files: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidates = [
            p
            for p in raw_dir.rglob(f"{split}.json")
            if "rotate" not in str(p).lower() and ".git" not in p.parts
        ]
        if candidates:
            candidates.sort(key=lambda p: (0 if "normal_json" in str(p).lower() else 1, len(str(p))))
            json_files[split] = candidates[0]

    if len(json_files) < 3:
        return None

    img_index = _index_images(raw_dir)
    if not img_index:
        return None

    print(f"[INFO] Detected COCO JSON layout: {json_files}")
    parsed: Dict[str, List[Tuple[Path, int, int, List[Tuple[int, float, float, float, float]]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for split in ("train", "val", "test"):
        data = json.loads(json_files[split].read_text(encoding="utf-8"))
        # Build category_id -> canonical 0..4 class id
        cat_to_cid: Dict[int, int] = {}
        for cat in data.get("categories", []):
            cname = str(cat["name"]).strip().lower().replace(" ", "")
            if cname in CLASS_MAP_5:
                cat_to_cid[int(cat["id"])] = CLASS_MAP_5[cname]

        anns_by_img: Dict[int, List[dict]] = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_img[int(ann["image_id"])].append(ann)

        for img_info in data.get("images", []):
            stem = Path(img_info["file_name"]).stem
            img_path = img_index.get(stem)
            if img_path is None:
                continue
            w = int(img_info.get("width") or 0)
            h = int(img_info.get("height") or 0)
            if w <= 0 or h <= 0:
                w, h = _read_image_size(img_path)

            boxes: List[Tuple[int, float, float, float, float]] = []
            for ann in anns_by_img.get(int(img_info["id"]), []):
                raw_cat = int(ann["category_id"])
                cid = cat_to_cid.get(raw_cat, raw_cat)
                if cid not in range(5):
                    continue
                bx, by, bw, bh = [float(v) for v in ann["bbox"][:4]]
                # Standard COCO uses [x_min, y_min, w, h]
                norm = _clamp_box(bx, by, bx + bw, by + bh, w, h)
                if norm is not None:
                    boxes.append((cid, *norm))
            parsed[split].append((img_path, w, h, boxes))

    total = sum(len(v) for v in parsed.values())
    if total == 0:
        return None
    return parsed


def _parse_presplit_yolo_dir(
    raw_dir: Path,
) -> Optional[Dict[str, List[Tuple[Path, int, int, List[Tuple[int, float, float, float, float]]]]]]:
    """Parse already-split YOLO directory (images/{train,val,test} + labels/{train,val,test})."""
    img_roots = [p for p in raw_dir.rglob("images") if p.is_dir() and (p / "train").is_dir()]
    if not img_roots:
        return None
    img_root = img_roots[0]
    lbl_root = img_root.parent / "labels"
    if not lbl_root.is_dir():
        return None

    print(f"[INFO] Detected pre-split YOLO layout: {img_root.parent}")
    parsed: Dict[str, List[Tuple[Path, int, int, List[Tuple[int, float, float, float, float]]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for split in ("train", "val", "test"):
        split_img_dir = img_root / split
        split_lbl_dir = lbl_root / split
        if not split_img_dir.is_dir():
            continue
        for img_path in sorted(split_img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            w, h = _read_image_size(img_path)
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            boxes: List[Tuple[int, float, float, float, float]] = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    xc, yc, bw, bh = [float(v) for v in parts[1:5]]
                    boxes.append((cid, xc, yc, bw, bh))
            parsed[split].append((img_path, w, h, boxes))

    total = sum(len(v) for v in parsed.values())
    if total == 0:
        return None
    return parsed


def _build_yaml_and_symlinks(bundle_dir: Path, out_root: Path) -> Tuple[Path, Path]:
    """Create lightweight sibling symlink views (`hit_uav_4cls`, `hit_uav_5cls`) + YAMLs."""
    yaml_paths = {}
    configs = [
        ("4cls", "labels_4cls", CLASS_NAMES_4),
        ("5cls", "labels_5cls", CLASS_NAMES_5),
    ]
    for tag, lbl_folder, names in configs:
        view_dir = out_root / f"hit_uav_{tag}"
        if view_dir.exists() or view_dir.is_symlink():
            shutil.rmtree(view_dir)
        view_dir.mkdir(parents=True, exist_ok=True)
        os.symlink((bundle_dir / "images").resolve(), view_dir / "images", target_is_directory=True)
        os.symlink((bundle_dir / lbl_folder).resolve(), view_dir / "labels", target_is_directory=True)

        yaml_obj = {
            "path": str(view_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(names),
            "names": {i: n for i, n in enumerate(names)},
        }
        yaml_path = out_root / f"hit_uav_{tag}.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_obj, sort_keys=False), encoding="utf-8")
        yaml_paths[tag] = yaml_path
        print(f"[OK] Wrote dataset YAML: {yaml_path}")

    return yaml_paths["4cls"], yaml_paths["5cls"]


def _render_sanity_visuals(bundle_dir: Path, vis_dir: Path, max_per_split: int = 2) -> None:
    """Save annotated ground-truth overlays to visually verify box coordinates."""
    try:
        import cv2
    except ImportError:
        print("[WARN] OpenCV not installed; skipping sanity visualizations.")
        return

    colors = {
        0: (0, 255, 0),      # Person: Green
        1: (255, 165, 0),    # Car: Blue-Orange
        2: (0, 255, 255),    # Bicycle: Yellow
        3: (255, 0, 255),    # OtherVehicle: Magenta
        4: (160, 160, 160),  # DontCare: Gray
    }
    vis_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        img_dir = bundle_dir / "images" / split
        lbl_dir = bundle_dir / "labels_5cls" / split
        saved = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(lines) < 4:
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            for ln in lines:
                cid, xc, yc, bw, bh = ln.split()[:5]
                cid_i = int(cid)
                x1 = int((float(xc) - float(bw) / 2.0) * w)
                y1 = int((float(yc) - float(bh) / 2.0) * h)
                x2 = int((float(xc) + float(bw) / 2.0) * w)
                y2 = int((float(yc) + float(bh) / 2.0) * h)
                color = colors.get(cid_i, (255, 255, 255))
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    im,
                    CLASS_NAMES_5[cid_i],
                    (max(0, x1), max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            out_img = vis_dir / f"sanity_{split}_{saved}_{img_path.stem}.png"
            cv2.imwrite(str(out_img), im)
            saved += 1
            if saved >= max_per_split:
                break
    print(f"[OK] Saved GT sanity visualizations to {vis_dir}")


def _compute_and_print_stats(bundle_dir: Path) -> dict:
    """Compute exact split counts, class counts, and COCO-style scale distribution."""
    stats = {
        "splits": {},
        "class_counts_5cls": Counter(),
        "scale_counts_4cls": Counter(),
        "resolutions": Counter(),
    }
    for split in ("train", "val", "test"):
        img_dir = bundle_dir / "images" / split
        lbl5_dir = bundle_dir / "labels_5cls" / split
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS] if img_dir.exists() else []
        empty_imgs = 0
        split_boxes = 0
        for img_path in imgs:
            w, h = _read_image_size(img_path)
            stats["resolutions"][f"{w}x{h}"] += 1
            lbl_path = lbl5_dir / f"{img_path.stem}.txt"
            lines = (
                [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if lbl_path.exists()
                else []
            )
            if not lines:
                empty_imgs += 1
            for ln in lines:
                parts = ln.split()
                cid = int(parts[0])
                bw_px = float(parts[3]) * w
                bh_px = float(parts[4]) * h
                area = bw_px * bh_px
                stats["class_counts_5cls"][CLASS_NAMES_5[cid]] += 1
                split_boxes += 1
                if cid != 4:  # Valid 4-class scale statistics
                    if area < 32 * 32:
                        stats["scale_counts_4cls"]["small(<32x32)"] += 1
                    elif area < 96 * 96:
                        stats["scale_counts_4cls"]["medium(32-96)"] += 1
                    else:
                        stats["scale_counts_4cls"]["large(>96x96)"] += 1
        stats["splits"][split] = {
            "images": len(imgs),
            "empty_images": empty_imgs,
            "boxes_5cls": split_boxes,
        }

    # Convert Counters for JSON serialization
    out = {
        "splits": stats["splits"],
        "class_counts_5cls": dict(stats["class_counts_5cls"]),
        "scale_counts_4cls": dict(stats["scale_counts_4cls"]),
        "resolutions": dict(stats["resolutions"]),
    }

    print("\n" + "=" * 68)
    print("HIT-UAV DATASET INTEGRITY AUDIT")
    print("=" * 68)
    for sp, info in out["splits"].items():
        print(f"  {sp:<6}: {info['images']:>5} images (empty={info['empty_images']:>2}), {info['boxes_5cls']:>6} boxes (5cls)")
    total_4 = sum(out["scale_counts_4cls"].values()) or 1
    print("\nScale Distribution (4 valid classes):")
    for k, v in out["scale_counts_4cls"].items():
        print(f"  {k:<16}: {v:>6} ({100.0 * v / total_4:5.2f}%)")
    print("\nPer-Class Counts (5 classes):")
    total_5 = sum(out["class_counts_5cls"].values()) or 1
    for cname in CLASS_NAMES_5:
        v = out["class_counts_5cls"].get(cname, 0)
        print(f"  {cname:<14}: {v:>6} ({100.0 * v / total_5:5.2f}%)")
    print(f"\nImage Resolutions: {out['resolutions']}")

    expected = {"train": 2029, "val": 290, "test": 579}
    actual = {sp: out["splits"][sp]["images"] for sp in ("train", "val", "test")}
    if actual != expected:
        print(
            f"[WARN] Split counts {actual} differ from canonical HIT-UAV {expected}! "
            "Verify your raw dataset source before publishing benchmark comparisons."
        )
    else:
        print("[OK] Split counts match canonical official HIT-UAV (2,029 / 290 / 579)!")
    print("=" * 68 + "\n")

    (bundle_dir / "dataset_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _zip_bundle(bundle_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    print(f"[INFO] Archiving HIT-UAV bundle to {zip_path} ...")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(bundle_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(bundle_dir))
    tmp_zip.replace(zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved Drive cache bundle ({size_mb:.1f} MB): {zip_path}")


def _extract_bundle(zip_path: Path, bundle_dir: Path) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Fast-restoring HIT-UAV bundle from {zip_path} -> {bundle_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(bundle_dir)
    print("[OK] Bundle extracted.")


def prepare_hituav(
    out_root: Path,
    raw_dir: Optional[Path] = None,
    drive_bundle: Optional[Path] = None,
    vis_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[Path, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = out_root / "hit_uav_bundle"

    # Fast path 1: Already prepared on local disk (/content/datasets/)
    if not force_rebuild and (bundle_dir / "images" / "train").is_dir() and (bundle_dir / "labels_4cls" / "train").is_dir():
        print(f"[OK] Found existing prepared dataset at {bundle_dir}")
        y4, y5 = _build_yaml_and_symlinks(bundle_dir, out_root)
        return y4, y5

    # Fast path 2: Cached bundle exists on Google Drive -> unzip to fast local SSD
    if not force_rebuild and drive_bundle is not None and drive_bundle.is_file():
        _extract_bundle(drive_bundle, bundle_dir)
        y4, y5 = _build_yaml_and_symlinks(bundle_dir, out_root)
        _compute_and_print_stats(bundle_dir)
        return y4, y5

    # Full build path: clone official GitHub repo or use provided raw_dir/zip
    work_raw = out_root / "_raw_hituav"
    if raw_dir is not None:
        if raw_dir.is_file() and raw_dir.suffix.lower() == ".zip":
            if work_raw.exists():
                shutil.rmtree(work_raw)
            work_raw.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(raw_dir, "r") as zf:
                zf.extractall(work_raw)
            src_dir = work_raw
        else:
            src_dir = raw_dir
    else:
        if not work_raw.exists():
            _run(["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(work_raw)])
        src_dir = work_raw

    # Auto-extract any nested .zip archives (e.g. if repo ships normal_xml.zip / images.zip)
    for nested_zip in sorted(src_dir.rglob("*.zip")):
        if ".git" in nested_zip.parts:
            continue
        extract_target = nested_zip.parent / nested_zip.stem
        if not extract_target.exists():
            print(f"[INFO] Extracting nested archive: {nested_zip} -> {extract_target}")
            with zipfile.ZipFile(nested_zip, "r") as zf:
                zf.extractall(extract_target)

    # Auto-detect format (prefer unambiguous VOC XML -> COCO JSON -> pre-split YOLO)
    parsed = _parse_voc_xml_dir(src_dir)
    if parsed is None:
        parsed = _parse_coco_json_dir(src_dir)
    if parsed is None:
        parsed = _parse_presplit_yolo_dir(src_dir)
    if parsed is None:
        raise RuntimeError(
            f"Could not detect HIT-UAV images & annotations inside {src_dir}.\n"
            "Note: If the official GitHub repo only contains annotation metadata without raw JPEGs, "
            "download the raw HIT-UAV archive (Zenodo/Kaggle/GitHub release) to Google Drive and pass "
            "`--raw-dir /content/drive/MyDrive/.../hit-uav.zip`."
        )

    # Materialize clean unified bundle
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    for split in ("train", "val", "test"):
        (bundle_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (bundle_dir / "labels_4cls" / split).mkdir(parents=True, exist_ok=True)
        (bundle_dir / "labels_5cls" / split).mkdir(parents=True, exist_ok=True)

        for img_path, _w, _h, boxes in parsed[split]:
            dst_img = bundle_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            lines_5 = [f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cid, xc, yc, bw, bh in boxes]
            lines_4 = [
                f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                for cid, xc, yc, bw, bh in boxes
                if cid != 4
            ]
            (bundle_dir / "labels_5cls" / split / f"{img_path.stem}.txt").write_text(
                "\n".join(lines_5), encoding="utf-8"
            )
            (bundle_dir / "labels_4cls" / split / f"{img_path.stem}.txt").write_text(
                "\n".join(lines_4), encoding="utf-8"
            )

    y4, y5 = _build_yaml_and_symlinks(bundle_dir, out_root)
    _compute_and_print_stats(bundle_dir)

    if vis_dir is not None:
        _render_sanity_visuals(bundle_dir, vis_dir)
    else:
        _render_sanity_visuals(bundle_dir, out_root / "sanity_vis")

    if drive_bundle is not None:
        _zip_bundle(bundle_dir, drive_bundle)

    return y4, y5


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HIT-UAV dataset in 4-class and 5-class YOLO formats.")
    parser.add_argument("--out-root", type=Path, default=Path("/content/datasets"), help="Fast local dataset root.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional pre-downloaded raw dir or .zip file.")
    parser.add_argument(
        "--drive-bundle",
        type=Path,
        default=None,
        help="Optional Google Drive zip cache path (e.g., /content/drive/MyDrive/ir-yolo26-paper/datasets/hit_uav_bundle.zip).",
    )
    parser.add_argument("--vis-dir", type=Path, default=None, help="Where to save 4 GT sanity-check PNGs.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore caches and rebuild from scratch.")
    args = parser.parse_args()

    y4, y5 = prepare_hituav(
        out_root=args.out_root,
        raw_dir=args.raw_dir,
        drive_bundle=args.drive_bundle,
        vis_dir=args.vis_dir,
        force_rebuild=args.force_rebuild,
    )
    print(f"[DONE] 4-class YAML: {y4}")
    print(f"[DONE] 5-class YAML: {y5}")


if __name__ == "__main__":
    main()
