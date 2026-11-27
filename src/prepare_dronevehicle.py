"""Deterministic DroneVehicle (RGB + IR) Colab-direct downloader, cropper, HBB converter & Day/Night splitter.

Hardened against partial downloads, corrupt zips/XMLs, failed image writes, and stale caches:
1. Downloads `val.zip`, `test.zip`, `train.zip` from `McCheng/DroneVehicle` to `.part` files using
   `curl --fail -L -C -`, verifies exact `Content-Length` + ZIP readability before atomic rename.
2. Checks local SSD free space before extracting into an isolated temp directory; NEVER skips
   extraction based on a non-empty partial directory.
3. Requires ALL FOUR source folders (`*img`, `*imgr`, `*label`, `*labelr`), strict 1:1 pairing
   (`ir_stems == rgb_stems`), valid XML parsing (failing fast on missing/corrupt XML while allowing
   legitimate empty `<annotation>`), identical post-crop IR/RGB geometry, and verified `cv2.imwrite`.
4. Writes canonical `dv_manifest.json` and only publishes `dronevehicle_bundle.zip` via atomic
   rename inside `bundle_path.parent` after all splits succeed. Fast-path restore verifies the
   full manifest and always rewrites YAML `path`/`rgb_root` to match `out_root.resolve()`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
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
    "feright": 4,
    "freight": 4,
}
IGNORE_LABELS = {"*", "ignore", "ignored", "dontcare", "dont_care", ""}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _remote_content_length(url: str) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


def _is_valid_zip(path: Path, expected_size: Optional[int] = None) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return len(zf.infolist()) > 0
    except Exception:
        return False


def _download_hf_file(filename: str, dst_path: Path) -> Path:
    """Resumable atomic download to `.part` with `curl --fail`, size check, and ZIP verification."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE_URL}/{filename}"
    expected_size = _remote_content_length(url)

    if _is_valid_zip(dst_path, expected_size=expected_size):
        print(f"[CACHE] Verified existing archive: {dst_path} ({dst_path.stat().st_size / 1e9:.2f} GB)")
        return dst_path

    part_path = dst_path.with_suffix(dst_path.suffix + ".part")
    print(f"[DOWNLOAD] Fetching {filename} from {url} (expected={expected_size or 'unknown'} B) ...")

    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [
            curl_bin,
            "--fail",
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
            str(part_path),
            url,
        ]
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as resp, part_path.open("wb") as out_f:
            shutil.copyfileobj(resp, out_f, length=4 * 1024 * 1024)

    if not _is_valid_zip(part_path, expected_size=expected_size):
        actual = part_path.stat().st_size if part_path.is_file() else 0
        raise RuntimeError(
            f"Downloaded archive '{part_path}' failed integrity check "
            f"(actual={actual} B vs expected={expected_size} B)!"
        )
    os.replace(part_path, dst_path)
    return dst_path


def _dv_vertices(obj: ET.Element, xml_path: Path) -> List[Tuple[float, float]]:
    """Return source-image vertices, or an empty list for a point annotation."""
    def coord(node: ET.Element, field: str) -> float:
        raw = node.findtext(field)
        try:
            value = float(raw or "")
            if math.isfinite(value):
                return value
        except ValueError:
            pass
        raise ValueError(f"Malformed XML {xml_path}: invalid <{node.tag}>/{field} coordinate {raw!r}")

    def numbered(node: ET.Element) -> List[Tuple[float, float]]:
        indices = dict.fromkeys(
            child.tag[1:] for child in node
            if child.tag.startswith(("x", "y")) and child.tag[1:].isdigit()
        )
        return [(coord(node, f"x{i}"), coord(node, f"y{i}")) for i in indices]

    poly = obj.find("polygon")
    if poly is not None:
        vertices = numbered(poly)
        if len(vertices) < 2:
            raise ValueError(f"Malformed XML {xml_path}: <polygon> needs at least two vertices")
        return vertices

    box = obj.find("bndbox")
    if box is not None:
        return [(coord(box, "xmin"), coord(box, "ymin")), (coord(box, "xmax"), coord(box, "ymax"))]

    rotated = obj.find("robndbox")
    if rotated is not None:
        cx, cy, w, h, angle = (coord(rotated, key) for key in ("cx", "cy", "w", "h", "angle"))
        c, s = math.cos(angle), math.sin(angle)
        return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
                for dx in (-w / 2, w / 2) for dy in (-h / 2, h / 2)]

    points = obj.find("points")
    if points is not None:
        vertices = numbered(points)
        if points.find("x") is not None or points.find("y") is not None:
            vertices.append((coord(points, "x"), coord(points, "y")))
        vertices.extend((coord(pt, "x"), coord(pt, "y")) for pt in points.findall("point"))
        return vertices if len(vertices) >= 2 else []

    point = obj.find("point")
    if point is not None:
        coord(point, "x")
        coord(point, "y")
        return []
    if obj.find("line") is not None:
        return []
    raise ValueError(f"Malformed XML {xml_path}: object missing recognized geometry")


def _parse_dv_xml_to_hbb(
    xml_path: Path,
    *,
    pad: int,
    out_w: int,
    out_h: int,
    unknown_counter: Counter,
    class_counter: Counter,
    skipped_counter: Optional[Counter] = None,
) -> List[Tuple[int, float, float, float, float]]:
    """Parse DroneVehicle VOC XML, shift by `-pad`, convert to normalized YOLO HBB.

    Fails fast on missing or corrupt XML; legitimate empty `<annotation>` with 0 objects returns `[]`.
    """
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing DroneVehicle XML annotation: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Corrupt DroneVehicle XML annotation '{xml_path}': {exc}") from exc

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

        vertices = _dv_vertices(obj, xml_path)
        if not vertices:
            if skipped_counter is not None:
                skipped_counter["point"] += 1
            continue

        xs, ys = zip(*vertices)
        raw_x1, raw_x2 = min(xs) - pad, max(xs) - pad
        raw_y1, raw_y2 = min(ys) - pad, max(ys) - pad
        raw_area = max((raw_x2 - raw_x1) * (raw_y2 - raw_y1), 1e-6)

        x1 = max(0.0, min(float(out_w), raw_x1))
        y1 = max(0.0, min(float(out_h), raw_y1))
        x2 = max(0.0, min(float(out_w), raw_x2))
        y2 = max(0.0, min(float(out_h), raw_y2))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1.5 or bh <= 1.5:
            continue
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
    """Locate ALL FOUR required subdirs (rgb_img, ir_img, rgb_lbl, ir_lbl); fail fast if any is missing."""
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

    if not all([rgb_img, ir_img, rgb_lbl, ir_lbl]):
        found = sorted(str(p.relative_to(extract_dir)) for p in all_dirs)
        raise FileNotFoundError(
            f"DroneVehicle '{split}' requires ALL 4 subdirs (rgb_img, ir_img, rgb_lbl, ir_lbl); "
            f"got rgb_img={rgb_img}, ir_img={ir_img}, rgb_lbl={rgb_lbl}, ir_lbl={ir_lbl}. Found: {found[:30]}"
        )
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
) -> Dict[str, object]:
    ir_im = cv2.imread(str(ir_src), cv2.IMREAD_COLOR)
    rgb_im = cv2.imread(str(rgb_src), cv2.IMREAD_COLOR)
    if ir_im is None:
        raise RuntimeError(f"Failed to read IR image: {ir_src}")
    if rgb_im is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_src}")

    ir_crop, pad_ir = _crop_border_if_needed(ir_im, border=100)
    rgb_crop, pad_rgb = _crop_border_if_needed(rgb_im, border=100)

    h_ir, w_ir = ir_crop.shape[:2]
    h_rgb, w_rgb = rgb_crop.shape[:2]
    if (h_rgb, w_rgb) != (h_ir, w_ir):
        # Preserve original RGB dimensions for accurate normalized coordinate scaling if resized,
        # but fail if aspect ratio diverges significantly (indicating mismatched/corrupt crop).
        ar_ir = w_ir / float(max(h_ir, 1))
        ar_rgb = w_rgb / float(max(h_rgb, 1))
        if abs(ar_ir - ar_rgb) > 0.05:
            raise ValueError(
                f"RGB/IR aspect mismatch on '{stem}': IR={w_ir}x{h_ir} vs RGB={w_rgb}x{h_rgb}"
            )
        rgb_crop = cv2.resize(rgb_crop, (w_ir, h_ir), interpolation=cv2.INTER_LINEAR)

    # Compute exact RGB illumination stats BEFORE JPEG re-save (normalized [0, 1])
    rgb_f = rgb_crop.astype(np.float32) * (1.0 / 255.0)
    lum_mean = float(rgb_f.mean())
    lum_std = float(rgb_f.std())

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    if not cv2.imwrite(str(out_ir_img), ir_crop, encode_params):
        raise OSError(f"cv2.imwrite failed for IR image: {out_ir_img}")
    if not cv2.imwrite(str(out_rgb_img), rgb_crop, encode_params):
        raise OSError(f"cv2.imwrite failed for RGB image: {out_rgb_img}")

    skipped: Counter = Counter()
    unk_ir: Counter = Counter()
    cls_ir: Counter = Counter()
    unk_rgb: Counter = Counter()
    cls_rgb: Counter = Counter()

    ir_boxes = _parse_dv_xml_to_hbb(
        ir_xml, pad=pad_ir, out_w=w_ir, out_h=h_ir, unknown_counter=unk_ir, class_counter=cls_ir,
        skipped_counter=skipped,
    )
    # Use pre-resize `(w_rgb, h_rgb)` so normalized RGB labels stay geometrically exact!
    rgb_boxes = _parse_dv_xml_to_hbb(
        rgb_xml, pad=pad_rgb, out_w=w_rgb, out_h=h_rgb, unknown_counter=unk_rgb, class_counter=cls_rgb,
        skipped_counter=skipped,
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
        "point_artifacts_skipped": skipped["point"],
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
        if not cv2.imwrite(str(vis_dir / f"dv_{tag}_{stem}.jpg"), pair):
            raise OSError(f"Failed writing sanity visual: {vis_dir / f'dv_{tag}_{stem}.jpg'}")
    print(f"[OK] Saved {len(chosen)} RGB|IR DroneVehicle sanity visuals -> {vis_dir}")


def _rewrite_dv_yamls(out_root: Path) -> None:
    ir_root = (out_root / "dronevehicle_ir").resolve()
    rgb_root = (out_root / "dronevehicle_rgb").resolve()
    names_dict = {i: n for i, n in enumerate(CLASS_NAMES)}

    def _dump(path: Path, payload: Dict[str, object]) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    _dump(
        out_root / "dronevehicle_ir.yaml",
        {
            "path": str(ir_root),
            "rgb_root": str(rgb_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(CLASS_NAMES),
            "names": names_dict,
        },
    )
    _dump(
        out_root / "dronevehicle_rgb.yaml",
        {
            "path": str(rgb_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(CLASS_NAMES),
            "names": names_dict,
        },
    )
    for tag in ("day", "night"):
        _dump(
            out_root / f"dronevehicle_ir_{tag}.yaml",
            {
                "path": str(ir_root),
                "train": "images/train",
                "val": f"val_{tag}.txt",
                "test": f"test_{tag}.txt",
                "nc": len(CLASS_NAMES),
                "names": names_dict,
            },
        )
        _dump(
            out_root / f"dronevehicle_rgb_{tag}.yaml",
            {
                "path": str(rgb_root),
                "train": "images/train",
                "val": f"val_{tag}.txt",
                "test": f"test_{tag}.txt",
                "nc": len(CLASS_NAMES),
                "names": names_dict,
            },
        )


def verify_prepared_dronevehicle(out_root: Path) -> Tuple[bool, str]:
    """Verify full dual-modality manifest, YAML paths, exact stem sets, and Day/Night partitions."""
    manifest_p = out_root / "dronevehicle_ir" / "dv_manifest.json"
    if not manifest_p.is_file():
        return False, f"Missing manifest {manifest_p}"
    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Unreadable manifest: {exc}"

    req_yamls = [
        "dronevehicle_ir.yaml",
        "dronevehicle_rgb.yaml",
        "dronevehicle_ir_day.yaml",
        "dronevehicle_ir_night.yaml",
        "dronevehicle_rgb_day.yaml",
        "dronevehicle_rgb_night.yaml",
    ]
    for y in req_yamls:
        yp = out_root / y
        if not yp.is_file():
            return False, f"Missing YAML {y}"
        try:
            ydata = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
            if not Path(str(ydata.get("path", ""))).is_dir():
                return False, f"YAML {y} has non-existent path={ydata.get('path')}"
            if y == "dronevehicle_ir.yaml" and not Path(str(ydata.get("rgb_root", ""))).is_dir():
                return False, f"YAML {y} has non-existent rgb_root={ydata.get('rgb_root')}"
        except Exception as exc:
            return False, f"Unreadable YAML {y}: {exc}"

    splits_meta = manifest.get("splits", {})
    for sp in ("train", "val", "test"):
        sp_meta = splits_meta.get(sp, {}) or {}
        exp_n = int(sp_meta.get("pairs", 0))
        if exp_n <= 0:
            return False, f"Manifest has invalid count for split '{sp}': {exp_n}"
        ir_img_stems = {p.stem for p in (out_root / "dronevehicle_ir" / "images" / sp).glob("*.jpg")}
        ir_lbl_stems = {p.stem for p in (out_root / "dronevehicle_ir" / "labels" / sp).glob("*.txt")}
        rgb_img_stems = {p.stem for p in (out_root / "dronevehicle_rgb" / "images" / sp).glob("*.jpg")}
        rgb_lbl_stems = {p.stem for p in (out_root / "dronevehicle_rgb" / "labels" / sp).glob("*.txt")}
        if len(ir_img_stems) != exp_n or not (ir_img_stems == ir_lbl_stems == rgb_img_stems == rgb_lbl_stems):
            return (
                False,
                f"Stem/count mismatch in '{sp}': expected={exp_n}, ir_img={len(ir_img_stems)}, "
                f"ir_lbl={len(ir_lbl_stems)}, rgb_img={len(rgb_img_stems)}, rgb_lbl={len(rgb_lbl_stems)}",
            )
        for mod in ("dronevehicle_ir", "dronevehicle_rgb"):
            day_p = out_root / mod / f"{sp}_day.txt"
            ngt_p = out_root / mod / f"{sp}_night.txt"
            if not day_p.is_file() or not ngt_p.is_file():
                return False, f"Missing Day/Night split list under {mod}/{sp}"
            day_stems = {Path(line.strip()).stem for line in day_p.read_text(encoding="utf-8").splitlines() if line.strip()}
            ngt_stems = {Path(line.strip()).stem for line in ngt_p.read_text(encoding="utf-8").splitlines() if line.strip()}
            if not day_stems or not ngt_stems:
                return False, f"Empty Day ({len(day_stems)}) or Night ({len(ngt_stems)}) list in {mod}/{sp}"
            if day_stems & ngt_stems:
                return False, f"Overlapping Day/Night stems in {mod}/{sp}: {len(day_stems & ngt_stems)}"
            if (day_stems | ngt_stems) != ir_img_stems:
                return (
                    False,
                    f"Day∪Night stems ({len(day_stems | ngt_stems)}) != image stems ({len(ir_img_stems)}) in {mod}/{sp}",
                )
    return True, f"Verified complete DroneVehicle manifest ({ {k: v['pairs'] for k, v in splits_meta.items()} })"


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
            raise RuntimeError(f"Empty records for split '{split}'")
        day_ir: List[str] = []
        night_ir: List[str] = []
        day_rgb: List[str] = []
        night_rgb: List[str] = []

        lum_vals = [float(r["lum_mean"]) for r in recs]
        for r in recs:
            stem = str(r["stem"])
            rel = f"./images/{split}/{stem}.jpg"
            if float(r["lum_mean"]) >= day_lum_thresh:
                day_ir.append(rel)
                day_rgb.append(rel)
            else:
                night_ir.append(rel)
                night_rgb.append(rel)

        if not day_ir or not night_ir:
            raise RuntimeError(
                f"Split '{split}' produced empty Day ({len(day_ir)}) or Night ({len(night_ir)}) subset "
                f"at day_lum_thresh={day_lum_thresh}!"
            )

        (ir_root / f"{split}_day.txt").write_text("\n".join(day_ir) + "\n", encoding="utf-8")
        (ir_root / f"{split}_night.txt").write_text("\n".join(night_ir) + "\n", encoding="utf-8")
        (rgb_root / f"{split}_day.txt").write_text("\n".join(day_rgb) + "\n", encoding="utf-8")
        (rgb_root / f"{split}_night.txt").write_text("\n".join(night_rgb) + "\n", encoding="utf-8")

        cls_ir_tot: Counter = Counter()
        cls_rgb_tot: Counter = Counter()
        unk_tot: Counter = Counter()
        for r in recs:
            cls_ir_tot.update(r["cls_ir"])    # type: ignore[arg-type]
            cls_rgb_tot.update(r["cls_rgb"])  # type: ignore[arg-type]
            unk_tot.update(r["unk"])          # type: ignore[arg-type]

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
            "point_artifacts_skipped": sum(int(r["point_artifacts_skipped"]) for r in recs),
            "unknown_skipped": dict(unk_tot),
        }

    _rewrite_dv_yamls(out_root)
    stats_path = ir_root / "illumination_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (ir_root / "dv_manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote DroneVehicle YAMLs + Day/Night splits + manifest -> {stats_path}")
    return stats


def _bundle_to_drive(out_root: Path, bundle_path: Path) -> None:
    """Pack verified dataset directly into `bundle_path.parent` (no extra local SSD copy) + atomic replace."""
    ok, msg = verify_prepared_dronevehicle(out_root)
    if not ok:
        raise RuntimeError(f"Refusing to bundle incomplete DroneVehicle dataset: {msg}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = bundle_path.parent / f".{bundle_path.name}.tmp"
    print(f"[BUNDLE] Packing verified DroneVehicle -> {bundle_path} (ZIP_STORED) ...")
    t0 = time.perf_counter()
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for sub in ("dronevehicle_ir", "dronevehicle_rgb"):
            root_sub = out_root / sub
            for p in root_sub.rglob("*"):
                if p.is_file() and not p.name.endswith(".cache"):
                    zf.write(p, arcname=str(p.relative_to(out_root)))
        for y in out_root.glob("dronevehicle_*.yaml"):
            zf.write(y, arcname=y.name)
    if not _is_valid_zip(tmp_zip):
        raise RuntimeError(f"Packed bundle failed ZIP validation: {tmp_zip}")
    os.replace(tmp_zip, bundle_path)
    print(
        f"[OK] Saved verified DroneVehicle bundle ({bundle_path.stat().st_size / 1e9:.2f} GB) "
        f"in {time.perf_counter() - t0:.1f}s -> {bundle_path}"
    )


def _restore_from_bundle(out_root: Path, bundle_path: Path) -> bool:
    # 1. Fast path: verify full manifest locally, rewrite YAML paths, and repair missing Drive bundle if needed!
    ok_local, msg_local = verify_prepared_dronevehicle(out_root)
    if ok_local:
        _rewrite_dv_yamls(out_root)
        if not _is_valid_zip(bundle_path):
            print(f"[REPAIR] Local DroneVehicle verified, rebuilding missing/invalid Drive bundle -> {bundle_path}")
            _bundle_to_drive(out_root, bundle_path)
        print(f"[FAST] {msg_local}")
        return True

    if not bundle_path.is_file():
        return False

    print(f"[RESTORE] Unpacking cached DroneVehicle bundle from Drive: {bundle_path} ...")
    t0 = time.perf_counter()
    stage_restore = out_root / "_dv_restore_tmp"
    if stage_restore.exists():
        shutil.rmtree(stage_restore)
    stage_restore.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            need_bytes = sum(i.file_size for i in zf.infolist())
            free_bytes = shutil.disk_usage(out_root).free
            if free_bytes < need_bytes + 512 * 1024 * 1024:
                raise OSError(
                    f"Insufficient SSD space to unpack DroneVehicle bundle: "
                    f"need {need_bytes/1e9:.2f} GB, free {free_bytes/1e9:.2f} GB"
                )
            zf.extractall(stage_restore)
        _rewrite_dv_yamls(stage_restore)
        ok_after, msg_after = verify_prepared_dronevehicle(stage_restore)
        if not ok_after:
            raise RuntimeError(f"Restored Drive bundle failed manifest verification: {msg_after}")
    except Exception as exc:
        # Quarantine corrupt/legacy bundle so we fall back cleanly to fresh preparation instead of looping forever!
        shutil.rmtree(stage_restore, ignore_errors=True)
        quarantine = bundle_path.with_name(f"{bundle_path.name}.quarantine_{int(time.time())}")
        print(f"[WARN] Cached Drive bundle {bundle_path} is invalid/legacy ({exc}); quarantining -> {quarantine.name}")
        try:
            bundle_path.rename(quarantine)
        except Exception:
            pass
        return False

    for sub in ("dronevehicle_ir", "dronevehicle_rgb"):
        dst_sub = out_root / sub
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.move(str(stage_restore / sub), str(dst_sub))
    shutil.rmtree(stage_restore, ignore_errors=True)
    _rewrite_dv_yamls(out_root)
    print(f"[OK] Restored & verified DroneVehicle in {time.perf_counter() - t0:.1f}s -> {out_root}")
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
    elif drive_bundle is None:
        ok_local, msg_local = verify_prepared_dronevehicle(out_root)
        if ok_local:
            _rewrite_dv_yamls(out_root)
            print(f"[FAST] {msg_local}")
            stats_p = out_root / "dronevehicle_ir" / "illumination_stats.json"
            return json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.is_file() else {}

    work_raw = raw_dir if raw_dir is not None else (out_root / "_dv_raw_tmp")
    work_raw.mkdir(parents=True, exist_ok=True)

    # Build into isolated staging dirs; publish atomically only after ALL splits pass 100%!
    stage_root = out_root / "_dv_stage_tmp"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    ir_root = stage_root / "dronevehicle_ir"
    rgb_root = stage_root / "dronevehicle_rgb"
    split_records: Dict[str, List[Dict[str, object]]] = {}

    for split in splits:
        t0 = time.perf_counter()
        print("\n" + "=" * 72)
        print(f"[DRONEVEHICLE] Preparing split: {split}")
        print("=" * 72)

        extract_dir = work_raw / f"extracted_{split}"
        if raw_dir is None:
            # Always extract freshly into clean dir so partial crashed extractions are never trusted
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            zip_path = work_raw / f"{split}.zip"
            _download_hf_file(f"{split}.zip", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                need_bytes = sum(i.file_size for i in zf.infolist())
                free_bytes = shutil.disk_usage(work_raw).free
                if free_bytes < need_bytes + 1024 * 1024 * 1024:
                    raise OSError(
                        f"Insufficient SSD space to extract {zip_path.name}: "
                        f"need {need_bytes/1e9:.2f} GB, free {free_bytes/1e9:.2f} GB"
                    )
                print(f"[UNZIP] Extracting {zip_path.name} ({need_bytes/1e9:.2f} GB) -> {extract_dir} ...")
                extract_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_dir)
            if not keep_raw:
                zip_path.unlink(missing_ok=True)
        else:
            extract_dir = raw_dir

        rgb_img_dir, ir_img_dir, rgb_lbl_dir, ir_lbl_dir = _find_split_subdirs(extract_dir, split)
        ir_files = {p.stem: p for p in ir_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        rgb_files = {p.stem: p for p in rgb_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        if not ir_files or set(ir_files.keys()) != set(rgb_files.keys()):
            only_ir = len(set(ir_files.keys()) - set(rgb_files.keys()))
            only_rgb = len(set(rgb_files.keys()) - set(ir_files.keys()))
            raise RuntimeError(
                f"DroneVehicle '{split}' IR/RGB stem mismatch! "
                f"IR={len(ir_files)}, RGB={len(rgb_files)}, only_ir={only_ir}, only_rgb={only_rgb}"
            )
        common_stems = sorted(ir_files.keys())
        print(f"[INFO] {split}: verified strict 1:1 pairing across {len(common_stems)} RGB-IR frames.")

        out_ir_img_dir = ir_root / "images" / split
        out_ir_lbl_dir = ir_root / "labels" / split
        out_rgb_img_dir = rgb_root / "images" / split
        out_rgb_lbl_dir = rgb_root / "labels" / split
        for d in (out_ir_img_dir, out_ir_lbl_dir, out_rgb_img_dir, out_rgb_lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        def _worker(st: str) -> Dict[str, object]:
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
                recs.append(res)
                if idx % 2000 == 0 or idx == len(common_stems):
                    print(f"  [{split}] processed {idx}/{len(common_stems)} pairs ({time.perf_counter() - t0:.1f}s)")

        split_records[split] = recs

        if not keep_raw and raw_dir is None:
            shutil.rmtree(extract_dir, ignore_errors=True)

    if not keep_raw and raw_dir is None:
        shutil.rmtree(work_raw, ignore_errors=True)

    stats = _write_yamls_and_split_lists(stage_root, split_records, day_lum_thresh=day_lum_thresh)
    ok_stage, msg_stage = verify_prepared_dronevehicle(stage_root)
    if not ok_stage:
        raise RuntimeError(f"Staged DroneVehicle failed self-verification: {msg_stage}")

    # Atomically publish staged directories into `out_root` and rewrite YAMLs to final paths
    for sub in ("dronevehicle_ir", "dronevehicle_rgb"):
        dst_sub = out_root / sub
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.move(str(stage_root / sub), str(dst_sub))
    shutil.rmtree(stage_root, ignore_errors=True)
    _rewrite_dv_yamls(out_root)

    if vis_dir is not None and split_records.get("val"):
        _save_sanity_visuals(
            out_root / "dronevehicle_ir",
            out_root / "dronevehicle_rgb",
            split_records["val"],
            vis_dir,
            day_lum_thresh,
        )

    if drive_bundle is not None:
        _bundle_to_drive(out_root, drive_bundle)

    print("\n[SUMMARY] DroneVehicle preparation complete & verified:")
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
