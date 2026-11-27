"""Synchronized Paired `(IR, RGB)` Dataset & Transforms for Track B Privileged Cross-Modal KD.

Guarantees pixel-exact geometric alignment between Infrared (student + primary teacher input),
Privileged RGB (frozen auxiliary teacher input during training ONLY), and `rgb_valid_mask`
(`255` on authentic RGB scene pixels, `0` on artificial 114-gray LetterBox/Mosaic/affine borders):
- `PairedMosaic`, `PairedLetterBox`, `PairedRandomPerspective`, and `PairedRandomFlip` draw
  a SINGLE set of random coordinates / homography matrix `M` / flip decision and apply it
  identically to `img` (IR), `rgb_img` (RGB), and `rgb_valid_mask` (1-ch uint8 validity mask).
- Carrying `rgb_valid_mask` guarantees that artificial `114`-gray borders NEVER leak variance
  into the Spatially-Aware Night-Blackout Circuit Breaker (0.00% false trigger on pitch-black
  night frames padded with 114 gray!).
- Photometric augmentation applies ONLY IR intensity jitter (`hsv_v`, `erasing=0.0` matching
  Ultralytics native detection augmentation so Track A vs Track B has zero confounded cutout!).
- Validation & Test NEVER use this dataset — they run 100% native IR-only `YOLODataset`,
  guaranteeing zero privileged leakage and +0.00 ms inference overhead.
"""

from __future__ import annotations

import inspect
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset


def _resize_to_match(im: np.ndarray, target_hw: Tuple[int, int], *, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    if im.shape[0] == th and im.shape[1] == tw:
        return im
    return cv2.resize(im, (tw, th), interpolation=interp)


def _compute_letterbox_params(
    orig_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    *,
    scaleup: bool,
) -> Tuple[int, int, int, int, int, int]:
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    th, tw = int(target_hw[0]), int(target_hw[1])
    r = min(th / float(oh), tw / float(ow))
    if not scaleup:
        r = min(r, 1.0)
    unpad_w = int(round(ow * r))
    unpad_h = int(round(oh * r))
    dw = (tw - unpad_w) * 0.5
    dh = (th - unpad_h) * 0.5
    top = int(round(dh - 0.1))
    bottom = max(0, th - unpad_h - top)
    left = int(round(dw - 0.1))
    right = max(0, tw - unpad_w - left)
    return unpad_w, unpad_h, top, bottom, left, right


class PairedLetterBox:
    """Apply identical LetterBox scaling & padding to `img` (IR), `rgb_img` (RGB), and `rgb_valid_mask`."""

    def __init__(self, new_shape: int | Tuple[int, int] = (640, 640), scaleup: bool = True) -> None:
        self.new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else (int(new_shape[0]), int(new_shape[1]))
        self.scaleup = bool(scaleup)
        self.lb = LetterBox(new_shape=self.new_shape, auto=False, scaleup=self.scaleup)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        rgb = labels.pop("rgb_img", None)
        vmask = labels.pop("rgb_valid_mask", None)
        orig_h, orig_w = labels["img"].shape[:2]
        out = self.lb(labels)
        new_h, new_w = out["img"].shape[:2]

        # Derive exact LetterBox parameters matching `out["img"]`'s actual target shape (handles rect_shape too)
        unpad_w, unpad_h, top, bottom, left, right = _compute_letterbox_params(
            (orig_h, orig_w), (new_h, new_w), scaleup=self.scaleup
        )
        if rgb is not None:
            rgb = _resize_to_match(rgb, (orig_h, orig_w))
            if (orig_w, orig_h) != (unpad_w, unpad_h):
                rgb = cv2.resize(rgb, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            out["rgb_img"] = _resize_to_match(rgb, (new_h, new_w))
        if vmask is None:
            vmask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
        else:
            vmask = _resize_to_match(vmask, (orig_h, orig_w), interp=cv2.INTER_NEAREST)
        if (orig_w, orig_h) != (unpad_w, unpad_h):
            vmask = cv2.resize(vmask, (unpad_w, unpad_h), interpolation=cv2.INTER_NEAREST)
        vmask = cv2.copyMakeBorder(vmask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        out["rgb_valid_mask"] = _resize_to_match(vmask, (new_h, new_w), interp=cv2.INTER_NEAREST)
        return out


class PairedMosaic:
    """4-image Mosaic applying identical quadrant crops & placements to IR, RGB, and `rgb_valid_mask`."""

    def __init__(self, dataset: "PairedRGBIRDataset", imgsz: int = 640, p: float = 1.0) -> None:
        self.dataset = dataset
        self.imgsz = int(imgsz)
        self.border = (-self.imgsz // 2, -self.imgsz // 2)
        self.p = float(p)
        self.buffer_len = len(dataset)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        if self.p <= 0.0 or random.random() >= self.p or self.buffer_len < 4:
            return labels

        s = self.imgsz
        yc = int(random.uniform(-self.border[0], 2 * s + self.border[0]))
        xc = int(random.uniform(-self.border[1], 2 * s + self.border[1]))

        extra_idxs = [random.randint(0, self.buffer_len - 1) for _ in range(3)]
        samples = [labels] + [self.dataset.get_image_and_label(idx) for idx in extra_idxs]

        img4_ir = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        img4_rgb = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mask4 = np.zeros((s * 2, s * 2), dtype=np.uint8)  # 0 on empty 114-gray canvas!

        cls4: List[np.ndarray] = []
        inst4 = []

        for i, sample in enumerate(samples):
            im_ir = sample["img"]
            h, w = im_ir.shape[:2]
            im_rgb = _resize_to_match(sample["rgb_img"], (h, w))
            vm = sample.get("rgb_valid_mask")
            if vm is None:
                vm = np.full((h, w), 255, dtype=np.uint8)
            else:
                vm = _resize_to_match(vm, (h, w), interp=cv2.INTER_NEAREST)

            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4_ir[y1a:y2a, x1a:x2a] = im_ir[y1b:y2b, x1b:x2b]
            img4_rgb[y1a:y2a, x1a:x2a] = im_rgb[y1b:y2b, x1b:x2b]
            mask4[y1a:y2a, x1a:x2a] = vm[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            inst = sample["instances"]
            inst.convert_bbox(format="xyxy")
            if inst.normalized:
                inst.denormalize(w, h)
            inst.add_padding(padw, padh)
            inst4.append(inst)
            cls4.append(sample["cls"])

        from ultralytics.utils.instance import Instances

        merged_inst = Instances.concatenate(inst4, axis=0)
        merged_inst.clip(2 * s, 2 * s)
        labels["img"] = img4_ir
        labels["rgb_img"] = img4_rgb
        labels["rgb_valid_mask"] = mask4
        labels["cls"] = np.concatenate(cls4, axis=0) if cls4 else np.zeros((0, 1), dtype=np.float32)
        labels["instances"] = merged_inst
        labels["resized_shape"] = (2 * s, 2 * s)
        labels["mosaic_border"] = self.border
        return labels


class PairedRandomPerspective:
    """Apply identical affine transform `M` to IR image, RGB image, `rgb_valid_mask`, and boxes."""

    def __init__(
        self,
        *,
        imgsz: int = 640,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
    ) -> None:
        self.imgsz = int(imgsz)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        img = labels["img"]
        h0, w0 = img.shape[:2]
        rgb = _resize_to_match(labels["rgb_img"], (h0, w0))
        vm = labels.get("rgb_valid_mask")
        if vm is None:
            vm = np.full((h0, w0), 255, dtype=np.uint8)
        else:
            vm = _resize_to_match(vm, (h0, w0), interp=cv2.INTER_NEAREST)
        border = labels.pop("mosaic_border", (0, 0))

        out_w = w0 + border[1] * 2 if border != (0, 0) else self.imgsz
        out_h = h0 + border[0] * 2 if border != (0, 0) else self.imgsz

        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w0 / 2.0
        C[1, 2] = -h0 / 2.0

        R = np.eye(3, dtype=np.float32)
        ang = random.uniform(-self.degrees, self.degrees)
        sc = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=ang, center=(0, 0), scale=sc)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180.0)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180.0)

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * out_w
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * out_h

        M = (T @ S @ R @ C)[:2]  # (2, 3)

        img_w = cv2.warpAffine(img, M, dsize=(out_w, out_h), borderValue=(114, 114, 114))
        rgb_w = cv2.warpAffine(rgb, M, dsize=(out_w, out_h), borderValue=(114, 114, 114))
        vm_w = cv2.warpAffine(vm, M, dsize=(out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)

        inst = labels["instances"]
        inst.convert_bbox(format="xyxy")
        if inst.normalized:
            inst.denormalize(w0, h0)

        bboxes = inst.bboxes
        n = len(bboxes)
        if n > 0:
            xy = np.ones((n * 4, 3), dtype=np.float32)
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy_t = xy @ M.T
            xy_t = xy_t.reshape(n, 8)
            xs = xy_t[:, [0, 2, 4, 6]]
            ys = xy_t[:, [1, 3, 5, 7]]
            new_b = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)
            new_b[:, [0, 2]] = new_b[:, [0, 2]].clip(0.0, float(out_w))
            new_b[:, [1, 3]] = new_b[:, [1, 3]].clip(0.0, float(out_h))

            orig_w = (bboxes[:, 2] - bboxes[:, 0]) * sc
            orig_h = (bboxes[:, 3] - bboxes[:, 1]) * sc
            nw = new_b[:, 2] - new_b[:, 0]
            nh = new_b[:, 3] - new_b[:, 1]
            ar = np.maximum(nw / (nh + 1e-6), nh / (nw + 1e-6))
            keep = (nw > 2.0) & (nh > 2.0) & ((nw * nh) / (orig_w * orig_h + 1e-6) > 0.1) & (ar < 100.0)
            inst.update(bboxes=new_b)
            inst = inst[keep]
            labels["cls"] = labels["cls"][keep]

        labels["img"] = img_w
        labels["rgb_img"] = rgb_w
        labels["rgb_valid_mask"] = vm_w
        labels["instances"] = inst
        labels["resized_shape"] = (out_h, out_w)
        return labels


class PairedRandomFlip:
    """Horizontal flip synchronized across IR image, RGB image, `rgb_valid_mask`, and boxes."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            img = labels["img"]
            h, w = img.shape[:2]
            labels["img"] = np.ascontiguousarray(np.fliplr(img))
            if "rgb_img" in labels and labels["rgb_img"] is not None:
                labels["rgb_img"] = np.ascontiguousarray(np.fliplr(labels["rgb_img"]))
            if "rgb_valid_mask" in labels and labels["rgb_valid_mask"] is not None:
                labels["rgb_valid_mask"] = np.ascontiguousarray(np.fliplr(labels["rgb_valid_mask"]))
            inst = labels["instances"]
            inst.convert_bbox(format="xyxy")
            if inst.normalized:
                inst.denormalize(w, h)
            inst.fliplr(w)
            labels["instances"] = inst
        return labels


class PairedIRPhotometric:
    """IR-only thermal gain jitter (`erasing=0.0` matching Ultralytics native detection aug!).

    Leaves privileged `rgb_img` photometrically clean so the frozen RGB teacher and our
    Spatially-Aware Illumination Circuit Breaker observe true scene luminance.
    """

    def __init__(self, hsv_v: float = 0.4, erasing: float = 0.0) -> None:
        self.hsv_v = float(hsv_v)
        self.erasing = float(erasing)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        img = labels["img"]
        if self.hsv_v > 0.0:
            gain = random.uniform(1.0 - self.hsv_v, 1.0 + self.hsv_v)
            lut = np.clip(np.arange(256, dtype=np.float32) * gain, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, lut)
        if self.erasing > 0.0 and random.random() < self.erasing:
            h, w = img.shape[:2]
            area = h * w
            target_area = random.uniform(0.02, 0.15) * area
            ar = random.uniform(0.3, 3.3)
            eh = int(round(math.sqrt(target_area * ar)))
            ew = int(round(math.sqrt(target_area / ar)))
            if 0 < eh < h and 0 < ew < w:
                y0 = random.randint(0, h - eh)
                x0 = random.randint(0, w - ew)
                img = img.copy()
                img[y0 : y0 + eh, x0 : x0 + ew] = 114
        labels["img"] = img
        return labels


class PairedFormat:
    """Wrap Ultralytics `Format` to emit synchronized `img`, `rgb_img`, and `rgb_valid_mask`."""

    def __init__(self, **fmt_kwargs: Any) -> None:
        self.fmt = Format(**fmt_kwargs)

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        rgb = labels.pop("rgb_img")
        vm = labels.pop("rgb_valid_mask", None)
        h, w = labels["img"].shape[:2]
        inst = labels.get("instances")
        if inst is not None and getattr(inst, "normalized", False):
            inst.denormalize(w, h)
        out = self.fmt(labels)
        th, tw = int(out["img"].shape[1]), int(out["img"].shape[2])
        rgb = _resize_to_match(rgb, (th, tw))
        rgb_chw = np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1))
        out["rgb_img"] = torch.from_numpy(rgb_chw)

        if vm is None:
            vm = np.full((th, tw), 255, dtype=np.uint8)
        else:
            vm = _resize_to_match(vm, (th, tw), interp=cv2.INTER_NEAREST)
        out["rgb_valid_mask"] = torch.from_numpy(np.ascontiguousarray((vm > 127).astype(np.uint8)[None, :, :]))
        return out


class PairedRGBIRDataset(YOLODataset):
    """YOLODataset subclass delivering synchronized `(img=IR, rgb_img=RGB, rgb_valid_mask)` batches."""

    def __init__(self, *args: Any, rgb_root: str | Path, **kwargs: Any) -> None:
        self.rgb_root = Path(rgb_root)
        if not self.rgb_root.is_dir():
            raise FileNotFoundError(f"Paired RGB root not found: {self.rgb_root}")
        from ultralytics.data.base import BaseDataset

        allowed = set(inspect.signature(BaseDataset.__init__).parameters.keys()) | {"data", "task"}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        super().__init__(*args, **filtered)

    def _rgb_path_for(self, ir_im_file: str) -> Path:
        p = Path(ir_im_file)
        parts = list(p.parts)
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx] == "images":
                cand = self.rgb_root / Path(*parts[idx:])
                if cand.is_file():
                    return cand
                break
        cand2 = self.rgb_root / "images" / p.parent.name / p.name
        if cand2.is_file():
            return cand2
        raise FileNotFoundError(f"Missing paired RGB image for IR frame '{ir_im_file}' (checked {cand2})")

    def get_image_and_label(self, index: int, full: bool = False) -> Dict[str, Any]:
        label = super().get_image_and_label(index, full=full)
        rgb_path = self._rgb_path_for(label["im_file"])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read paired RGB image: {rgb_path}")
        h, w = label["img"].shape[:2]
        label["rgb_img"] = _resize_to_match(rgb, (h, w))
        label["rgb_valid_mask"] = np.full((h, w), 255, dtype=np.uint8)
        return label

    def build_transforms(self, hyp: Any) -> Compose:
        if self.augment:
            mosaic_p = float(getattr(hyp, "mosaic", 1.0)) if not self.rect else 0.0
            tfs: List[Any] = []
            if mosaic_p > 0.0:
                tfs.append(PairedMosaic(self, imgsz=self.imgsz, p=mosaic_p))
                tfs.append(
                    PairedRandomPerspective(
                        imgsz=self.imgsz,
                        degrees=float(getattr(hyp, "degrees", 0.0)),
                        translate=float(getattr(hyp, "translate", 0.1)),
                        scale=float(getattr(hyp, "scale", 0.5)),
                        shear=float(getattr(hyp, "shear", 0.0)),
                    )
                )
            else:
                tfs.append(PairedLetterBox(new_shape=self.imgsz, scaleup=True))
                tfs.append(
                    PairedRandomPerspective(
                        imgsz=self.imgsz,
                        degrees=float(getattr(hyp, "degrees", 0.0)),
                        translate=float(getattr(hyp, "translate", 0.1)),
                        scale=float(getattr(hyp, "scale", 0.25)),
                        shear=float(getattr(hyp, "shear", 0.0)),
                    )
                )
            tfs.append(PairedRandomFlip(p=float(getattr(hyp, "fliplr", 0.5))))
            # Keep `erasing=0.0` so Track B matches Ultralytics native detection aug (zero confounded cutout!)
            tfs.append(
                PairedIRPhotometric(
                    hsv_v=float(getattr(hyp, "hsv_v", 0.4)),
                    erasing=0.0,
                )
            )
            transforms = Compose(tfs)
        else:
            transforms = Compose([PairedLetterBox(new_shape=self.imgsz, scaleup=False)])

        transforms.append(
            PairedFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                bgr=0.0,
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        rgb_list = [b.pop("rgb_img") for b in batch]
        vm_list = [b.pop("rgb_valid_mask") for b in batch]
        out = YOLODataset.collate_fn(batch)
        out["rgb_img"] = torch.stack(rgb_list, 0)
        out["rgb_valid_mask"] = torch.stack(vm_list, 0)
        return out


def attach_paired_rgbt_dataloader_hook(yolo_model: Any, rgb_root: str | Path) -> None:
    """Hook `trainer.build_dataset` so ONLY `mode='train'` uses `PairedRGBIRDataset` (val stays pure IR)."""
    rgb_root_p = Path(rgb_root)

    def _on_pretrain_routine_start(trainer: Any) -> None:
        orig_build_dataset = trainer.build_dataset

        def _paired_build_dataset(img_path: str, mode: str = "train", batch: Optional[int] = None) -> Any:
            if mode == "train":
                m = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                gs = max(int(m.stride.max() if hasattr(m, "stride") else 32), 32)
                ds = PairedRGBIRDataset(
                    img_path=img_path,
                    imgsz=trainer.args.imgsz,
                    batch_size=batch,
                    augment=True,
                    hyp=trainer.args,
                    rect=False,
                    cache=False,
                    single_cls=getattr(trainer.args, "single_cls", False),
                    stride=gs,
                    pad=0.0,
                    prefix="train-rgbt: ",
                    task=trainer.args.task,
                    classes=getattr(trainer.args, "classes", None),
                    data=trainer.data,
                    fraction=getattr(trainer.args, "fraction", 1.0),
                    rgb_root=rgb_root_p,
                )
                print(
                    f"[TRACK B DATA] Attached synchronized PairedRGBIRDataset "
                    f"(len={len(ds)}, rgb_root={rgb_root_p}; val/test stay 100% pure IR!)"
                )
                return ds
            return orig_build_dataset(img_path, mode=mode, batch=batch)

        trainer.build_dataset = _paired_build_dataset

    yolo_model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
