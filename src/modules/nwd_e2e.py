"""Mod C1 & C2: Scale-Gated NWD for YOLO26 End-to-End Assignment (C1) & DFL-Free Regression (C2).

Verified directly against installed `ultralytics 8.4.218` (`phase0_yolo26_source_audit.md`):
- `E2ELoss` creates:
    - `self.one2many = v8DetectionLoss(model, tal_topk=10)`
    - `self.one2one  = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)`
- Coordinate convention audit (CRITICAL):
    - Inside `TaskAlignedAssigner.iou_calculation(gt_bboxes, pd_bboxes)`:
      inputs are ALREADY in **pixel xyxy** coordinates (`v8DetectionLoss` multiplies by `stride_tensor`).
    - Inside `BboxLoss.forward(..., pred_bboxes, ..., target_bboxes, ..., stride)`:
      `pred_bboxes` and `target_bboxes` are in **stride-normalized grid** units, and `stride` has
      shape `(num_anchors, 1)`. We MUST multiply by `stride[fg_mask[1]]` to recover exact pixel
      coordinates before evaluating pixel area `A_gt` and NWD constant `C`!
- Training-only supervision patch: ZERO extra parameters, ZERO FLOPs, ZERO inference latency.
"""

from __future__ import annotations

import types
from typing import Any, Tuple

import torch
import torch.nn.functional as F


def compute_scale_gated_nwd_xyxy(
    gt_xyxy: torch.Tensor,
    pd_xyxy: torch.Tensor,
    *,
    area_thresh: float = 1024.0,
    tau: float = 256.0,
    w_max: float = 0.4,
    c: float = 12.8,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Normalized Wasserstein Distance (NWD) and smooth GT-area gate omega(A_gt).

    Args:
        gt_xyxy: Ground-truth boxes in PIXEL xyxy format, shape `(..., 4)`.
        pd_xyxy: Predicted boxes in PIXEL xyxy format, shape `(..., 4)`.
        area_thresh: Area threshold A_0 in px^2 (default 1024 = 32x32 COCO small boundary).
        tau: Smooth sigmoid temperature in px^2.
        w_max: Maximum NWD weight for tiny targets (`0 < w_max <= 1`).
        c: NWD normalization constant in pixels (default 12.8).

    Returns:
        nwd: `(...,)` tensor in `(0, 1]`.
        omega: `(...,)` detached gate tensor in `[0, w_max]` (near 0 for large vehicles -> zero harm!).
    """
    gw = (gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp(min=eps)
    gh = (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp(min=eps)
    gx = 0.5 * (gt_xyxy[..., 0] + gt_xyxy[..., 2])
    gy = 0.5 * (gt_xyxy[..., 1] + gt_xyxy[..., 3])

    pw = (pd_xyxy[..., 2] - pd_xyxy[..., 0]).clamp(min=eps)
    ph = (pd_xyxy[..., 3] - pd_xyxy[..., 1]).clamp(min=eps)
    px = 0.5 * (pd_xyxy[..., 0] + pd_xyxy[..., 2])
    py = 0.5 * (pd_xyxy[..., 1] + pd_xyxy[..., 3])

    w2_sq = (gx - px).pow(2) + (gy - py).pow(2) + 0.25 * ((gw - pw).pow(2) + (gh - ph).pow(2))
    nwd = torch.exp(-torch.sqrt(w2_sq + eps) / c)

    # Detached pure GT-area gate: large vehicles (e.g. OtherVehicle 150x80) get omega ~ 0
    gt_area = (gw * gh).detach()
    omega = w_max * torch.sigmoid((area_thresh - gt_area) / tau)
    return nwd, omega


def _patch_assigner_c1(
    assigner: Any,
    *,
    area_thresh: float,
    tau: float,
    w_max: float,
    c: float,
) -> None:
    """Patch `TaskAlignedAssigner.iou_calculation` with Scale-Gated NWD (Mod C1)."""
    from ultralytics.utils.metrics import bbox_iou

    def _scale_gated_iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        ciou = bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        nwd, omega = compute_scale_gated_nwd_xyxy(
            gt_bboxes,
            pd_bboxes,
            area_thresh=area_thresh,
            tau=tau,
            w_max=w_max,
            c=c,
        )
        return ((1.0 - omega) * ciou + omega * nwd).clamp_(0.0, 1.0)

    assigner.iou_calculation = types.MethodType(_scale_gated_iou_calculation, assigner)


def _patch_bbox_loss_c2(
    bbox_loss_mod: Any,
    *,
    area_thresh: float,
    tau: float,
    w_max: float,
    c: float,
) -> None:
    """Patch `BboxLoss.forward` to couple Scale-Gated NWD into box regression loss (Mod C2)."""
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    def _scale_gated_bbox_forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fg_idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg_idx].sum(-1, keepdim=True)

        pred_fg = pred_bboxes[fg_idx]
        target_fg = target_bboxes[fg_idx]
        ciou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)

        # Convert stride-normalized grid boxes back to exact pixel coordinates for area gating & NWD
        fg_stride = stride[fg_idx[1]] if stride.ndim == 2 else stride[fg_idx]
        nwd, omega = compute_scale_gated_nwd_xyxy(
            target_fg * fg_stride,
            pred_fg * fg_stride,
            area_thresh=area_thresh,
            tau=tau,
            w_max=w_max,
            c=c,
        )
        nwd = nwd.unsqueeze(-1)
        omega = omega.unsqueeze(-1)
        coupled_sim = (1.0 - omega) * ciou + omega * nwd
        loss_iou = ((1.0 - coupled_sim) * weight).sum() / target_scores_sum

        # Preserve exact Ultralytics 8.4.218 DFL / reg_max=1 normalized L1 branch
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_idx].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_idx]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_norm = pred_dist * stride
            pred_dist_norm[..., 0::2] /= imgsz[1]
            pred_dist_norm[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_norm[fg_idx], target_ltrb[fg_idx], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    bbox_loss_mod.forward = types.MethodType(_scale_gated_bbox_forward, bbox_loss_mod)


def patch_criterion_with_nwd(
    criterion: Any,
    *,
    use_c1: bool,
    use_c2: bool,
    area_thresh: float = 1024.0,
    tau: float = 256.0,
    w_max: float = 0.4,
    c: float = 12.8,
) -> Any:
    """Patch both `one2many` and `one2one` branches of E2ELoss (or single v8DetectionLoss)."""
    branches = []
    if hasattr(criterion, "one2many") and hasattr(criterion, "one2one"):
        branches = [("one2many", criterion.one2many), ("one2one", criterion.one2one)]
    else:
        branches = [("single", criterion)]

    for bname, branch in branches:
        if use_c1 and hasattr(branch, "assigner"):
            _patch_assigner_c1(
                branch.assigner,
                area_thresh=area_thresh,
                tau=tau,
                w_max=w_max,
                c=c,
            )
        if use_c2 and hasattr(branch, "bbox_loss"):
            _patch_bbox_loss_c2(
                branch.bbox_loss,
                area_thresh=area_thresh,
                tau=tau,
                w_max=w_max,
                c=c,
            )
        print(
            f"[MOD C] Patched {bname} branch: C1(assigner)={use_c1}, C2(bbox_loss)={use_c2} "
            f"[A0={area_thresh}, tau={tau}, w_max={w_max}, C={c}]"
        )
    return criterion


def attach_nwd_e2e_hook(
    yolo_model: Any,
    *,
    use_c1: bool,
    use_c2: bool,
    area_thresh: float = 1024.0,
    tau: float = 256.0,
    w_max: float = 0.4,
    c: float = 12.8,
) -> None:
    """Attach training-only Scale-Gated NWD patch via Ultralytics `on_train_start` callback."""
    if not (use_c1 or use_c2):
        return

    def _on_train_start(trainer: Any) -> None:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        orig_init = raw_model.init_criterion

        def _wrapped_init_criterion() -> Any:
            crit = orig_init()
            return patch_criterion_with_nwd(
                crit,
                use_c1=use_c1,
                use_c2=use_c2,
                area_thresh=area_thresh,
                tau=tau,
                w_max=w_max,
                c=c,
            )

        raw_model.init_criterion = _wrapped_init_criterion
        # If criterion was already eagerly initialized before callback, patch it immediately too
        if getattr(raw_model, "criterion", None) is not None:
            patch_criterion_with_nwd(
                raw_model.criterion,
                use_c1=use_c1,
                use_c2=use_c2,
                area_thresh=area_thresh,
                tau=tau,
                w_max=w_max,
                c=c,
            )

    yolo_model.add_callback("on_train_start", _on_train_start)
    print(f"[MOD C] Registered on_train_start hook (C1={use_c1}, C2={use_c2}).")
