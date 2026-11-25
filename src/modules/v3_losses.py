"""V3 Pillar 1: Zero-Inference-Overhead O2O-Aligned & Scale-Calibrated Supervision (`ODCS`).

Directly resolves all mechanistic failure modes uncovered in our Phase 1 v1/v2 post-mortem:
1. **Sacred O2M Teacher (`Detect.forward` `x.detach()` discipline)**:
   - NEVER modifies `one2many.assigner` (which is the sole gradient source for Backbone & Neck),
     preserving the 90.0% O2M feature ceiling.
2. **Protected-Floor `ProgLoss` (Pillar 1a)**:
   - Replaces linear `0.8 -> 0.1` decay (which under-trained O2O at early-stop Ep 84 and decayed
     O2M backbone supervision below 0.25 after Ep 150) with:
       w_o2m(t) = max(o2m_floor, 0.8 - 0.7 * t / T)          (default o2m_floor = 0.45)
       w_o2o(t) = 0.2 + 0.7 * min(1.0, t / (o2o_ramp_frac*T)) (default o2o_ramp_frac = 0.60)
3. **Selective `O2M -> O2O` In-Model Self-Distillation (Pillar 1b)**:
   - Reuses the same-batch `one2many` predictions (strictly `.detach()`ed) as an online teacher:
     - **Teacher-Better Gated Box KD**: At each GT's top-1 O2M anchor `j*_g`, aligns `b_o2o` to
       `sg(b_o2m)` with CIoU ONLY when `IoU(b_o2m, gt) > IoU(b_o2o, gt)` and `IoU(b_o2m, gt) > 0.1`.
     - **Winner Classification Alignment + Runner-Up Suppression**: Soft-BCE aligns O2O logits to
       `sg(p_o2m)` at `j*_g`, while lightly suppressing O2M runner-up anchors `R_g` (top 2..7)
       EXCEPT any anchor already assigned as a positive in `one2one` (prevents self-conflict!).
4. **O2O-Only Annealed Assignment + Dead-Anchor Weight Floor + DFL-Free Exact Regression (Pillar 1c)**:
   - Tightens NWD area gating to `A0 = 256 px^2, tau = 64` (exact STAL `stride_val = 16 px` boundary ->
     <0.1% weight on `Car`, strong rescue on tiny 8x10 px `Person`) with cosine annealing back to pure
     CIoU over the first 60% of epochs to lock in `mAP50-95`.
   - Decouples `BboxLoss` foreground regression weight with a bounded floor:
       w_reg = max(target_scores.sum(-1), eps_reg * omega_eff * NWD)
     fixing the V2 dead-anchor zero-gradient defect without re-introducing V1 confidence collapse.
   - Replaces CIoU in DFL-free (`reg_max=1`) regression with `PIoU2` (Liu et al., Neural Networks 2024)
     or `SD Loss` (Yang et al., AAAI 2025), strictly computed in FP32 with detached focusing factors.

Zero extra inference parameters, zero extra FLOPs, +0.00 ms inference latency.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.modules.nwd_e2e import compute_scale_gated_nwd_xyxy


@dataclass
class V3ODCSConfig:
    """Hyperparameters for V3 Pillar 1 (`ODCS`) zero-latency supervision."""

    # Pillar 1a: Protected-Floor ProgLoss
    use_prog_floor: bool = True
    o2m_floor: float = 0.45
    o2o_ramp_frac: float = 0.60

    # Pillar 1b: Selective O2M -> O2O Self-Distillation
    use_o2m_kd: bool = False
    kd_weight: float = 0.30
    kd_sup_weight: float = 0.10
    kd_warmup_epochs: int = 10
    kd_topk_runners: int = 7

    # Pillar 1c: O2O-only Annealed NWD Assignment + Dead-Anchor Floor + Exact Regression
    use_o2o_annealed_assign: bool = False
    area_thresh: float = 256.0
    tau: float = 64.0
    w_max: float = 0.25
    nwd_c: float = 8.0
    anneal_frac: float = 0.60
    reg_floor_eps: float = 0.25

    # DFL-free bounding box regression loss: "ciou", "piou2", or "sdloss"
    reg_loss_type: str = "ciou"
    piou_lambda: float = 1.3
    sd_delta: float = 0.5
    sd_b_max: float = 256.0
    reg_both_heads: bool = False  # False = sacred O2M teacher; True = also apply PIoU2/SDLoss to O2M


def _cosine_anneal(epoch: int, total_epochs: int, anneal_frac: float) -> float:
    """Smoothly decays from 1.0 at epoch 0 to 0.0 at `anneal_frac * total_epochs`."""
    t_end = max(float(total_epochs - 1) * float(anneal_frac), 1.0)
    ratio = min(max(float(epoch) / t_end, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * ratio))


def compute_piou2_loss_fp32(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    *,
    lam: float = 1.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Powerful-IoU v2 (Liu et al., Neural Networks 2024) per-box loss strictly in FP32.

    Scale-invariant: works identically on stride-normalized grid boxes or pixel boxes.
    Returns shape `(N, 1)` matching Ultralytics `BboxLoss` conventions.
    """
    dev_type = pred_xyxy.device.type if pred_xyxy.device.type in ("cuda", "cpu") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        b1 = pred_xyxy.float()
        b2 = target_xyxy.float()

        b1_x1, b1_y1, b1_x2, b1_y2 = b1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = b2.chunk(4, dim=-1)

        w1 = (b1_x2 - b1_x1).clamp(min=eps)
        h1 = (b1_y2 - b1_y1).clamp(min=eps)
        w2 = (b2_x2 - b2_x1).clamp(min=eps)
        h2 = (b2_y2 - b2_y1).clamp(min=eps)

        # Standard IoU
        inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0.0)
        inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (w1 * h1 + w2 * h2 - inter).clamp(min=eps)
        iou = (inter / union).clamp(0.0, 1.0)

        # Target-size-adaptive four-edge penalty P
        dw1 = torch.abs(b1_x1 - b2_x1)
        dw2 = torch.abs(b1_x2 - b2_x2)
        dh1 = torch.abs(b1_y1 - b2_y1)
        dh2 = torch.abs(b1_y2 - b2_y2)
        p_factor = (0.25 * ((dw1 + dw2) / w2 + (dh1 + dh2) / h2)).clamp(min=0.0, max=20.0)

        l_piou = (1.0 - iou) + (1.0 - torch.exp(-(p_factor * p_factor)))

        # Non-monotonic focusing weight u(lambda * q) — strictly detached!
        q = torch.exp(-p_factor).detach()
        x = lam * q
        u = (3.0 * x * torch.exp(-(x * x))).detach()
        return u * l_piou


def compute_sd_bbox_loss_fp32(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    target_area_px: torch.Tensor,
    *,
    delta: float = 0.5,
    b_max: float = 256.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Scale-based Dynamic BBox Loss (`SDB Loss`, Yang et al., AAAI 2025) strictly in FP32.

    Splits CIoU into scale/shape term `L_BS = 1 - IoU + alpha*v` and center location term
    `L_BL = rho^2 / c^2`, dynamically reweighting tiny targets (`A_gt < b_max`) toward center
    stability while degenerating identically to standard CIoU for `A_gt >= b_max`.
    Returns shape `(N, 1)`.
    """
    dev_type = pred_xyxy.device.type if pred_xyxy.device.type in ("cuda", "cpu") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        b1 = pred_xyxy.float()
        b2 = target_xyxy.float()

        b1_x1, b1_y1, b1_x2, b1_y2 = b1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = b2.chunk(4, dim=-1)

        w1 = (b1_x2 - b1_x1).clamp(min=eps)
        h1 = (b1_y2 - b1_y1).clamp(min=eps)
        w2 = (b2_x2 - b2_x1).clamp(min=eps)
        h2 = (b2_y2 - b2_y1).clamp(min=eps)

        # IoU
        inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0.0)
        inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (w1 * h1 + w2 * h2 - inter).clamp(min=eps)
        iou = (inter / union).clamp(0.0, 1.0)

        # Center distance term rho^2 / c^2
        cw = (torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)).clamp(min=eps)
        ch = (torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)).clamp(min=eps)
        c2 = (cw * cw + ch * ch).clamp(min=eps)
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) * 0.25
        l_bl = (rho2 / c2).clamp(min=0.0, max=10.0)

        # Aspect ratio consistency term alpha * v (matching Ultralytics bbox_iou)
        v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1.0 + eps))
        l_bs = (1.0 - iou) + alpha * v

        # Scale-based dynamic weights (strictly detached from area)
        area = target_area_px.detach().float().view(-1, 1)
        beta_b = torch.clamp((area / float(b_max)) * float(delta), min=0.0, max=float(delta))
        w_bs = 1.0 - float(delta) + beta_b
        w_bl = 1.0 + float(delta) - beta_b
        return w_bs * l_bs + w_bl * l_bl


def _wrap_o2m_assigner_observer(assigner: Any) -> None:
    """Non-invasively record `one2many`'s exact `(align_metric, mask_pos, mask_gt)` for O2M->O2O KD.

    Does NOT alter any computation, assignment, or return value of the sacred `one2many` teacher!
    """
    orig_select_highest = assigner.select_highest_overlaps
    assigner._v3_last_cache = None

    def _observing_select_highest(self, mask_pos, overlaps, n_max_boxes, align_metric):
        # Capture pre-conflict positive mask and alignment scores before returning untouched outputs
        target_gt_idx, fg_mask, out_mask_pos = orig_select_highest(mask_pos, overlaps, n_max_boxes, align_metric)
        self._v3_last_cache = {
            "align_metric": align_metric.detach(),
            "mask_pos": out_mask_pos.detach(),
        }
        return target_gt_idx, fg_mask, out_mask_pos

    assigner.select_highest_overlaps = types.MethodType(_observing_select_highest, assigner)


def _patch_o2o_assigner_v3(assigner: Any, criterion: Any, cfg: V3ODCSConfig) -> None:
    """Patch ONLY `one2one.assigner.get_box_metrics` with annealed scale-gated NWD (A0=256)."""
    from ultralytics.utils.metrics import bbox_iou

    def _v3_o2o_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, **extra):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        indices = mask_gt.nonzero(as_tuple=True)
        bbox_scores = pd_scores[indices[0], indices[2], gt_labels[indices[0], indices[1], 0].long()]

        gt_sel = gt_bboxes[indices[0], indices[1]]
        pd_sel = pd_bboxes[indices[0], indices[2]]

        epoch = int(getattr(criterion, "_v3_epoch", getattr(criterion, "updates", 0)))
        total_epochs = int(getattr(criterion, "_v3_total_epochs", max(getattr(criterion.one2one.hyp, "epochs", 200), 1)))
        eta = _cosine_anneal(epoch, total_epochs, cfg.anneal_frac)

        dev_type = gt_sel.device.type if gt_sel.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            gt_f = gt_sel.float()
            pd_f = pd_sel.float()
            ciou = bbox_iou(gt_f, pd_f, xywh=False, CIoU=True).squeeze(-1).clamp_(0.0, 1.0)
            if eta > 1e-6:
                nwd, omega = compute_scale_gated_nwd_xyxy(
                    gt_f,
                    pd_f,
                    area_thresh=cfg.area_thresh,
                    tau=cfg.tau,
                    w_max=cfg.w_max,
                    c=cfg.nwd_c,
                )
                omega_eff = omega * eta
                u_match = ((1.0 - omega_eff) * ciou + omega_eff * nwd).clamp_(0.0, 1.0)
            else:
                u_match = ciou
            align_values = bbox_scores.float().pow(self.alpha) * u_match.pow(self.beta)

        # Keep float32 to prevent float16 eps=1e-9 -> 0.0 division-by-zero NaN in norm_align_metric
        overlap_values = ciou.float()
        align_values = align_values.float()

        flat = (indices[0] * self.n_max_boxes + indices[1]) * na + indices[2]
        overlaps = overlap_values.new_zeros(mask_gt.numel()).scatter_(0, flat, overlap_values).view_as(mask_gt)
        align_metric = align_values.new_zeros(mask_gt.numel()).scatter_(0, flat, align_values).view_as(mask_gt)
        return align_metric, overlaps

    assigner.get_box_metrics = types.MethodType(_v3_o2o_get_box_metrics, assigner)


def _patch_bbox_loss_v3(
    bbox_loss_mod: Any,
    criterion: Any,
    cfg: V3ODCSConfig,
    *,
    apply_dead_anchor_floor: bool,
) -> None:
    """Patch `BboxLoss.forward` with PIoU2 / SD Loss and optional dead-anchor regression floor."""
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    def _v3_bbox_forward(
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
        cls_weight = target_scores[fg_idx].sum(-1, keepdim=True)

        pred_fg = pred_bboxes[fg_idx]
        target_fg = target_bboxes[fg_idx]
        fg_stride = stride[fg_idx[1]] if stride.ndim == 2 else stride[fg_idx]

        dev_type = pred_fg.device.type if pred_fg.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            pred_fg_f = pred_fg.float()
            target_fg_f = target_fg.float()
            stride_f = fg_stride.float()

            target_px = target_fg_f * stride_f
            pred_px = pred_fg_f * stride_f
            gt_w_px = (target_px[..., 2] - target_px[..., 0]).clamp(min=1e-6)
            gt_h_px = (target_px[..., 3] - target_px[..., 1]).clamp(min=1e-6)
            gt_area_px = (gt_w_px * gt_h_px).detach()

            # 1. Decoupled regression weight floor fixes V2 dead-anchor zero-gradient defect
            reg_weight = cls_weight.float()
            if apply_dead_anchor_floor and pred_fg_f.numel() > 0:
                epoch = int(getattr(criterion, "_v3_epoch", getattr(criterion, "updates", 0)))
                total_epochs = int(
                    getattr(criterion, "_v3_total_epochs", max(getattr(criterion.one2one.hyp, "epochs", 200), 1))
                )
                eta = _cosine_anneal(epoch, total_epochs, cfg.anneal_frac)
                if eta > 1e-6:
                    nwd, omega = compute_scale_gated_nwd_xyxy(
                        target_px,
                        pred_px,
                        area_thresh=cfg.area_thresh,
                        tau=cfg.tau,
                        w_max=cfg.w_max,
                        c=cfg.nwd_c,
                    )
                    floor_w = (cfg.reg_floor_eps * (omega * eta) * nwd).detach().unsqueeze(-1)
                    reg_weight = torch.maximum(reg_weight, floor_w)

            reg_denom = torch.maximum(
                target_scores_sum.float(),
                reg_weight.sum().detach(),
            ).clamp_(min=1.0)

            # 2. DFL-free exact regression loss
            if cfg.reg_loss_type == "piou2":
                per_box_loss = compute_piou2_loss_fp32(pred_fg_f, target_fg_f, lam=cfg.piou_lambda)
            elif cfg.reg_loss_type == "sdloss":
                per_box_loss = compute_sd_bbox_loss_fp32(
                    pred_fg_f,
                    target_fg_f,
                    gt_area_px,
                    delta=cfg.sd_delta,
                    b_max=cfg.sd_b_max,
                )
            else:
                ciou = bbox_iou(pred_fg_f, target_fg_f, xywh=False, CIoU=True)
                per_box_loss = 1.0 - ciou

            loss_iou = (per_box_loss * reg_weight).sum() / reg_denom

        # 3. Preserve exact Ultralytics 8.4.218 DFL / reg_max=1 normalized L1 branch
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_idx].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_idx])
                * reg_weight.to(pred_dist.dtype)
            )
            loss_dfl = loss_dfl.sum() / reg_denom.to(pred_dist.dtype)
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_norm = pred_dist * stride
            pred_dist_norm[..., 0::2] /= imgsz[1]
            pred_dist_norm[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_norm[fg_idx], target_ltrb[fg_idx], reduction="none").mean(-1, keepdim=True)
                * reg_weight.to(pred_dist.dtype)
            )
            loss_dfl = loss_dfl.sum() / reg_denom.to(pred_dist.dtype)

        return loss_iou, loss_dfl

    bbox_loss_mod.forward = types.MethodType(_v3_bbox_forward, bbox_loss_mod)


def compute_selective_o2m_to_o2o_kd(
    *,
    one2many_branch: Any,
    one2one_branch: Any,
    o2m_preds: Dict[str, torch.Tensor],
    o2o_preds: Dict[str, torch.Tensor],
    o2m_cache: Optional[Dict[str, torch.Tensor]],
    o2o_assigned: Tuple[Any, ...],
    batch: Dict[str, torch.Tensor],
    cfg: V3ODCSConfig,
    epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute zero-inference-cost Selective O2M -> O2O Self-Distillation (box + cls) in FP32.

    Gradients flow EXCLUSIVELY into `o2o_preds`; all `o2m_preds` and teacher assignments are detached.
    Returns `(kd_box_loss, kd_cls_loss)` unmultiplied by `batch_size`, scaled by `hyp.box`/`hyp.cls`.
    """
    from ultralytics.utils.metrics import bbox_iou

    device = o2o_preds["scores"].device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if o2m_cache is None:
        return zero, zero

    o2o_fg_mask, _, _, anchor_points, stride_tensor = o2o_assigned
    o2m_align = o2m_cache["align_metric"]  # (B, M, A), detached
    o2m_mask_pos = o2m_cache["mask_pos"]   # (B, M, A), detached

    if o2m_align.numel() == 0 or o2m_align.shape[1] == 0:
        return zero, zero

    dev_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        pred_dist_o2m = o2m_preds["boxes"].permute(0, 2, 1).contiguous().detach().float()
        pred_scores_o2m = o2m_preds["scores"].permute(0, 2, 1).contiguous().detach().float()

        pred_dist_o2o = o2o_preds["boxes"].permute(0, 2, 1).contiguous().float()
        pred_scores_o2o = o2o_preds["scores"].permute(0, 2, 1).contiguous().float()

        batch_size = pred_scores_o2o.shape[0]
        num_anchors = pred_scores_o2o.shape[1]
        imgsz = (
            torch.tensor(o2m_preds["feats"][0].shape[2:], device=device, dtype=torch.float32)
            * one2many_branch.stride[0]
        )

        # Reconstruct GT pixel boxes (B, M, 4) identical to v8DetectionLoss
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = one2many_branch.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        _, gt_bboxes_px = targets.split((1, 4), 2)
        mask_gt = gt_bboxes_px.sum(2).gt(0.0)  # (B, M)

        if not mask_gt.any():
            return zero, zero

        # Teacher alignment ranking over assigned positive candidates per GT
        pos_align = (o2m_align.float() * (o2m_mask_pos > 0).float()).detach()
        k = min(int(cfg.kd_topk_runners), num_anchors)
        topk_vals, topk_idx = torch.topk(pos_align, k=k, dim=-1)  # (B, M, k)

        valid_gt = mask_gt & (topk_vals[..., 0] > 0.0)
        if not valid_gt.any():
            return zero, zero

        b_idx, g_idx = valid_gt.nonzero(as_tuple=True)
        n_pos = float(b_idx.numel())
        a_star = topk_idx[b_idx, g_idx, 0]  # Best teacher anchor j*_g for each valid GT

        # Decode grid-scale bounding boxes for teacher (detached) & student (grad)
        anchors_f = anchor_points.float()
        strides_f = stride_tensor.float()
        o2m_boxes = one2many_branch.bbox_decode(anchors_f, pred_dist_o2m).detach()
        o2o_boxes = one2one_branch.bbox_decode(anchors_f, pred_dist_o2o)

        box_t = o2m_boxes[b_idx, a_star]
        box_s = o2o_boxes[b_idx, a_star]
        gt_grid = gt_bboxes_px[b_idx, g_idx].float() / strides_f[a_star]

        # 1. Teacher-Better Gated Box KD
        iou_t = bbox_iou(box_t, gt_grid, xywh=False, CIoU=False).squeeze(-1).detach()
        iou_s = bbox_iou(box_s.detach(), gt_grid, xywh=False, CIoU=False).squeeze(-1).detach()
        better_mask = (iou_t > iou_s) & (iou_t > 0.10)
        if better_mask.any():
            ciou_ts = bbox_iou(box_s[better_mask], box_t[better_mask], xywh=False, CIoU=True).squeeze(-1)
            kd_box_raw = (1.0 - ciou_ts).sum() / max(n_pos, 1.0)
        else:
            kd_box_raw = zero

        # 2. Winner Classification Alignment at j*_g
        p_teacher = pred_scores_o2m[b_idx, a_star].sigmoid().detach()  # (N_pos, nc)
        z_student = pred_scores_o2o[b_idx, a_star]                      # (N_pos, nc)
        kd_cls_pos = F.binary_cross_entropy_with_logits(z_student, p_teacher, reduction="none").sum() / max(n_pos, 1.0)

        # 3. Runner-Up Redundancy Suppression on R_g = top 2..k \ fg_o2o
        kd_cls_sup = zero
        if k > 1 and cfg.kd_sup_weight > 0.0:
            run_vals = topk_vals[b_idx, g_idx, 1:]   # (N_pos, k-1)
            run_idx = topk_idx[b_idx, g_idx, 1:]     # (N_pos, k-1)
            o2o_fg_bool = o2o_fg_mask.bool()
            b_exp = b_idx.unsqueeze(1).expand_as(run_idx)
            is_o2o_pos = o2o_fg_bool[b_exp, run_idx]
            # CRITICAL: Exclude anchors currently assigned as positives in one2one to avoid gradient conflict
            sup_mask = (run_vals > 0.0) & (~is_o2o_pos)
            if sup_mask.any():
                b_sup = b_exp[sup_mask]
                a_sup = run_idx[sup_mask]
                z_run = pred_scores_o2o[b_sup, a_sup]
                kd_cls_sup = (
                    F.binary_cross_entropy_with_logits(z_run, torch.zeros_like(z_run), reduction="none").sum()
                    / max(n_pos, 1.0)
                )

        warmup = min(1.0, float(epoch + 1) / float(max(cfg.kd_warmup_epochs, 1)))
        scale = warmup * float(cfg.kd_weight)
        hyp_box = float(getattr(one2one_branch.hyp, "box", 7.5))
        hyp_cls = float(getattr(one2one_branch.hyp, "cls", 0.5))

        kd_box_scaled = scale * hyp_box * kd_box_raw
        kd_cls_scaled = scale * hyp_cls * (kd_cls_pos + float(cfg.kd_sup_weight) * kd_cls_sup)
        return kd_box_scaled, kd_cls_scaled


class V3DistillE2ELoss:
    """Module-level, pickle-safe E2ELoss wrapper adding Selective O2M -> O2O Self-Distillation."""

    def __init__(self, base_criterion: Any, cfg: V3ODCSConfig) -> None:
        self._base = base_criterion
        self._v3_cfg = cfg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_base", "_v3_cfg"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)

    def update(self) -> None:
        self._base.update()

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]):
        base = self._base
        parsed = base.one2many.parse_output(preds)
        o2m_preds, o2o_preds = parsed["one2many"], parsed["one2one"]
        bs = o2m_preds["boxes"].shape[0]

        base.one2many.assigner._v3_last_cache = None
        _, loss_o2m_raw, _ = base.one2many.get_assigned_targets_and_loss(o2m_preds, batch)
        o2m_cache = base.one2many.assigner._v3_last_cache

        o2o_assigned, loss_o2o_raw, loss_o2o_detach = base.one2one.get_assigned_targets_and_loss(
            o2o_preds, batch
        )

        epoch_now = int(getattr(base, "_v3_epoch", getattr(base, "updates", 0)))
        kd_box, kd_cls = compute_selective_o2m_to_o2o_kd(
            one2many_branch=base.one2many,
            one2one_branch=base.one2one,
            o2m_preds=o2m_preds,
            o2o_preds=o2o_preds,
            o2m_cache=o2m_cache,
            o2o_assigned=o2o_assigned,
            batch=batch,
            cfg=self._v3_cfg,
            epoch=epoch_now,
        )

        total_o2m = (loss_o2m_raw * bs) * base.o2m
        total_o2o = (loss_o2o_raw * bs) * base.o2o
        kd_vec = torch.stack([kd_box, kd_cls, torch.zeros_like(kd_box)]) * bs
        total_loss = total_o2m + total_o2o + kd_vec
        return total_loss, loss_o2o_detach


def patch_criterion_v3_odcs(criterion: Any, cfg: V3ODCSConfig) -> Any:
    """Patch `E2ELoss` in-place with V3 Pillar 1 (`ODCS`) components."""
    if not (hasattr(criterion, "one2many") and hasattr(criterion, "one2one")):
        raise TypeError(f"V3 ODCS requires Ultralytics E2ELoss with one2many/one2one, got {type(criterion)}")

    criterion._v3_cfg = cfg
    criterion._v3_epoch = int(getattr(criterion, "updates", 0))
    criterion._v3_total_epochs = int(max(getattr(criterion.one2one.hyp, "epochs", 200), 1))

    # --- Pillar 1a: Protected-Floor ProgLoss ---
    if cfg.use_prog_floor:
        def _sync_prog_weights(self, epoch_idx: int) -> None:
            t = max(int(epoch_idx), 0)
            total = max(int(getattr(self, "_v3_total_epochs", getattr(self.one2one.hyp, "epochs", 200))) - 1, 1)
            prog = min(max(t / float(total), 0.0), 1.0)
            self.o2m = max(float(cfg.o2m_floor), 0.8 - 0.7 * prog)
            ramp_den = max(float(total) * float(cfg.o2o_ramp_frac), 1.0)
            ramp = min(max(t / ramp_den, 0.0), 1.0)
            self.o2o = 0.2 + 0.7 * ramp

        def _v3_update(self) -> None:
            self.updates += 1
            self._v3_epoch = int(self.updates)
            self._v3_sync_prog_weights(self._v3_epoch)

        criterion._v3_sync_prog_weights = types.MethodType(_sync_prog_weights, criterion)
        criterion.update = types.MethodType(_v3_update, criterion)
        criterion._v3_sync_prog_weights(criterion._v3_epoch)

    # --- Pillar 1c: O2O-Only Annealed Assignment + Dead-Anchor Floor + Exact Regression ---
    if cfg.use_o2o_annealed_assign:
        _patch_o2o_assigner_v3(criterion.one2one.assigner, criterion, cfg)

    if cfg.use_o2o_annealed_assign or cfg.reg_loss_type != "ciou":
        _patch_bbox_loss_v3(
            criterion.one2one.bbox_loss,
            criterion,
            cfg,
            apply_dead_anchor_floor=cfg.use_o2o_annealed_assign,
        )
        if cfg.reg_both_heads and cfg.reg_loss_type != "ciou":
            _patch_bbox_loss_v3(
                criterion.one2many.bbox_loss,
                criterion,
                cfg,
                apply_dead_anchor_floor=False,
            )

    # --- Pillar 1b: Selective O2M -> O2O Self-Distillation ---
    out_criterion: Any = criterion
    if cfg.use_o2m_kd:
        _wrap_o2m_assigner_observer(criterion.one2many.assigner)
        out_criterion = V3DistillE2ELoss(criterion, cfg)

    print(
        f"[V3 PILLAR 1 (ODCS)] Patched E2ELoss: "
        f"ProgFloor={cfg.use_prog_floor}(o2m>={cfg.o2m_floor}, ramp={cfg.o2o_ramp_frac}) | "
        f"O2M->O2O_KD={cfg.use_o2m_kd}(w={cfg.kd_weight}, sup={cfg.kd_sup_weight}) | "
        f"O2O_AnnealAssign={cfg.use_o2o_annealed_assign}(A0={cfg.area_thresh}, tau={cfg.tau}, floor={cfg.reg_floor_eps}) | "
        f"RegLoss={cfg.reg_loss_type}(both={cfg.reg_both_heads})"
    )
    return out_criterion


def attach_v3_odcs_hook(yolo_model: Any, cfg: V3ODCSConfig) -> None:
    """Attach V3 ODCS training-only criterion patch + epoch sync callback."""
    enabled = (
        cfg.use_prog_floor
        or cfg.use_o2m_kd
        or cfg.use_o2o_annealed_assign
        or cfg.reg_loss_type != "ciou"
    )
    if not enabled:
        return

    def _on_train_start(trainer: Any) -> None:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        orig_init = raw_model.init_criterion

        def _wrapped_init_criterion() -> Any:
            crit = orig_init()
            patched = patch_criterion_v3_odcs(crit, cfg)
            patched._v3_total_epochs = int(max(getattr(trainer, "epochs", 200), 1))
            patched._v3_epoch = int(getattr(trainer, "epoch", 0))
            if hasattr(patched, "_v3_sync_prog_weights"):
                patched._v3_sync_prog_weights(patched._v3_epoch)
            return patched

        raw_model.init_criterion = _wrapped_init_criterion
        if getattr(raw_model, "criterion", None) is not None:
            patched = patch_criterion_v3_odcs(raw_model.criterion, cfg)
            patched._v3_total_epochs = int(max(getattr(trainer, "epochs", 200), 1))
            patched._v3_epoch = int(getattr(trainer, "epoch", 0))
            if hasattr(patched, "_v3_sync_prog_weights"):
                patched._v3_sync_prog_weights(patched._v3_epoch)
            raw_model.criterion = patched

    def _on_train_epoch_start(trainer: Any) -> None:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        crit = getattr(raw_model, "criterion", None)
        if crit is not None:
            crit._v3_epoch = int(getattr(trainer, "epoch", 0))
            crit._v3_total_epochs = int(max(getattr(trainer, "epochs", 200), 1))
            if hasattr(crit, "_v3_sync_prog_weights"):
                crit._v3_sync_prog_weights(crit._v3_epoch)

    yolo_model.add_callback("on_train_start", _on_train_start)
    yolo_model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    print(f"[V3 PILLAR 1 (ODCS)] Registered hooks (reg={cfg.reg_loss_type}, kd={cfg.use_o2m_kd}, assign={cfg.use_o2o_annealed_assign}).")
