"""Phase 3: Zero-Inference-Overhead Dual-Head & DFL-Free Knowledge Distillation for YOLO26.

Designed strictly around the verified Ultralytics YOLO26 (`8.4.218/219`) internals:
- `Detect.forward` applies `x_detach = [xi.detach() for xi in x]` before `one2one` so ONLY
  `one2many` sends task gradients to Backbone & Neck, while `E2ELoss` decays `o2m` from
  `0.8 -> 0.1` (starving Backbone/Neck representation learning in late epochs).
- `reg_max = 1` removes DFL discrete distributions completely, making KL-based Localization
  Distillation (LD/BCKD/D-FINE) inapplicable and leaving pure `1 - IoU` KD weak on tiny
  `8x10 px` IR targets with zero overlap.
- All teacher models and `1x1` channel adapters live OUTSIDE `trainer.model` and `trainer.ema`,
  guaranteeing that exported `best.pt` weights remain 100% pure native `YOLO26n`
  (`2.535M params / 6.035G FLOPs / +0.00 ms` inference overhead).
- Adapters use a dedicated `AdamW` optimizer stepped with the shared `GradScaler` BEFORE
  `trainer.optimizer_step()`, guaranteeing zero conflict with `LambdaLR` and zero Newton-Schulz
  distortion under `MuSGD` (critical on 18K-image DroneVehicle).

Three Orthogonal KD Pillars (all dimensionless & strictly computed in FP32):
  - M1 (`F-PKD`): STAL-floored foreground Gaussian mask (`eta_bg=0.05`) + channel-centered
    Pearson correlation alignment on Neck `P3/P4/P5` via single `1x1 Conv2d(..., bias=True)`.
    Also includes fair literature baselines (`cwd`, `official_l2`) for controlled comparison.
  - M2 (`CrossKD-O2O`): Feeds un-detached adapted student Neck features `Phi(F_S)` directly into
    the frozen `Teacher O2O` detection head (`one2one_cv2 / one2one_cv3`), establishing a
    differentiable gradient bridge from the teacher's sparse one-to-one decision boundary back
    to Student Backbone & Neck (bypassing `x_detach` without disturbing native task stability).
    Because `M1` Pearson correlation is invariant to affine shift/scale, `M2` calibrates `Phi`'s
    exact scale & bias for `Teacher O2O`'s frozen `BatchNorm2d` layers with zero conflict.
  - M3 (`QG-Head`): Teacher-better quality-gated DFL-free localization (`CIoU + GT-normalized
    4-edge Smooth-L1`, with baseline `"iou"`/`"giou"` modes) + temperature-smoothed dual-head
    classification (`O2M` + `O2O`) + optional soft `Teacher O2O` runner-up calibration.

Also ships Track B primitives (Decoupled Dual-Teacher Illumination Safety Switch, Per-Anchor
Advantage Gate, and `3x3` AvgPool Low-Frequency Structural Alignment) for partially aligned
`DroneVehicle` (`RGB + IR -> IR-only`) privileged cross-modal distillation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class YOLO26KDConfig:
    """Configuration for Phase 3 zero-inference-overhead YOLO26 knowledge distillation."""

    # --- Teacher checkpoint ---
    teacher_weights: str = ""
    use_ema_teacher: bool = True

    # --- Master pillar switches ---
    use_m1_feat: bool = True
    feat_kd_mode: str = "fpkd"          # "fpkd" (ours), "cwd" (ICCV21), "official_l2" (PR#24894), "none"
    use_m2_cross_o2o: bool = True
    use_m3_qg_head: bool = True

    # --- Dimensionless loss weights & independent schedule r(t) ---
    lambda_feat: float = 1.0
    lambda_cross: float = 0.5
    lambda_head: float = 1.0
    kd_warmup_epochs: int = 5
    kd_decay_frac: float = 0.10         # Smooth cosine decay during final 10% epochs (clean GT polish)
    kd_final_scale: float = 0.10        # Floor of r(t) at final epoch (0.0 = pure GT, 0.1 = gentle floor)

    # --- M1 (F-PKD) parameters ---
    stal_min_radius: float = 1.0        # Grid-cell floor for Gaussian mask radius (preserves 8x10px targets)
    bg_weight: float = 0.05             # Background floor eta_bg in [0, 1]
    cwd_tau: float = 4.0                # Softmax temperature for CWD baseline
    lowpass_feat: bool = False          # 3x3 AvgPool low-frequency structural filter (Track B RGB->IR)

    # --- M2 (CrossKD-O2O) parameters ---
    cross_box_weight: float = 1.0
    cross_cls_weight: float = 1.0
    cross_sup_weight: float = 0.25      # Soft Teacher-O2O neighborhood suppression on O2M\O2O anchors
    cross_tau: float = 2.0
    cross_q_thresh: float = 0.20

    # --- M3 (QG-Head: DFL-Free QG-CIoU-L1 + Dual-Head Cls) parameters ---
    box_kd_mode: str = "ciou_l1"        # "ciou_l1" (ours), "iou" (QG-KD 2026), "giou" (BMGKD 2026)
    gamma_l1: float = 1.0               # Weight for GT-normalized 4-edge Smooth-L1
    smooth_l1_beta: float = 0.1
    q_thresh: float = 0.20              # Teacher quality threshold theta
    q_margin: float = 0.01              # Teacher-better margin delta (q_T > q_S + delta)
    use_advantage_weight: bool = True   # Multiply gate by clamp(q_T - q_S, 0, 1)
    small_boost_kappa: float = 1.0      # Bounded small-target weight: 1 + kappa * exp(-sqrt(A)/S0)
    small_boost_s0: float = 32.0        # Reference scale S0 in pixels
    cls_tau: float = 2.0                # Temperature for soft BCE classification KD
    m3_o2m_weight: float = 0.5          # Dense O2M head KD weight relative to O2O head KD
    m3_o2o_weight: float = 1.0          # Sparse O2O head KD weight (directly targets Δ_E2E)
    m3_box_weight: float = 1.0
    m3_cls_weight: float = 1.0
    use_runner_up_kd: bool = True       # Soft Teacher-O2O runner-up calibration on O2M\O2O anchors
    runner_up_weight: float = 0.25
    use_soft_gt_blend: bool = False     # Optional blend b_soft = eta*b_T + (1-eta)*b_gt

    # --- Adapter optimizer ---
    adapter_lr_mult: float = 1.0
    adapter_weight_decay: float = 1e-4

    # --- Track B (Privileged Cross-Modal RGB -> IR) safety gates ---
    night_mean_thresh: float = 0.08     # Illumination safety switch: mu_rgb < 0.08 -> alpha_cm = 0
    night_std_thresh: float = 0.05      # Low-contrast switch: sigma_rgb < 0.05 -> alpha_cm = 0
    cm_margin: float = 0.03             # Auxiliary teacher advantage margin delta_m
    cm_tau: float = 0.20                # Softmax temperature tau_m for dual-teacher preference gate


def compute_kd_schedule(
    epoch: int,
    total_epochs: int,
    *,
    warmup_epochs: int = 5,
    decay_frac: float = 0.10,
    final_scale: float = 0.10,
) -> float:
    """Independent KD schedule r(t): linear warmup -> plateau 1.0 -> cosine decay."""
    t = max(int(epoch), 0)
    T = max(int(total_epochs), 1)
    w_up = min(1.0, float(t + 1) / float(max(int(warmup_epochs), 1)))
    if decay_frac <= 0.0 or T <= 2:
        return w_up
    decay_start = max(int(round(T * (1.0 - float(decay_frac)))), warmup_epochs)
    if t < decay_start:
        return w_up
    span = max(T - 1 - decay_start, 1)
    prog = min(max(float(t - decay_start) / float(span), 0.0), 1.0)
    cos_val = 0.5 * (1.0 + math.cos(math.pi * prog))
    decay = float(final_scale) + (1.0 - float(final_scale)) * cos_val
    return w_up * decay


def _dev_type(t: torch.Tensor) -> str:
    return t.device.type if t.device.type in ("cuda", "cpu") else "cpu"


def compute_box_iou_variants_fp32(
    b1_xyxy: torch.Tensor,
    b2_xyxy: torch.Tensor,
    *,
    mode: str = "ciou",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute elementwise IoU / GIoU / CIoU strictly in FP32. Returns shape `(N,)`."""
    with torch.amp.autocast(device_type=_dev_type(b1_xyxy), enabled=False):
        b1 = b1_xyxy.float()
        b2 = b2_xyxy.float()
        x1, y1, x2, y2 = b1.unbind(-1)
        gx1, gy1, gx2, gy2 = b2.unbind(-1)

        w1 = (x2 - x1).clamp(min=eps)
        h1 = (y2 - y1).clamp(min=eps)
        w2 = (gx2 - gx1).clamp(min=eps)
        h2 = (gy2 - gy1).clamp(min=eps)

        inter_w = (torch.minimum(x2, gx2) - torch.maximum(x1, gx1)).clamp(min=0.0)
        inter_h = (torch.minimum(y2, gy2) - torch.maximum(y1, gy1)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (w1 * h1 + w2 * h2 - inter).clamp(min=eps)
        iou = (inter / union).clamp(0.0, 1.0)
        if mode == "iou":
            return iou

        cw = (torch.maximum(x2, gx2) - torch.minimum(x1, gx1)).clamp(min=eps)
        ch = (torch.maximum(y2, gy2) - torch.minimum(y1, gy1)).clamp(min=eps)
        if mode == "giou":
            c_area = (cw * ch).clamp(min=eps)
            return iou - (c_area - union) / c_area

        # CIoU
        c2 = (cw * cw + ch * ch).clamp(min=eps)
        rho2 = ((gx1 + gx2 - x1 - x2).pow(2) + (gy1 + gy2 - y1 - y2).pow(2)) * 0.25
        v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1.0 + eps))
        return iou - (rho2 / c2 + alpha * v)


def build_stal_fg_mask(
    gt_bboxes_px: torch.Tensor,
    mask_gt: torch.Tensor,
    feat_hw: Tuple[int, int],
    stride: float,
    *,
    min_radius: float = 1.0,
    bg_weight: float = 0.05,
) -> torch.Tensor:
    """Build STAL-floored Gaussian foreground spatial weight map `(B, 1, H, W)` in FP32.

    Every GT box is mapped to feature-grid coordinates with radius floored at `min_radius`
    grid cells so tiny `8x10 px` IR targets retain a clean 3x3 contextual footprint even on P4/P5.
    """
    bs, max_m = gt_bboxes_px.shape[:2]
    h, w = int(feat_hw[0]), int(feat_hw[1])
    device = gt_bboxes_px.device

    if max_m == 0 or not mask_gt.any():
        return torch.full((bs, 1, h, w), float(bg_weight), device=device, dtype=torch.float32)

    valid = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool()  # (B, M)
    gy = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
    gx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)

    s = float(stride)
    boxes = gt_bboxes_px.float()
    cx = (boxes[..., 0] + boxes[..., 2]) * (0.5 / s) - 0.5  # (B, M), matches make_anchors(..., 0.5)
    cy = (boxes[..., 1] + boxes[..., 3]) * (0.5 / s) - 0.5
    rx = ((boxes[..., 2] - boxes[..., 0]) * (0.5 / s)).clamp(min=float(min_radius))
    ry = ((boxes[..., 3] - boxes[..., 1]) * (0.5 / s)).clamp(min=float(min_radius))

    # Factorized 1D Gaussians keep peak memory tiny even for large batches
    dx2 = ((gx - cx.view(bs, max_m, 1, 1)) / rx.view(bs, max_m, 1, 1)).pow(2)  # (B, M, 1, W)
    dy2 = ((gy - cy.view(bs, max_m, 1, 1)) / ry.view(bs, max_m, 1, 1)).pow(2)  # (B, M, H, 1)
    gauss = torch.exp(-0.5 * (dx2 + dy2))  # (B, M, H, W)
    gauss = gauss * valid.view(bs, max_m, 1, 1).float()
    fg_map = gauss.amax(dim=1, keepdim=True).clamp(0.0, 1.0)  # (B, 1, H, W)
    return fg_map + float(bg_weight) * (1.0 - fg_map)


def compute_fpkd_loss(
    adapted_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    gt_bboxes_px: torch.Tensor,
    mask_gt: torch.Tensor,
    strides: Sequence[float],
    *,
    min_radius: float = 1.0,
    bg_weight: float = 0.05,
    lowpass: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """M1: Foreground-Focused Channel-Centered Pearson Correlation Distillation (F-PKD).

    Bounded in `[0, 2]`, invariant to thermal polarity inversion and teacher-student channel scale.
    Optional `lowpass=True` applies `3x3 AvgPool` to absorb 1-3 px RGB-IR parallax (Track B).
    """
    dev = _dev_type(adapted_s[0])
    with torch.amp.autocast(device_type=dev, enabled=False):
        level_losses: List[torch.Tensor] = []
        for fs, ft, st in zip(adapted_s, feats_t, strides):
            u = fs.float()
            v = ft.detach().float()
            if lowpass:
                u = F.avg_pool2d(u, kernel_size=3, stride=1, padding=1)
                v = F.avg_pool2d(v, kernel_size=3, stride=1, padding=1)

            # Channel-wise zero-mean centering + L2 normalization -> exact Pearson correlation
            u_c = u - u.mean(dim=1, keepdim=True)
            v_c = v - v.mean(dim=1, keepdim=True)
            u_n = u_c / u_c.norm(dim=1, keepdim=True).clamp(min=eps)
            v_n = v_c / v_c.norm(dim=1, keepdim=True).clamp(min=eps)
            corr = (u_n * v_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # (B, 1, H, W)
            dist = 1.0 - corr

            w_sp = build_stal_fg_mask(
                gt_bboxes_px,
                mask_gt,
                (u.shape[2], u.shape[3]),
                float(st),
                min_radius=min_radius,
                bg_weight=bg_weight,
            )
            if sample_weights is not None:
                w_sp = w_sp * sample_weights.float().view(-1, 1, 1, 1)

            denom = w_sp.sum()
            if denom < 1e-4:
                level_losses.append(torch.zeros((), device=u.device, dtype=torch.float32))
            else:
                level_losses.append((dist * w_sp).sum() / denom.clamp(min=eps))
        return torch.stack(level_losses).mean()


def compute_cwd_loss(
    adapted_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    *,
    tau: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Baseline M1 comparison: Channel-wise Knowledge Distillation (Shu et al., ICCV 2021)."""
    dev = _dev_type(adapted_s[0])
    with torch.amp.autocast(device_type=dev, enabled=False):
        t = float(max(tau, 1e-3))
        losses: List[torch.Tensor] = []
        for fs, ft in zip(adapted_s, feats_t):
            b, c, h, w = fs.shape
            u = fs.float().view(b, c, h * w) / t
            v = ft.detach().float().view(b, c, h * w) / t
            log_ps = F.log_softmax(u, dim=-1)
            pt = F.softmax(v, dim=-1)
            # Mean over batch & channels so loss is dimensionless (~0.1 - 1.5)
            kl = F.kl_div(log_ps, pt, reduction="none").sum(dim=-1).mean() * (t * t)
            losses.append(kl)
        return torch.stack(losses).mean()


def compute_official_conf_l2_loss(
    adapted_s: Sequence[torch.Tensor],
    feats_t: Sequence[torch.Tensor],
    t_scores_logits: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Baseline M1 comparison: Ultralytics PR #24894 style confidence-weighted L2 feature KD."""
    dev = _dev_type(adapted_s[0])
    with torch.amp.autocast(device_type=dev, enabled=False):
        # t_scores_logits: (B, nc, total_anchors)
        t_conf = t_scores_logits.detach().float().sigmoid().amax(dim=1)  # (B, total_anchors)
        offset = 0
        losses: List[torch.Tensor] = []
        for fs, ft in zip(adapted_s, feats_t):
            b, c, h, w = fs.shape
            n_loc = h * w
            conf_map = t_conf[:, offset : offset + n_loc].view(b, 1, h, w)
            offset += n_loc
            # Channel-mean squared error weighted by teacher objectness confidence
            mse_sp = (fs.float() - ft.detach().float()).pow(2).mean(dim=1, keepdim=True)
            denom = conf_map.sum().clamp(min=eps)
            losses.append((mse_sp * conf_map).sum() / denom)
        return torch.stack(losses).mean()


def compute_soft_bce_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    tau: float = 2.0,
) -> torch.Tensor:
    """Temperature-smoothed per-anchor binary KL divergence (mean over classes), strictly in FP32.

    Subtracts detached teacher self-BCE (`BCE(z_t/T, sigmoid(z_t/T))`) without clamping the
    differentiable tensor so gradients match exact soft BCE everywhere (zero cutoff at extreme logits)
    while remaining exact `0.0` at `z_s == z_t` and scale-invariant across 4/5/80-class tasks.
    """
    t = float(max(tau, 1e-3))
    zs = student_logits.float() / t
    zt = teacher_logits.detach().float() / t
    pt = zt.sigmoid()
    h_pt = F.binary_cross_entropy_with_logits(zt, pt, reduction="none").detach()
    kl = F.binary_cross_entropy_with_logits(zs, pt, reduction="none") - h_pt
    return kl.mean(dim=-1) * (t * t)


def compute_dfl_free_box_kd_per_anchor(
    s_boxes_grid: torch.Tensor,
    t_boxes_grid: torch.Tensor,
    gt_boxes_grid: torch.Tensor,
    anchor_points: torch.Tensor,
    *,
    mode: str = "ciou_l1",
    gamma_l1: float = 1.0,
    smooth_l1_beta: float = 0.1,
    stride_px: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-anchor DFL-free localization KD loss `(N,)` in FP32.

    Modes:
      - `"ciou_l1"` (Ours): `(1 - CIoU(b_S, b_T)) + gamma_l1 * SmoothL1(ltrb_S / s_gt, ltrb_T / s_gt)`
        where `s_gt = max(wh_gt, 8/stride)` guarantees strong sub-pixel boundary gradients on
        tiny `8x10 px` IR objects even when `IoU(b_S, b_T) == 0`.
      - `"iou"` (QG-KD, Electronics 2026 baseline): `1 - IoU(b_S, b_T)`.
      - `"giou"` (BMGKD 2026 baseline): `1 - GIoU(b_S, b_T)`.
    """
    dev = _dev_type(s_boxes_grid)
    with torch.amp.autocast(device_type=dev, enabled=False):
        bs_f = s_boxes_grid.float()
        bt_f = t_boxes_grid.detach().float()
        if mode == "iou":
            return 1.0 - compute_box_iou_variants_fp32(bs_f, bt_f, mode="iou", eps=eps)
        if mode == "giou":
            return 1.0 - compute_box_iou_variants_fp32(bs_f, bt_f, mode="giou", eps=eps)

        # Ours: CIoU + GT-scale-normalized 4-edge (l, t, r, b) Smooth-L1
        ciou_loss = 1.0 - compute_box_iou_variants_fp32(bs_f, bt_f, mode="ciou", eps=eps)
        if float(gamma_l1) <= 0.0:
            return ciou_loss

        ap = anchor_points.float()
        # 4-edge distances (l, t, r, b) in grid units
        s_ltrb = torch.cat([ap - bs_f[..., :2], bs_f[..., 2:] - ap], dim=-1)
        t_ltrb = torch.cat([ap - bt_f[..., :2], bt_f[..., 2:] - ap], dim=-1)

        gt_f = gt_boxes_grid.detach().float()
        gt_w = (gt_f[..., 2] - gt_f[..., 0]).clamp(min=eps)
        gt_h = (gt_f[..., 3] - gt_f[..., 1]).clamp(min=eps)
        if stride_px is not None:
            min_grid = (8.0 / stride_px.float().squeeze(-1)).clamp(min=0.25)
            norm_w = torch.maximum(gt_w, min_grid)
            norm_h = torch.maximum(gt_h, min_grid)
        else:
            norm_w = gt_w.clamp(min=0.5)
            norm_h = gt_h.clamp(min=0.5)
        scale = torch.stack([norm_w, norm_h, norm_w, norm_h], dim=-1).detach()
        l1_edge = F.smooth_l1_loss(
            s_ltrb / scale,
            t_ltrb / scale,
            beta=float(smooth_l1_beta),
            reduction="none",
        ).mean(dim=-1)
        return ciou_loss + float(gamma_l1) * l1_edge


# ---------------------------------------------------------------------------
# Track B Primitives: Privileged Cross-Modal (RGB + IR -> IR-only) Safety Gates
# ---------------------------------------------------------------------------

def compute_illumination_safety_switch(
    rgb_img: torch.Tensor,
    *,
    mean_thresh: float = 0.08,
    std_thresh: float = 0.05,
) -> torch.Tensor:
    """Image-level night-blackout circuit breaker `(B,)` in `{0.0, 1.0}`.

    When paired RGB is dark (`mean < 0.08`) or flat (`std < 0.05`), returns `0.0` so the
    student unconditionally falls back to the pure IR teacher with zero negative transfer.
    """
    with torch.no_grad():
        x = rgb_img.detach().float()
        # Handle both [0, 1] and [0, 255] gracefully
        if x.max() > 1.5:
            x = x / 255.0
        mu = x.mean(dim=(1, 2, 3))
        sigma = x.flatten(1).std(dim=1)
        safe = (mu >= float(mean_thresh)) & (sigma >= float(std_thresh))
        return safe.float()


def compute_dual_teacher_advantage_gate(
    q_ir: torch.Tensor,
    q_aux: torch.Tensor,
    *,
    margin: float = 0.03,
    tau: float = 0.20,
    image_safe: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Instance/anchor-level dual-teacher preference gate `g_cm` in `[0, 1]`.

    Only opens cross-modal transfer when auxiliary (RGB / RGB-T) teacher strictly outperforms
    the pure IR teacher by `margin`. Otherwise `g_cm = 0` (100% pure IR fallback).
    """
    with torch.no_grad():
        q_ir_f = q_ir.detach().float()
        q_aux_f = q_aux.detach().float()
        t = float(max(tau, 1e-3))
        prefer = (q_aux_f > (q_ir_f + float(margin))).float()
        soft_pref = torch.sigmoid((q_aux_f - q_ir_f) / t)
        gate = prefer * soft_pref
        if image_safe is not None:
            gate = gate * image_safe.detach().float()
        return gate


# ---------------------------------------------------------------------------
# Teacher Loading & Neck Feature Adapters
# ---------------------------------------------------------------------------

def _infer_detect_in_channels(model: nn.Module) -> List[int]:
    """Infer exact `[P3, P4, P5, ...]` input channel counts from `Detect.cv2`."""
    detect = model.model[-1] if hasattr(model, "model") else model[-1]
    if getattr(detect, "cv2", None) is None:
        raise TypeError("Detect head is missing `cv2` (stripped O2M branch is incompatible with dual-head KD).")
    channels: List[int] = []
    for branch in detect.cv2:
        first = branch[0]
        if hasattr(first, "conv") and hasattr(first.conv, "in_channels"):
            channels.append(int(first.conv.in_channels))
        elif hasattr(first, "in_channels"):
            channels.append(int(first.in_channels))
        else:
            raise RuntimeError(f"Cannot infer Detect input channels from {type(first)}")
    return channels


def validate_teacher_compatibility(student_model: nn.Module, teacher_model: nn.Module) -> None:
    """Pre-flight check ensuring teacher has intact dual-head (O2M + O2O), matching nc/stride/reg_max."""
    s_det = student_model.model[-1] if hasattr(student_model, "model") else student_model[-1]
    t_det = teacher_model.model[-1] if hasattr(teacher_model, "model") else teacher_model[-1]

    for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branch = getattr(t_det, attr, None)
        if branch is None or not isinstance(branch, nn.ModuleList) or len(branch) == 0:
            raise TypeError(
                f"Incompatible KD teacher: Detect.{attr} is missing or stripped! "
                "Use an unstripped dual-head YOLO26 checkpoint (regular trained best.pt)."
            )
    if int(getattr(t_det, "nc", -1)) != int(getattr(s_det, "nc", -2)):
        raise ValueError(f"KD teacher nc mismatch: teacher={getattr(t_det, 'nc', None)} vs student={getattr(s_det, 'nc', None)}")
    if int(getattr(t_det, "reg_max", -1)) != int(getattr(s_det, "reg_max", -2)):
        raise ValueError(
            f"KD teacher reg_max mismatch: teacher={getattr(t_det, 'reg_max', None)} vs student={getattr(s_det, 'reg_max', None)}"
        )
    if int(getattr(t_det, "nl", -1)) != int(getattr(s_det, "nl", -2)):
        raise ValueError(f"KD teacher scale count mismatch: teacher={getattr(t_det, 'nl', None)} vs student={getattr(s_det, 'nl', None)}")

    s_stride = [float(x) for x in s_det.stride.tolist()]
    t_stride = [float(x) for x in t_det.stride.tolist()]
    if s_stride != t_stride:
        raise ValueError(f"KD teacher stride mismatch: teacher={t_stride} vs student={s_stride}")

    s_names = getattr(student_model, "names", None)
    t_names = getattr(teacher_model, "names", None)
    if isinstance(s_names, dict) and isinstance(t_names, dict) and s_names and t_names:
        if [s_names.get(i) for i in range(int(s_det.nc))] != [t_names.get(i) for i in range(int(t_det.nc))]:
            raise ValueError(f"KD teacher class name order mismatch: teacher={t_names} vs student={s_names}")


def load_frozen_teacher(
    weights_path: str | Path,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> nn.Module:
    """Load frozen FP32 EMA teacher `DetectionModel` cleanly without polluting student state."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"KD teacher weights not found: {path}")

    teacher_model: Optional[nn.Module] = None
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            cand = ckpt.get("ema") if use_ema else None
            if cand is None:
                cand = ckpt.get("model")
            if isinstance(cand, nn.Module):
                teacher_model = cand
    except Exception:
        teacher_model = None

    if teacher_model is None:
        from ultralytics import YOLO

        teacher_model = YOLO(str(path)).model

    teacher_model = teacher_model.float().to(device).eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    # Keep Detect.export = False so eval forward returns `(decoded_y, raw_dict)`
    t_detect = teacher_model.model[-1]
    if hasattr(t_detect, "export"):
        t_detect.export = False
    return teacher_model


class NeckChannelAdapters(nn.Module):
    """Minimal single-layer `1x1 Conv2d(C_s, C_t, bias=True)` adapters for P3/P4/P5.

    - Zero nonlinearity (prevents 2K small-sample overfitting).
    - `bias=True` gives `M2 CrossKD-O2O` exact affine freedom to match `Teacher O2O`'s
      frozen `BatchNorm2d(running_mean, running_var)` orthogonally to scale/shift-invariant `M1 F-PKD`.
    - Stored OUTSIDE `trainer.model`, so exported `best.pt` has +0 params / +0 FLOPs / +0.00 ms.
    """

    def __init__(self, s_channels: Sequence[int], t_channels: Sequence[int]) -> None:
        super().__init__()
        if len(s_channels) != len(t_channels):
            raise ValueError(f"Student/Teacher Detect scale mismatch: {s_channels} vs {t_channels}")
        projs: List[nn.Module] = []
        for cs, ct in zip(s_channels, t_channels):
            conv = nn.Conv2d(int(cs), int(ct), kernel_size=1, stride=1, padding=0, bias=True)
            if int(cs) == int(ct):
                nn.init.eye_(conv.weight.view(ct, cs))
            else:
                nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            nn.init.zeros_(conv.bias)
            projs.append(conv)
        self.projs = nn.ModuleList(projs)

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [proj(f.float()) for proj, f in zip(self.projs, feats)]


def _return_none() -> None:
    """Pickle-safe reducer returning `None` so criterion wrappers never serialize teacher weights."""
    return None


class _FeatureCaptureHook:
    """Module-level, deepcopy/pickle-safe forward_pre_hook capturing Neck `[P3, P4, P5]`."""

    def __init__(self, detach: bool = False) -> None:
        self.detach = detach
        self.feats: Optional[List[torch.Tensor]] = None

    def __call__(self, module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], (list, tuple)):
            if self.detach:
                self.feats = [f.detach() for f in inputs[0]]
            elif module.training:
                self.feats = list(inputs[0])

    def pop(self) -> Optional[List[torch.Tensor]]:
        out = self.feats
        self.feats = None
        return out

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_FeatureCaptureHook":
        return _FeatureCaptureHook(detach=self.detach)

    def __getstate__(self) -> Dict[str, Any]:
        return {"detach": self.detach}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.detach = bool(state.get("detach", False))
        self.feats = None


# ---------------------------------------------------------------------------
# Main Zero-Inference-Overhead YOLO26 Distillation Criterion
# ---------------------------------------------------------------------------

class YOLO26DistillCriterion:
    """Wraps Ultralytics `E2ELoss` with M1 (F-PKD), M2 (CrossKD-O2O), and M3 (QG-Head).

    - Runs ONLY when `student_model.training` is True (validation loss remains 100% native).
    - Keeps teacher and adapters outside `student_model` -> `best.pt` is pure 2.535M `YOLO26n`.
    - `__deepcopy__` / `__getstate__` return `None` so `save_model()` NEVER copies or pickles
      frozen teacher weights or adapters into `last.pt` / `best.pt`.
    - All KD computations run strictly in FP32 with zero-mask guards against NaN.
    """

    def __init__(
        self,
        base_criterion: Any,
        student_model: nn.Module,
        teacher_model: nn.Module,
        adapters: NeckChannelAdapters,
        cfg: YOLO26KDConfig,
        *,
        expected_s_channels: Sequence[int],
        teacher_hook: Optional[_FeatureCaptureHook] = None,
    ) -> None:
        self._base = base_criterion
        self._student_model = student_model
        self._teacher_model = teacher_model
        self._adapters = adapters
        self._kd_cfg = cfg
        self._expected_s_channels = [int(c) for c in expected_s_channels]
        self._teacher_hook = teacher_hook
        self._kd_epoch = 0
        self._kd_total_epochs = int(max(getattr(base_criterion.one2one.hyp, "epochs", 100), 1))
        self._verified_feats = False
        self._epoch_kd_sum: Optional[torch.Tensor] = None
        self._epoch_kd_count: int = 0
        self._total_kd_batches: int = 0
        self._last_r_t: float = 0.0
        self._last_epoch_avg_kd: Dict[str, float] = {}

    def flush_epoch_kd_stats(self) -> Dict[str, float]:
        """Convert device-accumulated KD losses to host floats ONCE per epoch (zero per-batch sync)."""
        if self._epoch_kd_sum is None or self._epoch_kd_count == 0:
            return {}
        avg = (self._epoch_kd_sum / float(self._epoch_kd_count)).detach().cpu().tolist()
        stats = {
            "r_t": float(self._last_r_t),
            "m1": float(avg[0]),
            "m2": float(avg[1]),
            "m3": float(avg[2]),
            "batches": int(self._epoch_kd_count),
        }
        self._epoch_kd_sum.zero_()
        self._epoch_kd_count = 0
        self._last_epoch_avg_kd = stats
        return stats

    def __deepcopy__(self, memo: Dict[int, Any]) -> None:
        # Never deep-copy frozen teacher or adapters when Ultralytics snapshots `trainer.model`
        return None

    def __reduce__(self):
        return (_return_none, ())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)

    def update(self) -> None:
        self._base.update()

    @staticmethod
    def _parse_raw_e2e(out: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        parsed = out[1] if isinstance(out, tuple) else out
        if not isinstance(parsed, dict) or "one2many" not in parsed or "one2one" not in parsed:
            raise TypeError(f"Expected YOLO26 E2E dict with 'one2many'/'one2one', got {type(parsed)}")
        return parsed

    def _compute_teacher_assignments(
        self,
        t_o2m: Dict[str, torch.Tensor],
        t_o2o: Dict[str, torch.Tensor],
        gt_labels: torch.Tensor,
        gt_bboxes_px: torch.Tensor,
        mask_gt: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute clean O2M (`topk=10`) & O2O (`topk=7, topk2=1`) assignments using frozen teacher."""
        base = self._base
        with torch.no_grad():
            t_o2m_dist = t_o2m["boxes"].permute(0, 2, 1).contiguous().float()
            t_o2m_scores = t_o2m["scores"].permute(0, 2, 1).contiguous().float()
            t_o2o_dist = t_o2o["boxes"].permute(0, 2, 1).contiguous().float()
            t_o2o_scores = t_o2o["scores"].permute(0, 2, 1).contiguous().float()

            anchors_f = anchor_points.float()
            strides_f = stride_tensor.float()

            t_o2m_boxes = base.one2many.bbox_decode(anchors_f, t_o2m_dist)
            t_o2o_boxes = base.one2one.bbox_decode(anchors_f, t_o2o_dist)

            t_o2m_labels, t_o2m_gt_px, _, t_o2m_fg, _ = base.one2many.assigner(
                t_o2m_scores.sigmoid(),
                (t_o2m_boxes * strides_f).type(gt_bboxes_px.dtype),
                anchors_f * strides_f,
                gt_labels,
                gt_bboxes_px,
                mask_gt,
            )
            t_o2o_labels, t_o2o_gt_px, _, t_o2o_fg, _ = base.one2one.assigner(
                t_o2o_scores.sigmoid(),
                (t_o2o_boxes * strides_f).type(gt_bboxes_px.dtype),
                anchors_f * strides_f,
                gt_labels,
                gt_bboxes_px,
                mask_gt,
            )

        return {
            "t_o2m_scores": t_o2m_scores,
            "t_o2m_boxes": t_o2m_boxes,
            "t_o2m_labels": t_o2m_labels.long(),
            "t_o2m_gt_px": t_o2m_gt_px.float(),
            "t_o2m_fg": t_o2m_fg.bool(),
            "t_o2o_scores": t_o2o_scores,
            "t_o2o_boxes": t_o2o_boxes,
            "t_o2o_labels": t_o2o_labels.long(),
            "t_o2o_gt_px": t_o2o_gt_px.float(),
            "t_o2o_fg": t_o2o_fg.bool(),
        }

    def _compute_qg_weights(
        self,
        s_scores: torch.Tensor,
        s_boxes_grid: torch.Tensor,
        t_scores: torch.Tensor,
        t_boxes_grid: torch.Tensor,
        gt_labels_pos: torch.Tensor,
        gt_boxes_px_pos: torch.Tensor,
        strides_pos: torch.Tensor,
        *,
        require_teacher_better: bool = True,
    ) -> torch.Tensor:
        """Compute quality-gated + small-target-boosted weights `omega_i` `(N,)` under no_grad."""
        cfg = self._kd_cfg
        with torch.no_grad():
            n = s_scores.shape[0]
            if n == 0:
                return torch.zeros((0,), device=s_scores.device, dtype=torch.float32)

            idx = torch.arange(n, device=s_scores.device)
            cls_idx = gt_labels_pos.clamp(0, s_scores.shape[-1] - 1)
            gt_grid = gt_boxes_px_pos.float() / strides_pos.float()

            p_t = t_scores.detach().float().sigmoid()[idx, cls_idx]
            iou_t = compute_box_iou_variants_fp32(t_boxes_grid.detach(), gt_grid, mode="iou")
            q_t = p_t * iou_t

            p_s = s_scores.detach().float().sigmoid()[idx, cls_idx]
            iou_s = compute_box_iou_variants_fp32(s_boxes_grid.detach(), gt_grid, mode="iou")
            q_s = p_s * iou_s

            if require_teacher_better:
                gate = (q_t >= float(cfg.q_thresh)) & (q_t > (q_s + float(cfg.q_margin)))
                if cfg.use_advantage_weight:
                    m_i = gate.float() * (q_t - q_s).clamp(0.0, 1.0)
                else:
                    m_i = gate.float()
            else:
                gate = q_t >= float(cfg.cross_q_thresh)
                m_i = gate.float() * q_t.clamp(0.0, 1.0)

            gt_w = (gt_boxes_px_pos[:, 2] - gt_boxes_px_pos[:, 0]).clamp(min=1.0)
            gt_h = (gt_boxes_px_pos[:, 3] - gt_boxes_px_pos[:, 1]).clamp(min=1.0)
            scale_px = torch.sqrt(gt_w * gt_h)
            w_s = 1.0 + float(cfg.small_boost_kappa) * torch.exp(-scale_px / float(max(cfg.small_boost_s0, 1.0)))
            return m_i * w_s

    def _compute_m2_cross_o2o(
        self,
        adapted_feats: List[torch.Tensor],
        t_assign: Dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """M2: Feed un-detached `Phi(F_S)` through frozen `Teacher O2O` head (bypasses `x_detach`)."""
        cfg = self._kd_cfg
        device = adapted_feats[0].device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        t_detect = self._teacher_model.model[-1]
        with torch.amp.autocast(device_type=_dev_type(adapted_feats[0]), enabled=False):
            # Frozen teacher O2O weights (`requires_grad=False`, `.eval()` BN), differentiable wrt `adapted_feats`!
            cross_raw = t_detect.forward_head(adapted_feats, **t_detect.one2one)
            c_dist = cross_raw["boxes"].permute(0, 2, 1).contiguous().float()
            c_scores = cross_raw["scores"].permute(0, 2, 1).contiguous().float()
            anchors_f = anchor_points.float()
            strides_f = stride_tensor.float()
            c_boxes = self._base.one2one.bbox_decode(anchors_f, c_dist)

            t_o2o_fg = t_assign["t_o2o_fg"]
            t_o2m_fg = t_assign["t_o2m_fg"]
            if not t_o2o_fg.any():
                return zero

            b_idx, a_idx = t_o2o_fg.nonzero(as_tuple=True)
            t_scores_pos = t_assign["t_o2o_scores"][b_idx, a_idx]
            t_boxes_pos = t_assign["t_o2o_boxes"][b_idx, a_idx]
            gt_labels_pos = t_assign["t_o2o_labels"][b_idx, a_idx]
            gt_px_pos = t_assign["t_o2o_gt_px"][b_idx, a_idx]
            st_pos = strides_f[a_idx]
            ap_pos = anchors_f[a_idx]

            c_scores_pos = c_scores[b_idx, a_idx]
            c_boxes_pos = c_boxes[b_idx, a_idx]

            # Weight by teacher O2O quality (does not gate out on q_T > q_S so neck always tracks teacher O2O manifold)
            omega = self._compute_qg_weights(
                c_scores_pos,
                c_boxes_pos,
                t_scores_pos,
                t_boxes_pos,
                gt_labels_pos,
                gt_px_pos,
                st_pos,
                require_teacher_better=False,
            )
            w_sum = omega.sum()
            if w_sum < 1e-4:
                return zero

            loss_cls_pos = (
                compute_soft_bce_kd(c_scores_pos, t_scores_pos, tau=cfg.cross_tau) * omega
            ).sum() / w_sum.clamp(min=1e-6)

            loss_box_pos = (
                compute_dfl_free_box_kd_per_anchor(
                    c_boxes_pos,
                    t_boxes_pos,
                    gt_px_pos / st_pos,
                    ap_pos,
                    mode=cfg.box_kd_mode,
                    gamma_l1=cfg.gamma_l1,
                    smooth_l1_beta=cfg.smooth_l1_beta,
                    stride_px=st_pos,
                )
                * omega
            ).sum() / w_sum.clamp(min=1e-6)

            # Soft neighborhood suppression on `O2M \ O2O` anchors -> shapes sharp single-peak Neck features!
            loss_sup = zero
            if float(cfg.cross_sup_weight) > 0.0:
                run_mask = t_o2m_fg & (~t_o2o_fg)
                if run_mask.any():
                    rb, ra = run_mask.nonzero(as_tuple=True)
                    loss_sup = compute_soft_bce_kd(
                        c_scores[rb, ra],
                        t_assign["t_o2o_scores"][rb, ra],
                        tau=cfg.cross_tau,
                    ).mean()

            return (
                float(cfg.cross_cls_weight) * loss_cls_pos
                + float(cfg.cross_box_weight) * loss_box_pos
                + float(cfg.cross_sup_weight) * loss_sup
            )

    def _compute_m3_qg_head(
        self,
        s_o2m: Dict[str, torch.Tensor],
        s_o2o: Dict[str, torch.Tensor],
        s_o2o_fg: torch.Tensor,
        t_assign: Dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """M3: Quality-Gated DFL-Free Dual-Head Calibration (O2M + O2O + Soft Runner-Up)."""
        cfg = self._kd_cfg
        device = s_o2o["scores"].device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        with torch.amp.autocast(device_type=_dev_type(s_o2o["scores"]), enabled=False):
            anchors_f = anchor_points.float()
            strides_f = stride_tensor.float()

            def _head_branch_loss(
                s_pred: Dict[str, torch.Tensor],
                branch_loss_obj: Any,
                t_scores_all: torch.Tensor,
                t_boxes_all: torch.Tensor,
                t_labels_all: torch.Tensor,
                t_gt_px_all: torch.Tensor,
                fg_mask: torch.Tensor,
            ) -> torch.Tensor:
                if not fg_mask.any():
                    return zero
                b_idx, a_idx = fg_mask.nonzero(as_tuple=True)
                s_dist = s_pred["boxes"].permute(0, 2, 1).contiguous().float()
                s_scores = s_pred["scores"].permute(0, 2, 1).contiguous().float()
                s_boxes = branch_loss_obj.bbox_decode(anchors_f, s_dist)

                s_sc_pos = s_scores[b_idx, a_idx]
                s_bx_pos = s_boxes[b_idx, a_idx]
                t_sc_pos = t_scores_all[b_idx, a_idx]
                t_bx_pos = t_boxes_all[b_idx, a_idx]
                gt_lbl_pos = t_labels_all[b_idx, a_idx]
                gt_px_pos = t_gt_px_all[b_idx, a_idx]
                st_pos = strides_f[a_idx]
                ap_pos = anchors_f[a_idx]

                omega = self._compute_qg_weights(
                    s_sc_pos,
                    s_bx_pos,
                    t_sc_pos,
                    t_bx_pos,
                    gt_lbl_pos,
                    gt_px_pos,
                    st_pos,
                    require_teacher_better=True,
                )
                w_sum = omega.sum()
                if w_sum < 1e-4:
                    return zero

                target_bx = t_bx_pos
                gt_grid = gt_px_pos / st_pos
                if cfg.use_soft_gt_blend:
                    with torch.no_grad():
                        iou_tg = compute_box_iou_variants_fp32(t_bx_pos, gt_grid, mode="iou").unsqueeze(-1)
                        eta = 0.5 * iou_tg
                        target_bx = eta * t_bx_pos + (1.0 - eta) * gt_grid

                l_box = (
                    compute_dfl_free_box_kd_per_anchor(
                        s_bx_pos,
                        target_bx,
                        gt_grid,
                        ap_pos,
                        mode=cfg.box_kd_mode,
                        gamma_l1=cfg.gamma_l1,
                        smooth_l1_beta=cfg.smooth_l1_beta,
                        stride_px=st_pos,
                    )
                    * omega
                ).sum() / w_sum.clamp(min=1e-6)

                l_cls = (
                    compute_soft_bce_kd(s_sc_pos, t_sc_pos, tau=cfg.cls_tau) * omega
                ).sum() / w_sum.clamp(min=1e-6)

                return float(cfg.m3_box_weight) * l_box + float(cfg.m3_cls_weight) * l_cls

            loss_o2m = zero
            if float(cfg.m3_o2m_weight) > 0.0:
                loss_o2m = _head_branch_loss(
                    s_o2m,
                    self._base.one2many,
                    t_assign["t_o2m_scores"],
                    t_assign["t_o2m_boxes"],
                    t_assign["t_o2m_labels"],
                    t_assign["t_o2m_gt_px"],
                    t_assign["t_o2m_fg"],
                )

            loss_o2o = zero
            if float(cfg.m3_o2o_weight) > 0.0:
                loss_o2o = _head_branch_loss(
                    s_o2o,
                    self._base.one2one,
                    t_assign["t_o2o_scores"],
                    t_assign["t_o2o_boxes"],
                    t_assign["t_o2o_labels"],
                    t_assign["t_o2o_gt_px"],
                    t_assign["t_o2o_fg"],
                )

            # Soft Teacher-O2O runner-up calibration on `T_O2M \ (T_O2O ∪ S_O2O)` (zero GT conflict!)
            loss_run = zero
            if cfg.use_runner_up_kd and float(cfg.runner_up_weight) > 0.0:
                run_mask = t_assign["t_o2m_fg"] & (~t_assign["t_o2o_fg"]) & (~s_o2o_fg.bool())
                if run_mask.any():
                    rb, ra = run_mask.nonzero(as_tuple=True)
                    s_o2o_scores = s_o2o["scores"].permute(0, 2, 1).contiguous().float()
                    loss_run = compute_soft_bce_kd(
                        s_o2o_scores[rb, ra],
                        t_assign["t_o2o_scores"][rb, ra],
                        tau=cfg.cls_tau,
                    ).mean()

            return (
                float(cfg.m3_o2m_weight) * loss_o2m
                + float(cfg.m3_o2o_weight) * loss_o2o
                + float(cfg.runner_up_weight) * loss_run
            )

    def __call__(
        self,
        preds: Any,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Always clear teacher capture buffer cleanly even if validation or error occurs
        if self._teacher_hook is not None:
            self._teacher_hook.pop()

        base = self._base
        parsed_s = self._parse_raw_e2e(preds)
        s_o2m, s_o2o = parsed_s["one2many"], parsed_s["one2one"]

        # Validation mode: skip KD completely so val losses stay 100% comparable & fast
        if not self._student_model.training:
            return base(preds, batch)

        from ultralytics.utils.tal import make_anchors

        bs = s_o2m["boxes"].shape[0]
        device = s_o2m["boxes"].device

        # 1. Exact native YOLO26 E2ELoss computation (captures `s_o2o_fg` cleanly without extra forward)
        _, loss_o2m_raw, _ = base.one2many.get_assigned_targets_and_loss(s_o2m, batch)
        o2o_assigned, loss_o2o_raw, loss_o2o_detach = base.one2one.get_assigned_targets_and_loss(s_o2o, batch)
        s_o2o_fg = o2o_assigned[0]
        base_total = (loss_o2m_raw * bs) * base.o2m + (loss_o2o_raw * bs) * base.o2o

        # 2. Schedule factor r(t)
        cfg = self._kd_cfg
        r_t = compute_kd_schedule(
            self._kd_epoch,
            self._kd_total_epochs,
            warmup_epochs=cfg.kd_warmup_epochs,
            decay_frac=cfg.kd_decay_frac,
            final_scale=cfg.kd_final_scale,
        )
        self._last_r_t = float(r_t)
        if r_t <= 1e-6 or "img" not in batch:
            return base_total, loss_o2o_detach

        # Directly consume un-detached Neck features `s_o2m["feats"]` (zero forward hook on student!)
        s_feats = s_o2m["feats"]
        if not self._verified_feats:
            got_ch = [int(f.shape[1]) for f in s_feats]
            if got_ch != self._expected_s_channels:
                raise RuntimeError(
                    f"Student `one2many['feats']` channel mismatch: got {got_ch}, expected {self._expected_s_channels}."
                )
            if not any(f.requires_grad for f in s_feats):
                raise RuntimeError("Student `one2many['feats']` is detached; Backbone/Neck KD gradients would be blocked.")
            self._verified_feats = True

        # 3. Frozen teacher forward on identical batch["img"] (zero host-device sync)
        raw_img = batch["img"]
        if raw_img.dtype == torch.uint8:
            img = raw_img.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
        else:
            img = raw_img.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=_dev_type(img), enabled=False):
                t_out = self._teacher_model(img)
            parsed_t = self._parse_raw_e2e(t_out)
            t_o2m, t_o2o = parsed_t["one2many"], parsed_t["one2one"]
            t_feats = self._teacher_hook.pop() if self._teacher_hook is not None else None
            if t_feats is None:
                t_feats = t_o2m["feats"]

        # 4. Shared GT pixel boxes & anchors in FP32
        anchor_points, stride_tensor = make_anchors(s_o2m["feats"], base.one2many.stride, 0.5)
        imgsz = (
            torch.tensor(s_o2m["feats"][0].shape[2:], device=device, dtype=torch.float32)
            * base.one2many.stride[0].float()
        )
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = base.one2many.preprocess(targets.to(device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes_px = targets.split((1, 4), 2)
        mask_gt = gt_bboxes_px.sum(2, keepdim=True).gt(0.0)

        # 5. Compute adapted student Neck features once (shared by M1 & M2!)
        need_adapter = (cfg.use_m1_feat and cfg.feat_kd_mode != "none") or cfg.use_m2_cross_o2o
        adapted_feats: Optional[List[torch.Tensor]] = None
        if need_adapter:
            with torch.amp.autocast(device_type=_dev_type(s_feats[0]), enabled=False):
                adapted_feats = self._adapters(s_feats)

        # --- M1: Neck Feature Distillation ---
        zero = torch.zeros((), device=device, dtype=torch.float32)
        l_m1 = zero
        if cfg.use_m1_feat and cfg.feat_kd_mode != "none" and adapted_feats is not None:
            strides_list = [float(s) for s in base.one2many.stride.tolist()]
            if cfg.feat_kd_mode == "fpkd":
                l_m1 = compute_fpkd_loss(
                    adapted_feats,
                    t_feats,
                    gt_bboxes_px,
                    mask_gt,
                    strides_list,
                    min_radius=cfg.stal_min_radius,
                    bg_weight=cfg.bg_weight,
                    lowpass=cfg.lowpass_feat,
                )
            elif cfg.feat_kd_mode == "cwd":
                l_m1 = compute_cwd_loss(adapted_feats, t_feats, tau=cfg.cwd_tau)
            elif cfg.feat_kd_mode == "official_l2":
                l_m1 = compute_official_conf_l2_loss(adapted_feats, t_feats, t_o2m["scores"])
            else:
                raise ValueError(f"Unknown feat_kd_mode: {cfg.feat_kd_mode}")

        # --- Teacher Assignments (shared by M2 & M3) ---
        t_assign: Optional[Dict[str, torch.Tensor]] = None
        if cfg.use_m2_cross_o2o or cfg.use_m3_qg_head:
            t_assign = self._compute_teacher_assignments(
                t_o2m,
                t_o2o,
                gt_labels,
                gt_bboxes_px,
                mask_gt,
                anchor_points,
                stride_tensor,
            )

        # --- M2: CrossKD-O2O (bypasses `x_detach`) ---
        l_m2 = zero
        if cfg.use_m2_cross_o2o and adapted_feats is not None and t_assign is not None:
            l_m2 = self._compute_m2_cross_o2o(adapted_feats, t_assign, anchor_points, stride_tensor)

        # --- M3: QG-Head (DFL-free QG-CIoU-L1 + Dual-Head Cls) ---
        l_m3 = zero
        if cfg.use_m3_qg_head and t_assign is not None:
            l_m3 = self._compute_m3_qg_head(s_o2m, s_o2o, s_o2o_fg, t_assign, anchor_points, stride_tensor)

        # Combine into Ultralytics 3-slot loss tensor `[box, cls, l1]` * batch_size
        # Strict fail-fast on non-finite KD losses (NEVER silently degrade into pure-detection training!)
        kd_m1_s = r_t * float(cfg.lambda_feat) * l_m1
        kd_m2_s = r_t * float(cfg.lambda_cross) * l_m2
        kd_m3_s = r_t * float(cfg.lambda_head) * l_m3
        parts = torch.stack((kd_m3_s, kd_m2_s, kd_m1_s))
        if not torch.isfinite(parts).all():
            raise FloatingPointError(
                f"Non-finite KD loss at epoch={self._kd_epoch}: [M3,M2,M1]={parts.detach().cpu().tolist()}"
            )
        total_loss = base_total + parts.to(base_total.dtype) * bs

        # Device-side epoch accumulator (zero per-batch GPU->CPU sync!)
        if self._epoch_kd_sum is None:
            self._epoch_kd_sum = torch.zeros(3, device=device, dtype=torch.float32)
        self._epoch_kd_sum += torch.stack((l_m1.detach(), l_m2.detach(), l_m3.detach())).float()
        self._epoch_kd_count += 1
        self._total_kd_batches += 1
        return total_loss, loss_o2o_detach


# ---------------------------------------------------------------------------
# Ultralytics Callback Integration (Zero `best.pt` Pollution, MuSGD-Safe)
# ---------------------------------------------------------------------------

def attach_yolo26_kd_hook(yolo_model: Any, cfg: YOLO26KDConfig) -> None:
    """Attach Phase 3 zero-inference-overhead KD hooks to an Ultralytics `YOLO` instance."""
    if not (cfg.use_m1_feat or cfg.use_m2_cross_o2o or cfg.use_m3_qg_head):
        return
    if not cfg.teacher_weights:
        raise ValueError("YOLO26KDConfig.teacher_weights must be set when KD is enabled.")

    state: Dict[str, Any] = {
        "teacher": None,
        "adapters": None,
        "adapter_opt": None,
        "criterion": None,
        "base_criterion": None,
        "orig_opt_step": None,
        "handles": [],
        "adapter_steps": 0,
        "teacher_sig": "",
    }

    def _on_train_start(trainer: Any) -> None:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        device = next(raw_model.parameters()).device

        teacher_path = Path(cfg.teacher_weights)
        teacher = load_frozen_teacher(teacher_path, device, use_ema=cfg.use_ema_teacher)
        validate_teacher_compatibility(raw_model, teacher)
        teacher_sig = f"{teacher_path.name}:{teacher_path.stat().st_size}"
        state["teacher_sig"] = teacher_sig

        s_ch = _infer_detect_in_channels(raw_model)
        t_ch = _infer_detect_in_channels(teacher)
        adapters = NeckChannelAdapters(s_ch, t_ch).to(device)

        # Dedicated AdamW for adapters: completely isolated from `MuSGD` Newton-Schulz & `LambdaLR`
        base_lr = float(trainer.optimizer.param_groups[0]["lr"])
        adapter_opt = torch.optim.AdamW(
            adapters.parameters(),
            lr=base_lr * float(cfg.adapter_lr_mult),
            weight_decay=float(cfg.adapter_weight_decay),
        )

        # Restore BOTH adapters and adapter_opt from `kd_state.pt` when resuming an interrupted run
        save_dir = Path(getattr(trainer, "save_dir", "."))
        kd_state_ckpt = save_dir / "weights" / "kd_state.pt"
        if getattr(trainer.args, "resume", False) and kd_state_ckpt.is_file():
            try:
                kd_st = torch.load(str(kd_state_ckpt), map_location=device, weights_only=False)
                saved_sig = kd_st.get("teacher_sig", "")
                if saved_sig and saved_sig != teacher_sig:
                    raise ValueError(f"Teacher mismatch on resume: saved={saved_sig} vs current={teacher_sig}")
                adapters.load_state_dict(kd_st["adapters"])
                if "adapter_opt" in kd_st:
                    adapter_opt.load_state_dict(kd_st["adapter_opt"])
                print(
                    f"[PHASE 3 KD] Restored adapters + AdamW moments from {kd_state_ckpt} "
                    f"(saved_epoch={kd_st.get('epoch')})"
                )
            except Exception as exc:
                raise RuntimeError(f"Failed restoring KD resume state from {kd_state_ckpt}: {exc}") from exc

        # Pre-hook ONLY on frozen teacher (ZERO hooks attached to student_model -> immune to NaN EMA rebuild!)
        t_hook = _FeatureCaptureHook(detach=True)
        h_t = teacher.model[-1].register_forward_pre_hook(t_hook)
        state["handles"] = [h_t]
        state["teacher"] = teacher
        state["adapters"] = adapters
        state["adapter_opt"] = adapter_opt

        # Initialize native criterion eagerly & replace with deepcopy-safe KD wrapper
        base_crit = getattr(raw_model, "criterion", None)
        if base_crit is None:
            base_crit = raw_model.init_criterion()
        state["base_criterion"] = base_crit

        kd_crit = YOLO26DistillCriterion(
            base_crit,
            raw_model,
            teacher,
            adapters,
            cfg,
            expected_s_channels=s_ch,
            teacher_hook=t_hook,
        )
        kd_crit._kd_total_epochs = int(max(getattr(trainer, "epochs", 100), 1))
        kd_crit._kd_epoch = int(getattr(trainer, "epoch", 0))
        state["criterion"] = kd_crit
        raw_model.criterion = kd_crit

        # Wrap trainer.optimizer_step so adapters track exact per-step LR (incl. warmup) & step under GradScaler
        orig_opt_step = trainer.optimizer_step
        state["orig_opt_step"] = orig_opt_step

        def _wrapped_optimizer_step() -> None:
            has_grad = any(p.grad is not None for p in adapters.parameters())
            if has_grad:
                if getattr(trainer, "optimizer", None) is not None:
                    cur_lr = float(trainer.optimizer.param_groups[0]["lr"]) * float(cfg.adapter_lr_mult)
                    for pg in adapter_opt.param_groups:
                        pg["lr"] = cur_lr
                scaler = getattr(trainer, "scaler", None)
                if scaler is not None:
                    scaler.unscale_(adapter_opt)
                    torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=10.0)
                    scaler.step(adapter_opt)
                else:
                    torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=10.0)
                    adapter_opt.step()
                state["adapter_steps"] += 1
                adapter_opt.zero_grad(set_to_none=True)
            orig_opt_step()

        trainer.optimizer_step = _wrapped_optimizer_step

        n_adapt_params = sum(p.numel() for p in adapters.parameters())
        print(
            f"[PHASE 3 KD] Attached zero-latency YOLO26 KD | teacher={teacher_path.name} "
            f"| ch: {s_ch} -> {t_ch} (adapters={n_adapt_params:,} train-only params, +0 inference params) "
            f"| M1={cfg.use_m1_feat}:{cfg.feat_kd_mode}(w={cfg.lambda_feat}) "
            f"| M2_CrossO2O={cfg.use_m2_cross_o2o}(w={cfg.lambda_cross}, sup={cfg.cross_sup_weight}) "
            f"| M3_QGHead={cfg.use_m3_qg_head}:{cfg.box_kd_mode}(w={cfg.lambda_head}, run={cfg.use_runner_up_kd})"
        )

    def _on_train_epoch_start(trainer: Any) -> None:
        crit = state.get("criterion")
        if crit is not None:
            crit._kd_epoch = int(getattr(trainer, "epoch", 0))
            crit._kd_total_epochs = int(max(getattr(trainer, "epochs", 100), 1))
        adapter_opt = state.get("adapter_opt")
        if adapter_opt is not None and getattr(trainer, "optimizer", None) is not None:
            cur_lr = float(trainer.optimizer.param_groups[0]["lr"]) * float(cfg.adapter_lr_mult)
            for pg in adapter_opt.param_groups:
                pg["lr"] = cur_lr

    def _on_fit_epoch_end(trainer: Any) -> None:
        crit = state.get("criterion")
        epoch_idx = int(getattr(trainer, "epoch", 0)) + 1
        if crit is not None:
            st = crit.flush_epoch_kd_stats()
            if st and (epoch_idx <= 2 or epoch_idx % 10 == 0):
                print(
                    f"[PHASE 3 KD][Ep {epoch_idx}] r(t)={st['r_t']:.2f} | "
                    f"M1={st['m1']:.4f} | M2={st['m2']:.4f} | M3={st['m3']:.4f} "
                    f"(batches={st['batches']}, adapter_steps={state['adapter_steps']})"
                )
        # Save exact-epoch `kd_state.pt` (~0.8MB, <5ms) alongside `last.pt` for lossless resume
        adapters = state.get("adapters")
        adapter_opt = state.get("adapter_opt")
        if adapters is not None and adapter_opt is not None:
            try:
                w_dir = Path(trainer.save_dir) / "weights"
                w_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch_idx,
                        "teacher_sig": state["teacher_sig"],
                        "adapters": adapters.state_dict(),
                        "adapter_opt": adapter_opt.state_dict(),
                    },
                    str(w_dir / "kd_state.pt"),
                )
            except Exception as exc:
                print(f"[WARN] Could not save kd_state.pt at epoch {epoch_idx}: {exc}")

    def _on_train_end(trainer: Any) -> None:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        crit = state.get("criterion")
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            try:
                peak_vram_gb = round(float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3), 3)
            except Exception:
                peak_vram_gb = 0.0

        # Persist KD health diagnostics for Gate 1 verification
        try:
            health = {
                "teacher_sig": state.get("teacher_sig", ""),
                "adapter_steps": int(state.get("adapter_steps", 0)),
                "total_kd_batches": int(getattr(crit, "_total_kd_batches", 0)) if crit is not None else 0,
                "last_epoch_avg_kd": getattr(crit, "_last_epoch_avg_kd", {}) if crit is not None else {},
                "peak_vram_gb": peak_vram_gb,
                "needs_adapters": bool((cfg.use_m1_feat and cfg.feat_kd_mode != "none") or cfg.use_m2_cross_o2o),
            }
            out_p = Path(trainer.save_dir) / "kd_health.json"
            out_p.write_text(json.dumps(health, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"[PHASE 3 KD] Health summary -> adapter_steps={health['adapter_steps']}, "
                f"kd_batches={health['total_kd_batches']}, peak_vram={peak_vram_gb:.2f} GB"
            )
        except Exception as exc:
            print(f"[WARN] Could not write kd_health.json: {exc}")

        # Clean teardown: restore native criterion & optimizer_step, release teacher/adapters VRAM
        if state.get("base_criterion") is not None:
            raw_model.criterion = state["base_criterion"]
        if state.get("orig_opt_step") is not None:
            trainer.optimizer_step = state["orig_opt_step"]
        for h in state.get("handles", []):
            try:
                h.remove()
            except Exception:
                pass
        state["handles"] = []
        state["teacher"] = None
        state["adapters"] = None
        state["adapter_opt"] = None
        state["criterion"] = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    yolo_model.add_callback("on_train_start", _on_train_start)
    yolo_model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    yolo_model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    yolo_model.add_callback("on_train_end", _on_train_end)
