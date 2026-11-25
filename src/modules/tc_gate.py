"""Mod A_v2: Zero-Initialized Thermal Clutter Suppression Gate (`TC-Gate`) on Neck P3/P4.

Why we pivoted from Mod A v1 (prune P5 + from-scratch P2 head) to Mod A v2 (`TC-Gate`):
- Real Phase 1 `p1_mod_a` data showed that replacing P5 with a from-scratch P2 head:
  1. Dropped pretrained weight transfer from 708/708 -> 355/732,
  2. Crashed `OtherVehicle` AP50 by -10.1% and overall Test mAP50 by -3.51%,
  3. Flooded O2O assignment with 25,600 stride-4 background anchors (Precision -6.4%,
     latency +0.86 ms, Δ_E2E degraded from 2.32% -> 2.96%).
- `TC-Gate` preserves 100% of `yolo26n.pt` (all 708/708 pretrained weights, fast 8,400 anchors,
  intact P5 for `OtherVehicle`), and is 100% distinct from `CVDI-YOLO26` (no P2->P3 transfer):
  - Directly suppresses thermal background clutter (sun-baked roofs, road glare, foliage) on
    neck outputs P3 (#16, 80x80) and P4 (#19, 40x40) right before the detached O2O head.
  - Uses local feature contrast `|x - AvgPool3x3(x)|` -> spatial saliency map `M in (0, 1)`
    modulated by a strictly zero-initialized per-channel residual gate `tanh(gate)`:
        y = x + tanh(gate) * (x * M)
  - Overhead: ~230 parameters total (<0.01%), <0.12 ms latency, 100% ONNX/TensorRT friendly,
    and yields directly visualizable spatial saliency maps `M` for paper interpretability figures.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCGateWrapper(nn.Module):
    """Wraps a neck output block (e.g. P3 #16, P4 #19) with zero-init local-contrast spatial gating."""

    def __init__(self, base_module: nn.Module, channels: int) -> None:
        super().__init__()
        self.base_module = base_module
        # Preserve Ultralytics routing metadata (.f, .i, .type)
        self.f = getattr(base_module, "f", -1)
        self.i = getattr(base_module, "i", 0)
        self.type = f"TCGateWrapper({getattr(base_module, 'type', type(base_module).__name__)})"

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True)
        # Strictly zero-initialized per-channel gate -> bit-for-bit baseline equivalence at step 0
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base_module(x)
        # Bidirectional local feature contrast vs 3x3 neighborhood
        diff = torch.abs(feat - F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1))
        mean_map = diff.mean(dim=1, keepdim=True)
        max_map = diff.amax(dim=1, keepdim=True)
        saliency = torch.sigmoid(self.spatial_conv(torch.cat([mean_map, max_map], dim=1)))
        return feat + torch.tanh(self.gate) * (feat * saliency)


def _infer_out_channels(module: nn.Module) -> int:
    """Infer output channel width from an Ultralytics neck C3k2/Conv block."""
    # Ultralytics C3k2 has .cv2.conv.out_channels; Conv has .conv.out_channels
    for attr in ("cv2", "cv3", "conv"):
        sub = getattr(module, attr, None)
        if isinstance(sub, nn.Conv2d):
            return int(sub.out_channels)
        if sub is not None and isinstance(getattr(sub, "conv", None), nn.Conv2d):
            return int(sub.conv.out_channels)
    # Fallback: last Conv2d out_channels
    last_ch = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_ch = int(m.out_channels)
    if last_ch is None:
        raise ValueError(f"Could not infer output channels for {type(module)}")
    return last_ch


def inject_tc_gate_into_model(nn_model: nn.Module, neck_indices: Iterable[int] = (16, 19)) -> List[int]:
    """Wrap neck P3/P4 output layers in-place AFTER pretrained weights are loaded."""
    seq = getattr(nn_model, "model", None)
    if not isinstance(seq, nn.Sequential):
        raise TypeError(f"Expected Ultralytics DetectionModel with .model Sequential, got {type(nn_model)}")

    wrapped: List[int] = []
    for idx in neck_indices:
        layer = seq[idx]
        if isinstance(layer, TCGateWrapper):
            wrapped.append(idx)
            continue
        ch = _infer_out_channels(layer)
        seq[idx] = TCGateWrapper(layer, channels=ch)
        wrapped.append(idx)
    print(f"[MOD A_v2 (TC-Gate)] Wrapped pretrained neck layers {wrapped} with Zero-Init Thermal Clutter Gate.")
    return wrapped
