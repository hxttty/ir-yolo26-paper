"""Mod B: Zero-Initialized Gated Residual Bidirectional Thermal Contrast Downsampling (TCAD).

Key physical & engineering guarantees:
1. Bidirectional local thermal contrast:
   - Positive contrast: MaxPool(x) - AvgPool(x)  -> bright thermal hotspots (night scenes)
   - Negative contrast: AvgPool(x) - MinPool(x)  -> dark thermal-inverted targets on sun-heated
     daytime pavement/concrete (directly observed in HIT-UAV sanity_val_0.png!)
2. Zero-initialized per-channel residual gate (`tanh(g)` with `g = 0` at init):
   - Preserves 100% of COCO pretrained stem Conv weights at iteration 0 (`y == base_conv(x)`).
   - Allows plotting converged `|tanh(g)|` channel weights as direct interpretability evidence.
3. 100% standard PyTorch/ONNX/TensorRT operators (`MaxPool2d`, `AvgPool2d`, `Conv2d`, `BN`, `SiLU`).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCADWrapper(nn.Module):
    """Wraps a pretrained stride-2 Ultralytics Conv with zero-init bidirectional TCAD residual."""

    def __init__(self, base_conv: nn.Module) -> None:
        super().__init__()
        self.base_conv = base_conv

        # Preserve Ultralytics BaseModel routing attributes (.f, .i, .type)
        self.f = getattr(base_conv, "f", -1)
        self.i = getattr(base_conv, "i", 0)
        self.type = f"TCADWrapper({getattr(base_conv, 'type', type(base_conv).__name__)})"

        conv_layer = getattr(base_conv, "conv", None)
        if not isinstance(conv_layer, nn.Conv2d):
            raise TypeError(f"TCADWrapper expects an Ultralytics Conv module with .conv, got {type(base_conv)}")

        c_in = conv_layer.in_channels
        c_out = conv_layer.out_channels
        stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else int(conv_layer.stride)
        if stride != 2:
            raise ValueError(f"TCADWrapper expects a stride-2 downsampling Conv, got stride={stride}")

        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride

        # Lightweight depthwise-separable fusion over [pos_contrast, neg_contrast] (2 * c_in -> c_out)
        self.tcad_branch = nn.Sequential(
            nn.Conv2d(2 * c_in, 2 * c_in, kernel_size=3, stride=1, padding=1, groups=2 * c_in, bias=False),
            nn.Conv2d(2 * c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        # Strictly zero-initialized channel gate -> exact pretrained equivalence at step 0
        self.gate = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_conv(x)

        avg = F.avg_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        pos = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1) - avg
        # AvgPool - MinPool == AvgPool + MaxPool(-x) >= 0 (captures daytime thermal inversion!)
        neg = avg + F.max_pool2d(-x, kernel_size=3, stride=self.stride, padding=1)

        contrast = torch.cat([pos, neg], dim=1)
        delta = self.tcad_branch(contrast)
        return base_out + torch.tanh(self.gate) * delta

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def inject_tcad_into_model(nn_model: nn.Module, stages: Iterable[int] = (0, 1)) -> List[int]:
    """Wrap specified shallow stride-2 backbone layers in-place AFTER loading pretrained weights."""
    seq = getattr(nn_model, "model", None)
    if not isinstance(seq, nn.Sequential):
        raise TypeError(f"Expected Ultralytics DetectionModel with .model Sequential, got {type(nn_model)}")

    wrapped: List[int] = []
    for idx in stages:
        layer = seq[idx]
        if isinstance(layer, TCADWrapper):
            wrapped.append(idx)
            continue
        seq[idx] = TCADWrapper(layer)
        wrapped.append(idx)
    print(f"[MOD B] Wrapped pretrained backbone stages {wrapped} with Zero-Init Bidirectional TCAD.")
    return wrapped


def extract_tcad_gate_stats(nn_model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Summarize learned |tanh(gate)| values across all TCADWrapper layers for paper analysis."""
    stats: Dict[str, Dict[str, float]] = {}
    for name, module in nn_model.named_modules():
        if isinstance(module, TCADWrapper):
            g = torch.tanh(module.gate.detach().float()).view(-1)
            abs_g = g.abs()
            stats[name or f"stage_{module.i}"] = {
                "mean_abs_gate": round(float(abs_g.mean().item()), 5),
                "max_abs_gate": round(float(abs_g.max().item()), 5),
                "min_gate": round(float(g.min().item()), 5),
                "max_gate": round(float(g.max().item()), 5),
                "channels": int(g.numel()),
            }
    return stats
