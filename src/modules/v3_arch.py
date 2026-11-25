"""V3 Orthogonal, Pretrained-Preserving, Border-Safe Architecture Modules.

All modules strictly obey two empirical laws established in our Phase 1 v1/v2 post-mortem:
1. Preserve 100% of COCO-pretrained `yolo26n.pt` weights (`708/708` items) via in-place
   zero-initialized residual wrapping (`gate = 0` -> exact baseline equivalence at step 0).
2. Eliminate letterbox gray-border contrast artifacts (`640x640` vs native `640x512`) by
   using explicit `F.pad(..., mode='replicate')` before every spatial pooling/strip conv
   (100% native ONNX opset 17 `Pad(mode='edge')` & TensorRT FP16 compatible).

Modules:
- Pillar 2 (`BorderSafeBiTCStemWrapper`): Wraps Stage 1 `Conv(16->32, s=2)` with replicate-
  padded band-pass thermal contrast + FP32 Charbonnier smooth magnitude (+608 params, +0.024%).
  Unifies bright nocturnal hotspots and daytime thermal-inverted dark targets.
- Pillar 3 Primary (`BorderSafeCAAResWrapper`): Wraps Neck P3 (#16) & P4 (#19) with
  replicate-padded Context Anchor Attention (`AvgPool7x7` + `1x11`/`11x1` orthogonal strip
  DWConvs, adapted from PKINet CVPR 2024) + zero-init residual gate (~11.8K params, +0.46%).
  Orthogonal to Pillar 2 (long-range strip context & clutter suppression vs shallow local contrast).
- Pillar 3 Fallback (`HaarResWrapper`): Zero-param fixed 2D Haar DWT -> half-res grouped
  3x3 DWConv -> exact IWT + zero-init residual gate (+8,640 params, +0.34%).
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.tc_gate import _infer_out_channels


class BorderSafeBiTCStemWrapper(nn.Module):
    """Pillar 2: Border-safe bidirectional local thermal contrast stem (`Border-Safe BiTC-Stem`).

    Formula:
        C_bi(x) = sqrt((AvgPool3x3_rep_s2(x) - AvgPool5x5_rep_s2(x))^2 + eps^2) - eps
        y = BaseConv(x) + tanh(g) * SiLU(BN(Conv1x1(C_bi(x)))),  g_0 = 0
    """

    def __init__(self, base_conv: nn.Module, eps: float = 1e-4) -> None:
        super().__init__()
        self.base_conv = base_conv
        self.f = getattr(base_conv, "f", -1)
        self.i = getattr(base_conv, "i", 0)
        self.type = f"BorderSafeBiTCStemWrapper({getattr(base_conv, 'type', type(base_conv).__name__)})"

        conv_layer = getattr(base_conv, "conv", None)
        if not isinstance(conv_layer, nn.Conv2d):
            raise TypeError(
                f"BorderSafeBiTCStemWrapper expects an Ultralytics Conv module with .conv, got {type(base_conv)}"
            )

        c_in = int(conv_layer.in_channels)
        c_out = int(conv_layer.out_channels)
        stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else int(conv_layer.stride)
        if stride != 2:
            raise ValueError(f"BorderSafeBiTCStemWrapper expects a stride-2 Conv, got stride={stride}")

        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.eps = float(eps)

        self.proj = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_conv(x)

        # Explicit replicate padding eliminates letterbox gray-border step artifacts
        inner = F.avg_pool2d(
            F.pad(x, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=self.stride,
            padding=0,
        )
        outer = F.avg_pool2d(
            F.pad(x, (2, 2, 2, 2), mode="replicate"),
            kernel_size=5,
            stride=self.stride,
            padding=0,
        )

        # FP32 Charbonnier smooth magnitude prevents FP16 underflow of eps^2=1e-8 -> 0/0 NaN grad
        dev_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            diff = (inner - outer).float()
            eps = self.eps
            contrast = (torch.sqrt(diff * diff + (eps * eps)) - eps).to(x.dtype)

        return base_out + torch.tanh(self.gate) * self.proj(contrast)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class BorderSafeCAAResWrapper(nn.Module):
    """Pillar 3 (Primary): Border-safe Residual Context Anchor Attention (`Border-Safe CAA-Res`).

    Adapted from PKINet (CVPR 2024) with replicate padding and zero-initialized per-channel
    residual gating so pretrained Neck P3/P4 features remain untouched at step 0.

    Formula:
        F_p = SiLU(BN(Conv1x1_{C->C/r}(AvgPool7x7_rep(X))))
        U   = DWConv_{k x 1}_rep(DWConv_{1 x k}_rep(F_p))
        M   = Sigmoid(Conv1x1_{C/r->C}(U))
        Y   = X + tanh(g) * (X * M),  g_0 = 0
    """

    def __init__(
        self,
        base_module: nn.Module,
        channels: int,
        *,
        reduction: int = 4,
        pool_kernel: int = 7,
        strip_kernel: int = 11,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self.f = getattr(base_module, "f", -1)
        self.i = getattr(base_module, "i", 0)
        self.type = f"BorderSafeCAAResWrapper({getattr(base_module, 'type', type(base_module).__name__)})"

        c_mid = max(8, channels // reduction)
        self.channels = int(channels)
        self.c_mid = int(c_mid)
        self.pool_kernel = int(pool_kernel)
        self.pool_pad = self.pool_kernel // 2
        self.strip_kernel = int(strip_kernel)
        self.strip_pad = self.strip_kernel // 2

        self.conv_down = nn.Sequential(
            nn.Conv2d(channels, c_mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
        )
        self.dw_h = nn.Conv2d(
            c_mid, c_mid, kernel_size=(1, self.strip_kernel), stride=1, padding=0, groups=c_mid, bias=False
        )
        self.dw_v = nn.Conv2d(
            c_mid, c_mid, kernel_size=(self.strip_kernel, 1), stride=1, padding=0, groups=c_mid, bias=False
        )
        self.conv_up = nn.Conv2d(c_mid, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base_module(x)
        p = self.pool_pad
        k = self.strip_pad

        pooled = F.avg_pool2d(
            F.pad(feat, (p, p, p, p), mode="replicate"),
            kernel_size=self.pool_kernel,
            stride=1,
            padding=0,
        )
        u = self.conv_down(pooled)
        u = self.dw_h(F.pad(u, (k, k, 0, 0), mode="replicate"))
        u = self.dw_v(F.pad(u, (0, 0, k, k), mode="replicate"))
        attn = torch.sigmoid(self.conv_up(u))
        return feat + torch.tanh(self.gate) * (feat * attn)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class HaarResWrapper(nn.Module):
    """Pillar 3 (Fallback): Zero-Init 1-Level 2D Haar Wavelet Low/High-Freq Residual (`Haar-Res`).

    Exact orthogonal 2D Haar DWT (0 params) -> half-resolution grouped 3x3 DWConv -> exact IWT
    -> zero-initialized per-channel residual gate `tanh(g)`.
    """

    def __init__(self, base_module: nn.Module, channels: int) -> None:
        super().__init__()
        self.base_module = base_module
        self.f = getattr(base_module, "f", -1)
        self.i = getattr(base_module, "i", 0)
        self.type = f"HaarResWrapper({getattr(base_module, 'type', type(base_module).__name__)})"
        self.channels = int(channels)

        # Orthogonal 2D Haar basis (2x2), normalized by 0.5 so IWT = exact transpose
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        # Shape: (4, 1, 2, 2) -> expanded per channel at runtime or stored as (4*C, 1, 2, 2)
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        self.register_buffer("haar_weight", kernels.repeat(channels, 1, 1, 1), persistent=False)

        self.subband_conv = nn.Sequential(
            nn.Conv2d(4 * channels, 4 * channels, kernel_size=3, stride=1, padding=0, groups=4 * channels, bias=False),
            nn.BatchNorm2d(4 * channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base_module(x)
        _, c, h, w = feat.shape

        pad_h = h % 2
        pad_w = w % 2
        f_in = F.pad(feat, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else feat

        weight = self.haar_weight.to(dtype=f_in.dtype, device=f_in.device)
        # DWT: (B, C, H, W) -> (B, 4*C, H/2, W/2)
        sub = F.conv2d(f_in, weight, stride=2, padding=0, groups=c)
        # Border-safe 3x3 depthwise filtering on each frequency subband
        sub_proc = self.subband_conv(F.pad(sub, (1, 1, 1, 1), mode="replicate"))
        # Exact IWT via transposed convolution
        rec = F.conv_transpose2d(sub_proc, weight, stride=2, padding=0, groups=c)
        if pad_h or pad_w:
            rec = rec[:, :, :h, :w]

        return feat + torch.tanh(self.gate) * rec

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def inject_v3_bitc_into_model(
    nn_model: nn.Module,
    stages: Iterable[int] = (1,),
    eps: float = 1e-4,
) -> List[int]:
    """Wrap shallow stride-2 backbone layers with BorderSafeBiTCStemWrapper AFTER weight load."""
    seq = getattr(nn_model, "model", None)
    if not isinstance(seq, nn.Sequential):
        raise TypeError(f"Expected Ultralytics DetectionModel with .model Sequential, got {type(nn_model)}")

    wrapped: List[int] = []
    for idx in stages:
        layer = seq[idx]
        if isinstance(layer, BorderSafeBiTCStemWrapper):
            wrapped.append(idx)
            continue
        seq[idx] = BorderSafeBiTCStemWrapper(layer, eps=eps)
        wrapped.append(idx)
    print(f"[V3 PILLAR 2 (Border-Safe BiTC-Stem)] Wrapped pretrained backbone stages {wrapped}.")
    return wrapped


def inject_v3_neck_into_model(
    nn_model: nn.Module,
    neck_indices: Iterable[int] = (16, 19),
    *,
    variant: str = "caa",
    reduction: int = 4,
    pool_kernel: int = 7,
    strip_kernel: int = 11,
) -> List[int]:
    """Wrap neck P3/P4 blocks with BorderSafeCAAResWrapper or HaarResWrapper AFTER weight load."""
    seq = getattr(nn_model, "model", None)
    if not isinstance(seq, nn.Sequential):
        raise TypeError(f"Expected Ultralytics DetectionModel with .model Sequential, got {type(nn_model)}")

    if variant not in ("caa", "haar"):
        raise ValueError(f"Unknown V3 neck variant '{variant}'. Expected 'caa' or 'haar'.")

    wrapped: List[int] = []
    for idx in neck_indices:
        layer = seq[idx]
        if isinstance(layer, (BorderSafeCAAResWrapper, HaarResWrapper)):
            wrapped.append(idx)
            continue
        ch = _infer_out_channels(layer)
        if variant == "caa":
            seq[idx] = BorderSafeCAAResWrapper(
                layer,
                channels=ch,
                reduction=reduction,
                pool_kernel=pool_kernel,
                strip_kernel=strip_kernel,
            )
        else:
            seq[idx] = HaarResWrapper(layer, channels=ch)
        wrapped.append(idx)

    tag = "Border-Safe CAA-Res" if variant == "caa" else "Haar-Res"
    print(f"[V3 PILLAR 3 ({tag})] Wrapped pretrained neck layers {wrapped}.")
    return wrapped
