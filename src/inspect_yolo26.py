"""Phase 0 Step 1: Source-code & runtime audit for installed Ultralytics YOLO26.

Run this (~20 seconds on Colab L4) BEFORE writing any Phase 1 patches (Mod A/B/C1/C2).
It inspects the exact installed version of:
  - yolo26.yaml & yolo26-p2.yaml
  - ultralytics/utils/tal.py (STAL dynamic stride & TaskAlignedAssigner)
  - ultralytics/utils/loss.py (BboxLoss reg_max=1 L1 loss & E2EDetectLoss/E2ELoss)
  - ultralytics/nn/modules/head.py (Detect forward/end2end/postprocess)
  - live YOLO("yolo26n.pt") instantiation & partial weight transfer test for P2

Saves a clean markdown report to `--out-md` (Google Drive) so every subsequent
code patch matches the exact installed Ultralytics source with zero guesswork.
"""

from __future__ import annotations

import argparse
import inspect
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _safe_read(path: Path, max_lines: int = 260) -> str:
    if not path.exists():
        return f"[NOT FOUND: {path}]"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(f"{i+1:4d}: {ln}" for i, ln in enumerate(lines))
    head = "\n".join(f"{i+1:4d}: {ln}" for i, ln in enumerate(lines[:max_lines]))
    return f"{head}\n... [truncated {len(lines) - max_lines} remaining lines]"


def _safe_source(obj: object) -> str:
    try:
        src, start = inspect.getsourcelines(obj)
        return "".join(f"{start + i:4d}: {ln}" for i, ln in enumerate(src))
    except Exception as exc:
        return f"[Could not extract source: {exc}]"


def run_audit(out_md: Optional[Path] = None, test_p2_load: bool = True) -> str:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import head as head_mod
    from ultralytics.utils import loss as loss_mod
    from ultralytics.utils import tal as tal_mod

    ultra_root = Path(ultralytics.__file__).resolve().parent
    sections: List[str] = []

    sections.append("# Phase 0: Installed Ultralytics YOLO26 Source Audit")
    sections.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    sections.append(f"- Python: `{sys.version.split()[0]}` ({platform.platform()})")
    sections.append(f"- PyTorch: `{torch.__version__}` | CUDA available: `{torch.cuda.is_available()}`")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        sections.append(
            f"- GPU: `{props.name}` | VRAM: `{props.total_memory / (1024**3):.2f} GB`"
        )
    sections.append(f"- Ultralytics: `{ultralytics.__version__}` (`{ultra_root}`)")

    # 1. Model YAMLs
    sections.append("\n## 1. Installed YOLO26 YAML Configs")
    for rel in (
        "cfg/models/26/yolo26.yaml",
        "cfg/models/26/yolo26-p2.yaml",
    ):
        p = ultra_root / rel
        sections.append(f"\n### `{rel}`")
        sections.append(f"```yaml\n{_safe_read(p, max_lines=160)}\n```")

    # 2. Live YOLO26n Inspection
    sections.append("\n## 2. Live `YOLO('yolo26n.pt')` Runtime Check")
    try:
        model = YOLO("yolo26n.pt")
        detect_head = model.model.model[-1]
        n_layers, n_params, n_grads, gflops = model.model.info(verbose=False)
        sections.append(f"- `model.info()`: layers={n_layers}, params={n_params:,}, gflops={gflops:.3f}")
        sections.append(f"- Detect head class: `{type(detect_head).__name__}`")
        for attr in ("end2end", "reg_max", "nc", "nl", "stride", "max_det", "legacy"):
            if hasattr(detect_head, attr):
                val = getattr(detect_head, attr)
                if hasattr(val, "tolist"):
                    val = val.tolist()
                sections.append(f"- `detect.{attr}` = `{val}`")
        sections.append(f"- `model.model.criterion` initialized via: `{getattr(model.model, 'init_criterion', None)}`")
        if hasattr(model.model, "init_criterion"):
            sections.append(f"```python\n{_safe_source(model.model.init_criterion)}\n```")
    except Exception as exc:
        sections.append(f"- [ERROR instantiating yolo26n.pt]: `{exc}`")

    # 3. Partial weight loading test for official yolo26n-p2.yaml
    if test_p2_load:
        sections.append("\n## 3. Partial Weight Load Check (`yolo26n-p2.yaml` <- `yolo26n.pt`)")
        try:
            p2_model = YOLO("yolo26n-p2.yaml").load("yolo26n.pt")
            p2_head = p2_model.model.model[-1]
            n_layers, n_params, _, gflops = p2_model.model.info(verbose=False)
            sections.append(f"- `yolo26n-p2` info: layers={n_layers}, params={n_params:,}, gflops={gflops:.3f}")
            sections.append(f"- `p2_head.stride` = `{getattr(p2_head, 'stride', None)}`")
            sections.append(f"- `p2_head.reg_max` = `{getattr(p2_head, 'reg_max', None)}`")
            sections.append(f"- `p2_head.end2end` = `{getattr(p2_head, 'end2end', None)}`")
        except Exception as exc:
            sections.append(f"- [WARN] Partial P2 load test raised: `{exc}`")

    # 4. tal.py key methods (STAL + TaskAlignedAssigner)
    sections.append("\n## 4. `ultralytics/utils/tal.py` — `TaskAlignedAssigner`")
    taa = getattr(tal_mod, "TaskAlignedAssigner", None)
    if taa is not None:
        for method_name in (
            "__init__",
            "select_candidates_in_gts",
            "get_box_metrics",
            "iou_calculation",
            "select_highest_overlaps",
            "get_targets",
        ):
            fn = getattr(taa, method_name, None)
            if fn is not None:
                sections.append(f"\n### `TaskAlignedAssigner.{method_name}`")
                sections.append(f"```python\n{_safe_source(fn)}\n```")
    else:
        sections.append("[TaskAlignedAssigner not found in tal.py]")

    # 5. loss.py key classes
    sections.append("\n## 5. `ultralytics/utils/loss.py` — Loss Classes")
    for cls_name in ("BboxLoss", "v8DetectionLoss", "E2EDetectLoss", "E2ELoss"):
        cls = getattr(loss_mod, cls_name, None)
        if cls is not None:
            sections.append(f"\n### `{cls_name}`")
            sections.append(f"```python\n{_safe_source(cls)}\n```")

    # 6. head.py Detect methods
    sections.append("\n## 6. `ultralytics/nn/modules/head.py` — `Detect` Key Methods")
    detect_cls = getattr(head_mod, "Detect", None)
    if detect_cls is not None:
        for method_name in ("__init__", "forward", "forward_end2end", "postprocess"):
            fn = getattr(detect_cls, method_name, None)
            if fn is not None:
                sections.append(f"\n### `Detect.{method_name}`")
                sections.append(f"```python\n{_safe_source(fn)}\n```")

    report = "\n".join(sections)
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        print(f"[OK] Saved Phase 0 YOLO26 source audit report to: {out_md}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect installed Ultralytics YOLO26 internals.")
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("phase0_yolo26_source_audit.md"),
        help="Path to write markdown audit report.",
    )
    parser.add_argument("--no-p2-test", action="store_true", help="Skip yolo26n-p2 partial weight test.")
    args = parser.parse_args()

    report = run_audit(out_md=args.out_md, test_p2_load=not args.no_p2_test)
    # Print concise summary to stdout (full source is saved in the markdown file)
    print("\n" + "=" * 68)
    print("PHASE 0 AUDIT SUMMARY (first 65 lines — full dump saved to --out-md)")
    print("=" * 68)
    print("\n".join(report.splitlines()[:65]))
    print("=" * 68)


if __name__ == "__main__":
    main()
