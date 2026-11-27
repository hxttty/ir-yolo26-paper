"""Regression coverage for the two native YOLO26n classifier widths."""
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn

from src.run_phase3_auto import verify_exported_student_purity


class _HeadFixture(nn.Module):
    def __init__(self, nc):
        super().__init__()
        width = max(64, min(nc, 100))
        for name in ("cv3", "one2one_cv3"):
            branches = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(nn.Conv2d(ch, width, 1), nn.BatchNorm2d(width)),
                    nn.Sequential(nn.Conv2d(width, width, 1), nn.BatchNorm2d(width)),
                    nn.Conv2d(width, nc, 1),
                )
                for ch in (64, 128, 256)
            )
            setattr(self, name, branches)


class _ModelFixture(nn.Module):
    def __init__(self, yaml, ch, nc, verbose):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(ch, 64, 3), _HeadFixture(nc))

    def fuse(self):
        self.model[-1].cv3 = None
        for branch in self.model[-1].one2one_cv3:
            for block in list(branch)[:-1]:
                block[0] = torch.nn.utils.fuse_conv_bn_eval(block[0].eval(), block[1].eval())
                del block[1]
        return self


class StudentPurityTest(unittest.TestCase):
    def test_checkpoint_schemas(self):
        try:
            from ultralytics.nn.tasks import DetectionModel
        except ModuleNotFoundError as exc:
            if exc.name != "ultralytics":
                raise
            DetectionModel = _ModelFixture
            tasks = ModuleType("ultralytics.nn.tasks")
            tasks.DetectionModel = DetectionModel
            dependency = patch.dict(sys.modules, {"ultralytics.nn.tasks": tasks})
        else:
            dependency = nullcontext()

        with dependency, tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "runs" / "student" / "weights"
            weights.mkdir(parents=True)

            def check(model, expected, message):
                torch.save({"model": model}, weights / "best.pt")
                ok, detail = verify_exported_student_purity(
                    "student", work_dir=Path(tmp), drive_dir=None, expected_nc=5,
                    max_params_m=10,  # Isolate schema failures from the parameter ceiling.
                )
                self.assertEqual(ok, expected, detail)
                self.assertIn(message, detail)

            for initial_nc, width in ((5, 64), (80, 80)):
                model = DetectionModel("yolo26n.yaml", ch=3, nc=initial_nc, verbose=False)
                head = model.model[-1]
                if initial_nc == 80:
                    check(model, False, "classification convolution")
                for branches in (head.cv3, head.one2one_cv3):
                    for branch in branches:
                        self.assertEqual(branch[-1].in_channels, width)
                        if initial_nc == 80:
                            branch[-1] = nn.Conv2d(width, 5, 1)
                check(model, True, "schema matched")
                check(model.fuse(), True, "fused")

            final = head.one2one_cv3[-1][-1]
            final.out_channels = 80  # Identical state_dict, inconsistent module metadata.
            check(model, False, "classification convolution")
            final.out_channels = 5
            final.register_buffer("unexpected", torch.zeros(1))
            check(model, False, "unexpected")
            del final.unexpected
            bias = final.bias
            final.bias = None
            check(model, False, "missing=")
            final.bias = bias
            final.weight = nn.Parameter(final.weight[:, :-1].clone())
            check(model, False, "shape_diff=[(")


if __name__ == "__main__":
    unittest.main()
