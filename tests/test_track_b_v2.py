"""Track B configuration, illumination geometry, loss scaling, and training-path regressions."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch import nn

from configs.experiments import get_experiment_config
from src import modules
from src.modules.kd_yolo26 import (
    NeckChannelAdapters,
    YOLO26DistillCriterion,
    YOLO26KDConfig,
    compute_fpkd_loss,
    compute_quantile_macro_illumination_masks,
)
from src.train_runner import _config_sha256


def registered_config(experiment, *, fresh=False):
    """Exercise the public runner entry points with local stand-ins for model and dataset I/O."""
    wrapper = SimpleNamespace(add_callback=Mock())
    yolo = ModuleType("ultralytics")
    yolo.YOLO = Mock(return_value=wrapper)
    paired = ModuleType("src.modules.paired_rgbt_data")
    paired.attach_paired_rgbt_dataloader_hook = Mock()
    experiment = replace(experiment, kd_teacher_weights="ir.pt", kd_aux_teacher_weights="rgb.pt",
                         kd_aux_rgb_root="paired")
    with patch.dict(sys.modules, {"ultralytics": yolo, paired.__name__: paired}):
        if fresh:
            wrapper = modules.build_model_with_switches(experiment, Path("."))
        else:
            modules.attach_phase1_callbacks(wrapper, experiment)
    paired.attach_paired_rgbt_dataloader_hook.assert_called_once_with(wrapper, "paired")
    return wrapper._yolo26_kd_config


class ConfigTests(unittest.TestCase):
    def test_pre_edit_digests(self):
        expected = {
            "dv_ir_yolo26n": "57d8d9fa6054091124aced9b74fa7513bc2e68ca2a291e3dd818392e13b1e3e7",
            "dv_ir_teacher_s": "202b228b2e424be3687ac148a14a986e974b57668f84359b1cefb1911beeb228",
            "dv_rgb_teacher_s": "4bbef8be126a7fde5008ce0e8d06a0e25dbd92ec9735af798a20891d3de3bdc7",
            "dv_kd_track_a_s": "a083f6ed8874005744a4a3cad1cff971cba5d9f42012896200c82ce0bcf979c6",
            "dv_kd_track_b_s": "1ad340ad97e771673b19ed7537374707aa76969d2e00aa93407cee5f00ee2ea8",
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_config_sha256(get_experiment_config(name)), digest)

    def test_each_option_affects_digest(self):
        base = get_experiment_config("dv_kd_track_b_s")
        for name, value in dict(kd_cm_illum_mode="quantile_macro", kd_night_q50_thresh=0.2,
                                kd_night_macro_std_thresh=0.04, kd_cm_cls_fg_norm=True,
                                kd_cm_feat_quality_gate=True).items():
            with self.subTest(field=name):
                self.assertNotEqual(_config_sha256(replace(base, **{name: value})), _config_sha256(base))

    def test_presets_and_runner_wiring(self):
        for size in ("s", "m"):
            cfg = get_experiment_config(f"dv_kd_track_b_v2_{size}")
            base = get_experiment_config(f"dv_kd_track_b_{size}")
            expected = replace(base, exp_name=cfg.exp_name, notes=cfg.notes,
                               kd_cm_illum_mode="quantile_macro", kd_night_q50_thresh=0.15,
                               kd_night_macro_std_thresh=0.025, kd_cm_feat_quality_gate=True,
                               kd_cm_cls_fg_norm=True, kd_lambda_cm_feat=0.35, kd_lambda_cm_cls=0.50)
            self.assertEqual(cfg, expected)
            for fresh in (False, True):
                kd = registered_config(replace(cfg, kd_night_q50_thresh=0.22,
                                                kd_night_macro_std_thresh=0.037), fresh=fresh)
                self.assertEqual((kd.cm_illum_mode, kd.night_q50_thresh, kd.night_macro_std_thresh,
                                  kd.cm_feat_quality_gate, kd.cm_cls_fg_norm),
                                 ("quantile_macro", 0.22, 0.037, True, True))


PYRAMID = [(80, 80), (40, 40), (20, 20)]


def textured_day(size):
    # Smooth brightness variation plus texture. Each macro window spans several cycles.
    x = torch.arange(size).float()
    lum = 0.48 + 0.12 * (x / 17).sin()[:, None] + 0.09 * (x / 11).cos()[None, :]
    return lum.expand(1, 3, -1, -1).clone()


class IlluminationTests(unittest.TestCase):
    def test_black_letterbox_flat_and_empty(self):
        rgb = torch.zeros(3, 3, 640, 640, dtype=torch.uint8)
        rgb[0, :, :64] = rgb[0, :, -64:] = 114
        rgb[1].fill_(147)
        rgb[2].fill_(255)
        valid = torch.ones(3, 640, 640, dtype=torch.uint8) * 255
        valid[0, :64] = valid[0, -64:] = 0
        valid[2].zero_()
        for m in compute_quantile_macro_illumination_masks(rgb, PYRAMID, valid_mask=valid):
            self.assertTrue(torch.equal(m, torch.zeros_like(m)))

    def test_streetlamp_blobs(self):
        rgb = textured_day(640) * 0.06
        # 20 bright 40x40 blobs covering 7.8125% of the frame, including edge windows.
        for row in (0, 4, 8, 12):
            for col in (0, 3, 6, 9, 12):
                rgb[:, :, row*40:(row+1)*40, col*40:(col+1)*40] = 0.95
        for m in compute_quantile_macro_illumination_masks(rgb, PYRAMID):
            self.assertEqual(m.count_nonzero().item(), 0)

    def test_centered_day_letterbox(self):
        rgb = textured_day(640)
        valid = torch.ones_like(rgb[:, :1])
        valid[:, :, :64] = valid[:, :, -64:] = 0
        rgb = torch.where(valid.bool(), rgb, 114 / 255)
        for m in compute_quantile_macro_illumination_masks(rgb, PYRAMID, valid_mask=valid):
            authentic = F.adaptive_avg_pool2d(valid, m.shape[-2:]) >= 0.5
            self.assertEqual(m[~authentic].count_nonzero().item(), 0)
            # The two partially covered macro rows contain 16/40 valid pixels each.
            self.assertEqual(m[authentic].mean().item(), 0.9375)

    def test_checkerboard_mosaic(self):
        rgb = textured_day(640)
        rgb[:, :, :320, 320:] = rgb[:, :, 320:, :320] = 0.02
        for m in compute_quantile_macro_illumination_masks(rgb, PYRAMID):
            half = m.size(-1) // 2
            self.assertEqual(m.mean().item(), 0.5)
            self.assertEqual(m[:, :, :half, half:].sum().item(), 0)
            self.assertEqual(m[:, :, half:, :half].sum().item(), 0)

    def test_even_median_and_minimum_valid_count(self):
        x = torch.zeros(1, 3, 16, 16)
        v = torch.zeros_like(x[:, :1])
        v[:, :, 6:8, 6:8] = 1
        x[:, :, 6, 6:8] = 0.4  # valid samples 0, 0, 0.4, 0.4 give median 0.2
        masks = compute_quantile_macro_illumination_masks(x, [(16, 16)], valid_mask=v, q50_thresh=0.19)
        self.assertTrue(torch.equal(masks[0], v))
        v[:, :, 7, 6:8] = 0
        masks = compute_quantile_macro_illumination_masks(x, [(16, 16)], valid_mask=v, macro_std_thresh=0)
        self.assertEqual(masks[0].count_nonzero().item(), 0)


class SparseTransferTests(unittest.TestCase):
    def setUp(self):
        self.shapes = [(32, 32), (16, 16), (8, 8)]
        self.strides = [8, 16, 32]
        self.n = sum(h*w for h, w in self.shapes)
        self.student = [torch.tensor([1., 0., -1.]).view(1, 3, 1, 1).expand(1, 3, h, w)
                        .clone().requires_grad_() for h, w in self.shapes]
        aux_feats = [fs.detach().roll(1, dims=1) for fs in self.student]
        self.gt = torch.tensor([[[0., 0., 32., 32.]]])
        self.st = torch.cat([torch.full((h*w, 1), st) for (h, w), st in zip(self.shapes, self.strides)])
        boxes = self.gt / self.st
        self.aux = dict(feats=aux_feats, scores=torch.ones(1, 2, self.n),
                        boxes=boxes.transpose(1, 2))
        self.pred = dict(scores=torch.zeros_like(self.aux["scores"], requires_grad=True))
        self.fg = torch.zeros((1, self.n), dtype=torch.bool)
        # Exactly 50 disjoint, interior 3x3 receptive neighborhoods on P3.
        self.centers = [y*32+x for y in range(1, 16, 3) for x in range(1, 31, 3)]
        self.fg[:, self.centers] = True
        self.ids = torch.zeros_like(self.fg, dtype=torch.long)
        self.assign = {}
        for branch in ("o2m", "o2o"):
            self.assign.update({f"t_{branch}_fg": self.fg, f"t_{branch}_gt_idx": self.ids,
                                f"t_{branch}_labels": self.ids, f"t_{branch}_gt_px": self.gt.expand(1, self.n, 4),
                                f"t_{branch}_boxes": boxes, f"t_{branch}_scores": torch.ones(1, self.n, 2)})
        branch = SimpleNamespace(hyp=SimpleNamespace(epochs=2), bbox_decode=lambda ap, boxes: boxes)
        self.criterion = YOLO26DistillCriterion(
            SimpleNamespace(one2many=branch, one2one=branch), nn.Identity(), nn.Identity(), None,
            YOLO26KDConfig(cm_illum_mode="quantile_macro", cm_feat_quality_gate=True, cm_cls_fg_norm=True),
            expected_s_channels=[3]*3,
            aux_teacher_model=lambda img: {"one2many": self.aux, "one2one": self.aux},
            aux_adapters=NeckChannelAdapters([3]*3, [3]*3),
        )
        self.rgb = textured_day(256)
        self.valid = torch.ones_like(self.rgb[:, :1])
        self.anchors = torch.zeros(self.n, 2)

    def losses(self, active, student_fg=None):
        for branch in ("o2m", "o2o"):
            scores = self.assign[f"t_{branch}_scores"]
            scores.fill_(1.0)
            scores[:, active, 0] = -1.0
        matched = self.fg if student_fg is None else student_fg
        return self.criterion._compute_track_b_cm_kd(
            self.rgb, self.valid, self.student, self.pred, self.pred, matched, self.ids,
            matched, self.ids, self.assign, self.gt, torch.ones(1, 1, 1, dtype=torch.bool),
            self.anchors, self.st, self.strides,
            precomputed_fg_maps=[torch.ones_like(f[:, :1]) for f in self.student],
        )

    def test_no_advantage_zero_loss_and_gradient(self):
        feat, cls, safe, active = self.losses([])
        self.assertEqual((feat.item(), cls.item(), safe.item(), active.item()), (0., 0., 1., 0.))
        feat.backward()
        for f in self.student:
            self.assertTrue(torch.equal(f.grad, torch.zeros_like(f)))

    def test_one_vs_twenty_five_of_fifty(self):
        one = self.losses(self.centers[:1])
        many = self.losses(self.centers[:25])
        for i in (0, 1):
            self.assertGreater(many[i].item(), 0)
            self.assertAlmostEqual((one[i] / many[i]).item(), 1/25, places=5)
        self.assertAlmostEqual(one[3].item(), 1/50)
        self.assertAlmostEqual(many[3].item(), 25/50)
        self.criterion._kd_cfg.cm_cls_fg_norm = False
        legacy_one = self.losses(self.centers[:1])[1]
        legacy_many = self.losses(self.centers[:25])[1]
        torch.testing.assert_close(legacy_one, legacy_many)

    def test_high_confidence_poor_localization_keeps_gate_closed(self):
        self.aux["boxes"] = self.aux["boxes"] + 100
        feat, cls, _, active = self.losses(self.centers)
        self.assertEqual((feat.item(), cls.item(), active.item()), (0., 0., 0.))

    def test_naive_control_keeps_foreground_denominator(self):
        self.criterion._kd_cfg.use_dual_advantage_gate = False
        day_loss = self.losses([])[0]
        self.rgb[:, :, :, 128:] = 0.02
        half_day_loss = self.losses([])[0]
        self.assertGreater(day_loss.item(), 0)
        self.assertAlmostEqual((half_day_loss / day_loss).item(), 0.5, places=5)

    def test_dilation_offsets_and_safety(self):
        # Active P4 cell borders the dark right half. CM_F uses teacher foreground
        # even when the student has no shared classification positive.
        self.rgb[:, :, :, 128:] = 0.02
        center = 32*32 + 6*16 + 7
        self.fg[:, center] = True
        self.criterion._kd_cfg.cm_lowpass_feat = False
        feat, cls, _, _ = self.losses([center], torch.zeros_like(self.fg))
        self.assertEqual(cls.item(), 0)
        feat.backward()
        for level in (0, 2):
            self.assertEqual(self.student[level].grad.count_nonzero().item(), 0)
        support = self.student[1].grad.abs().sum(1) > 0
        expected = torch.zeros_like(support)
        expected[:, 5:8, 6:8] = True
        self.assertTrue(torch.equal(support, expected))

    def test_anchor_layout_mismatch(self):
        self.anchors = self.anchors[:-1]
        with self.assertRaisesRegex(ValueError, "anchor layout"):
            self.losses([])

    def test_fpkd_normalization_includes_sample_weights_and_cutoffs(self):
        student = [torch.tensor([[[[1.]], [[-1.]]]]).expand(2, 2, 1, 1).clone().requires_grad_()]
        fg = torch.tensor([0.25, 0.5]).view(2, 1, 1, 1)
        weights = torch.tensor([4., 6.])

        def loss(gate, mode, fg_map=fg):
            return compute_fpkd_loss(
                student, [-student[0]], self.gt, self.fg, [8], denom_mode=mode,
                precomputed_fg_maps=[fg_map], sample_weights=weights,
                spatial_gate_maps=[torch.tensor([gate, 0.]).view(2, 1, 1, 1)],
            )

        self.assertAlmostEqual(loss(0.5, "gated").item(), 2.0, places=5)
        self.assertAlmostEqual(loss(0.5, "fg_only").item(), 0.25, places=5)
        self.assertEqual(loss(0.9e-6, "fg_only").item(), 0.0)
        self.assertGreater(loss(1.1e-6, "fg_only").item(), 0.0)
        self.assertEqual(loss(1.0, "fg_only", fg*1e-5).item(), 0.0)


def grid_anchors(feats, strides, offset):
    anchors, scales = [], []
    for feat, stride in zip(feats, strides):
        h, w = feat.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        anchors.append(torch.stack((x, y), dim=-1).reshape(-1, 2).float() + offset)
        scales.append(torch.full((h*w, 1), float(stride)))
    return torch.cat(anchors), torch.cat(scales)


class ToyHead(nn.Module):
    """Emit the raw dual-head contract, including the teacher's differentiable O2O entry point."""
    one2one = {}

    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward_head(self, feats):
        scores, boxes = [], []
        for feat in feats:
            b, _, h, w = feat.shape
            logits = self.bias + feat[:, :2].flatten(2) * 0.2
            scores.append(logits)
            box = feat.new_tensor([w/4, h/4, 3*w/4, 3*h/4]).view(1, 4, 1)
            boxes.append(box.expand(b, 4, h*w))
        return dict(feats=feats, scores=torch.cat(scores, dim=2), boxes=torch.cat(boxes, dim=2))


class ToyDetector(nn.Module):
    def __init__(self, channel_vector, bias):
        super().__init__()
        self.vector = nn.Parameter(torch.tensor(channel_vector).view(1, -1, 1, 1))
        self.model = nn.ModuleList([ToyHead(bias)])

    def forward(self, img):
        lum = img.mean(1, keepdim=True)
        feats = [F.avg_pool2d(lum, s) * self.vector for s in (8, 16, 32)]
        return {"one2many": self.model[-1].forward_head(feats),
                "one2one": self.model[-1].forward_head([f.detach() for f in feats])}


class ToyBranch:
    stride = torch.tensor([8., 16., 32.])
    hyp = SimpleNamespace(epochs=2)

    @staticmethod
    def preprocess(targets, bs, scale_tensor):
        labels = targets[:, 1:2]
        xy, wh = targets[:, 2:4], targets[:, 4:6]
        boxes = torch.cat([xy-wh/2, xy+wh/2], dim=1) * scale_tensor
        return torch.cat([labels, boxes], dim=1).view(bs, 1, 5)

    @staticmethod
    def bbox_decode(anchors, boxes):
        return boxes  # ToyHead emits decoded grid-coordinate boxes.

    @staticmethod
    def assigner(scores, boxes, anchors, labels, gt, mask):
        fg = ((anchors >= gt[..., :2]) & (anchors < gt[..., 2:])).all(-1)
        ids = torch.zeros_like(fg, dtype=torch.long)
        return ids, gt.expand_as(boxes), None, fg, ids

    def get_assigned_targets_and_loss(self, pred, batch):
        b, _, n = pred["scores"].shape
        fg = torch.ones(b, n, dtype=torch.bool)
        loss = pred["scores"].square().mean().expand(3)
        return (fg, torch.zeros_like(fg, dtype=torch.long)), loss, loss.detach()


class CriterionTests(unittest.TestCase):
    def mixed_batch_pass(self, cfg):
        day = textured_day(640)
        night = day * 0.06
        night[:, :, ::40, :] = 0.95
        valid = torch.ones(2, 1, 640, 640, dtype=torch.uint8)
        valid[:, :, :64] = valid[:, :, -64:] = 0
        rgb = torch.where(valid.bool(), torch.cat([day, night]), 114/255)
        batch = {"img": day.expand(2, -1, -1, -1), "rgb_img": (rgb*255).byte(),
                 "rgb_valid_mask": valid, "batch_idx": torch.arange(2), "cls": torch.zeros(2, 1),
                 "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]).expand(2, -1)}
        student = ToyDetector([1., -1., 0.], -2.)
        primary = ToyDetector([0., 1., -1.], 0.5).eval().requires_grad_(False)
        aux = ToyDetector([-1., 0., 1.], 2.).eval().requires_grad_(False)
        primary_adapter = NeckChannelAdapters([3]*3, [3]*3)
        aux_adapter = NeckChannelAdapters([3]*3, [3]*3)
        base = SimpleNamespace(one2many=ToyBranch(), one2one=ToyBranch(), o2m=0.8, o2o=0.2)
        criterion = YOLO26DistillCriterion(base, student, primary, primary_adapter, cfg,
                                           expected_s_channels=[3]*3, aux_teacher_model=aux,
                                           aux_adapters=aux_adapter)
        tal = ModuleType("ultralytics.utils.tal")
        tal.make_anchors = grid_anchors
        with patch.dict(sys.modules, {tal.__name__: tal}):
            loss, _ = criterion(student(batch["img"]), batch)
        self.assertTrue(loss.isfinite().all())
        loss.sum().backward()
        for param in [*student.parameters(), *aux_adapter.parameters()]:
            self.assertIsNotNone(param.grad)
            self.assertTrue(param.grad.isfinite().all())
        for teacher in (primary, aux):
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        if cfg.use_m1_feat:
            for param in primary_adapter.parameters():
                self.assertIsNotNone(param.grad)
                self.assertTrue(param.grad.isfinite().all())
        self.assertGreater(sum(p.grad.abs().sum().item() for p in aux_adapter.parameters()), 0)
        return criterion.flush_epoch_kd_stats()

    def test_full_preset_and_smoke_metrics(self):
        cfg = registered_config(get_experiment_config("dv_kd_track_b_v2_s"))
        stats = self.mixed_batch_pass(cfg)
        for key in ("m1", "m2", "m3", "cm_feat", "cm_cls"):
            self.assertGreater(stats[key], 0, key)
        self.assertTrue(0.05 < stats["illum_safe_frac"] < 0.95)
        self.assertTrue(0 < stats["cm_gate_active_frac"] <= 1)

    def test_feature_only_requests_teacher_assignments(self):
        cfg = YOLO26KDConfig(cm_illum_mode="quantile_macro", cm_feat_quality_gate=True,
                             lambda_cm_cls=0, use_m1_feat=False, use_m2_cross_o2o=False,
                             use_m3_qg_head=False)
        stats = self.mixed_batch_pass(cfg)
        self.assertGreater(stats["cm_feat"], 0)
        self.assertEqual(stats["cm_cls"], 0)


if __name__ == "__main__":
    unittest.main()
