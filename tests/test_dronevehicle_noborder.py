"""Run from the project root with python -m unittest discover -s tests."""
import hashlib
import io
import subprocess
import tempfile
import unittest
import urllib.request
import zipfile
from collections import Counter
from contextlib import redirect_stdout
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src import prepare_dronevehicle as dv


class DroneVehicleNoBorderTests(unittest.TestCase):
    def test_three_split_preparation(self):
        names = ["car", "truck 0", "bus 1", "van", "freight_car 0", "feright car 1", "freight car"]
        polygon = "32.0 64.0 96.0 64.0 96.0 128.0 32.0 128.0"
        labels = "".join(f"{polygon} {name}\n" for name in names)
        splits = ("train", "val", "test")
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            rng = np.random.default_rng(9)
            for split in splits:
                for mod in ("rgb", "ir"):
                    root = raw / split / mod
                    images = root / "images"
                    annotations = root / ("labels" if split == "test" else "dota_labels")
                    images.mkdir(parents=True)
                    annotations.mkdir()
                    (root / "classes.txt").write_text("bus\ncar\n")
                    for stem, low in (("bright", 120), ("dark", 0)):
                        im = rng.integers(low, low + 24, (512, 640, 3), dtype=np.uint8)
                        self.assertTrue(cv2.imwrite(str(images / f"{stem}.jpeg"), im))
                        populated = (stem == "bright") == (mod == "rgb")
                        (annotations / f"{stem}.txt").write_text(labels if populated else "")

            with patch.object(cv2, "imwrite", side_effect=AssertionError("unexpected JPEG re-encode")):
                with redirect_stdout(io.StringIO()):
                    stats = dv.prepare_dronevehicle(raw_dir=raw, out_root=out)
            for split in splits:
                self.assertEqual(stats["splits"][split]["day_pairs"], 1)
                for mod in ("rgb", "ir"):
                    for stem in ("bright", "dark"):
                        dest = out / f"dronevehicle_{mod}"
                        source = raw / split / mod / "images" / f"{stem}.jpeg"
                        self.assertEqual(source.read_bytes(), (dest / "images" / split / f"{stem}.jpg").read_bytes())
                        populated = (stem == "bright") == (mod == "rgb")
                        rows = (dest / "labels" / split / f"{stem}.txt").read_text().splitlines()
                        ids = [0, 1, 2, 3, 4, 4, 4] if populated else []
                        self.assertEqual(rows, [f"{i} 0.100000 0.187500 0.100000 0.125000" for i in ids])

            rgb, ir, rgb_lbl, ir_lbl = dv._find_split_subdirs(raw, "val")
            record = dv._process_single_pair("bright", ir / "bright.jpeg", rgb / "bright.jpeg",
                                             ir_lbl / "bright.txt", rgb_lbl / "bright.txt",
                                             out / "i.jpg", out / "r.jpg", out / "i.txt", out / "r.txt", 50)
            pixels = cv2.imread(str(rgb / "bright.jpeg")).astype(np.float32) * (1.0 / 255.0)
            self.assertEqual((record["lum_mean"], record["lum_std"]), (float(pixels.mean()), float(pixels.std())))

            missing = ir_lbl / "dark.txt"
            missing.rename(missing.with_suffix(".xml"))
            with self.assertRaises(FileNotFoundError), redirect_stdout(io.StringIO()):
                dv.prepare_dronevehicle(raw_dir=raw, out_root=Path(tmp) / "missing")

    def test_malformed_dota(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "bad.txt"
            for line in ("1 2 3 car", "1 2 3 4 nan 6 7 8 car", "1 2 3 4 5 6 7 8 0"):
                with self.subTest(line=line), self.assertRaisesRegex(ValueError, "bad.txt:2"):
                    label.write_text("\n" + line)
                    dv._parse_dv_dota_to_hbb(label, 0, 640, 512, Counter(), Counter())

    def test_head_metadata_across_redirect(self):
        linked_digest = "ab" * 32
        storage_digest = "cd" * 32
        for linked in (True, False):
            def opener(handler):
                def follow(req, timeout):
                    headers = Message()
                    headers["Content-Length"] = "0"
                    if linked:
                        headers["X-Linked-Etag"] = f'"{linked_digest.upper()}"'
                    redirected = handler.redirect_request(req, None, 302, "Found", headers, "https://cdn.test/z")
                    self.assertEqual(redirected.get_method(), "HEAD")
                    response = MagicMock()
                    response.__enter__.return_value.headers = {"ETag": storage_digest, "Content-Length": "123"}
                    return response
                return MagicMock(open=follow)

            with self.subTest(linked=linked), patch.object(urllib.request, "build_opener", side_effect=opener):
                self.assertEqual(dv._remote_archive_metadata("https://hf.test/z"),
                                 (123, linked_digest if linked else storage_digest))

    def test_archive_retry_discards_failed_partial(self):
        def archive(payload):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as zf:
                zf.writestr("frame", payload)
            return data.getvalue()

        good, bad = archive(b"good"), archive(b"oops")
        metadata = len(good), hashlib.sha256(good).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "val.zip"
            part = dst.with_suffix(".zip.part")
            dst.write_bytes(bad)
            attempts = iter(("curl error", "bad digest", "success"))

            def download(cmd, check):
                self.assertTrue(check)
                self.assertFalse(part.exists())
                attempt = next(attempts)
                part.write_bytes(good if attempt == "success" else bad)
                if attempt == "curl error":
                    raise subprocess.CalledProcessError(18, cmd)

            with patch.object(dv, "_remote_archive_metadata", return_value=metadata), \
                 patch.object(dv.shutil, "which", return_value="curl"), \
                 patch.object(dv.time, "sleep"), patch.object(dv.subprocess, "run", side_effect=download):
                dv._download_hf_file("val.zip", dst)
            self.assertEqual(dst.read_bytes(), good)
            self.assertFalse(part.exists())
