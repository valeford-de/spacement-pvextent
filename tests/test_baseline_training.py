# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Real optimizer/resume checks with a tiny model; encoder downloads are mocked."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    class TinyBatchNormModel(torch.nn.Module):
        """Cheap segmentation model that exposes accidental validation training."""

        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Conv2d(3, 2, kernel_size=1)
            self.normalization = torch.nn.BatchNorm2d(2)
            self.head = torch.nn.Conv2d(2, 1, kernel_size=1)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.fill_(2.0)

        def forward(self, image):
            return self.head(torch.relu(self.normalization(self.features(image))))


BASELINE_AVAILABLE = torch is not None and all(
    importlib.util.find_spec(name) is not None
    for name in ("torchvision", "segmentation_models_pytorch", "timm", "safetensors")
)


@unittest.skipUnless(BASELINE_AVAILABLE, "Install the baseline extra for optimizer tests")
class BaselineTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.models import resnet_unet
        from pvextent.training import baseline

        self.module, self.models = baseline, resnet_unet
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset, self.output = self.root / "dataset", self.root / "run"
        self.weights = self.root / "weights"
        self.weights.mkdir()
        self.weights_path = self.weights / "fixture.pt"
        self.weights_path.write_bytes(b"synthetic encoder fixture; model factory is mocked")
        self.config_path = self.root / "config.toml"
        self.config = {
            "schema_version": "1.0", "model": "unet-resnet18-v1", "epochs": 2,
            "batch_size": 2, "learning_rate": 0.0001, "weight_decay": 0.0001,
            "seed": 20261020, "threshold": 0.5, "horizontal_flip": 0.5,
            "vertical_flip": 0.5, "max_train_samples": 4, "max_val_samples": 2,
        }
        self.write_config()
        for directory in ("images", "masks"):
            (self.dataset / directory).mkdir(parents=True, exist_ok=True)
        (self.dataset / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = "baseline-fixture"\ndataset_version = "1"\n'
            '[sources.synthetic]\nname = "Generated baseline test pixels"\n'
            'url = "https://example.invalid/baseline-fixture"\nversion = "1"\n'
            'image_license = "CC0-1.0"\nannotation_license = "CC0-1.0"\n'
            'attribution = "Synthetic tests; no real imagery"\n', encoding="utf-8",
        )
        self.records = []
        for index, split in enumerate(("train", "train", "train", "train", "val", "val", "test")):
            identifier = f"fixture-{index}"
            image_path = self.dataset / "images" / f"{identifier}.png"
            mask_path = self.dataset / "masks" / f"{identifier}.png"
            if split == "test":
                image_path.write_bytes(b"test imagery must never be opened")
                mask_path.write_bytes(b"test labels must never be opened")
            else:
                image = Image.new("RGB", (320, 320), (30 + index, 80 + index, 140))
                ImageDraw.Draw(image).rectangle((40, 50, 79, 89), fill=(10 + index, 30, 80))
                image.save(image_path)
                label = Image.new("L", (320, 320), 0 if split == "train" else 255)
                if split == "train":
                    ImageDraw.Draw(label).rectangle((40, 50, 79, 89), fill=1)
                    label.putpixel((0, 0), 255)
                elif index == 4:
                    # 100 TP and 100 FP if the initial model predicts all PV.
                    ImageDraw.Draw(label).rectangle((0, 0, 9, 9), fill=1)
                    ImageDraw.Draw(label).rectangle((10, 0, 19, 9), fill=0)
                else:
                    # 10 TP and 990 FP; per-image averaging gives a wrong IoU.
                    ImageDraw.Draw(label).rectangle((0, 0, 99, 9), fill=0)
                    ImageDraw.Draw(label).rectangle((0, 0, 9, 0), fill=1)
                label.save(mask_path)
            self.records.append({
                "id": identifier, "image": f"images/{identifier}.png", "mask": f"masks/{identifier}.png",
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "width": 320, "height": 320, "gsd_m": 0.1, "source_id": "synthetic",
                "capture_date": "2025-06-20", "georeference": None,
                "spatial_group": f"fixture-area-{index}", "building_ids": [],
                "split": split, "annotation_status": "upstream",
            })
        self.write_manifest()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        metadata = {**self.models.WEIGHT_SPEC, "path": str(self.weights_path)}
        self.prepare = self.stack.enter_context(patch.object(
            self.models, "prepare_encoder_weights", return_value=metadata
        ))
        self.factory = self.stack.enter_context(patch.object(
            self.models, "build_resnet_unet", side_effect=lambda *args, **kwargs: TinyBatchNormModel()
        ))

    def write_config(self, **changes) -> None:
        config = {**self.config, **changes}
        self.config_path.write_text(
            "".join(f"{key} = {json.dumps(value)}\n" for key, value in config.items()), encoding="utf-8"
        )

    def write_manifest(self) -> None:
        (self.dataset / "manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in self.records), encoding="utf-8"
        )

    def set_native_dimensions(self, width: int, height: int) -> None:
        for index, record in enumerate(self.records):
            if record["split"] == "test":
                continue
            image_path, mask_path = self.dataset / record["image"], self.dataset / record["mask"]
            Image.new("RGB", (width, height), (30 + index, 80, 140)).save(image_path)
            label = Image.new("L", (width, height), 0)
            ImageDraw.Draw(label).rectangle((2, 2, 9, 9), fill=1)
            label.save(mask_path)
            record.update(
                width=width, height=height,
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            )
        self.write_manifest()

    @staticmethod
    def snapshot(directory: Path) -> dict[str, bytes]:
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def train(self, directory: Path | None = None, **kwargs):
        parameters = {"config_path": self.config_path, "weights_dir": self.weights, "device": "cpu", **kwargs}
        return self.module.run_baseline(
            self.dataset, directory or self.output, **parameters,
        )

    def checkpoint(self, directory: Path | None = None, name="last.pt"):
        return torch.load((directory or self.output) / name, map_location="cpu", weights_only=True)

    def assert_failure(self, **kwargs) -> None:
        before = self.snapshot(self.dataset)
        entries = {path.name for path in self.root.iterdir()}
        with self.assertRaises(self.module.BaselineTrainingError):
            self.train(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.snapshot(self.dataset), before)
        self.assertEqual({path.name for path in self.root.iterdir()}, entries)

    def assert_nested_equal(self, left, right) -> None:
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_real_updates_use_train_and_val_with_global_metrics_and_never_test_pixels(self) -> None:
        before = self.snapshot(self.dataset)
        with patch.object(torch, "load", wraps=torch.load) as loads:
            report = self.train()
        self.assertTrue(loads.call_args_list)
        for call in loads.call_args_list:
            self.assertIs(call.kwargs["weights_only"], True)
        self.assertTrue(report["ok"])
        self.assertFalse(report["test_used"])
        self.assertEqual(report["train_samples"], 4)
        self.assertEqual(report["val_samples"], 2)
        self.assertEqual(report["epochs_completed"], 2)
        self.assertGreater(report["parameter_update_l2"], 0)
        self.assertTrue(report["best_reload_verified"])
        self.assertEqual(report["initial_val"]["tp"], 110)
        self.assertEqual(report["initial_val"]["fp"], 1090)
        self.assertTrue(math.isfinite(report["initial_val"]["iou"]))
        self.assertAlmostEqual(report["initial_val"]["iou"], 110 / 1200)
        self.assertAlmostEqual(report["initial_val"]["dice"], 220 / 1310)
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint["global_step"], 4)
        self.assertEqual(checkpoint["model_state_dict"]["normalization.num_batches_tracked"].item(), 4)
        self.assertEqual(self.snapshot(self.dataset), before)
        self.assertTrue((self.output / "best.pt").is_file())
        for name in ("last.pt", "best.pt"):
            path = self.output / name
            self.assertEqual(path.with_suffix(".pt.sha256").read_text().strip(), hashlib.sha256(path.read_bytes()).hexdigest())
        json.dumps(report, allow_nan=False)

    def test_epoch_boundary_resume_exactly_matches_continuous_cpu_training(self) -> None:
        continuous, prefix, resumed = self.root / "continuous", self.root / "prefix", self.root / "resumed"
        continuous_report = self.train(continuous)
        self.assertEqual(continuous_report["best_epoch"], 1)
        self.write_config(epochs=1)
        self.train(prefix)
        # The last checkpoint carries the earlier best weights; neither the old
        # best bundle nor the original encoder cache is needed to continue.
        (prefix / "best.pt").unlink()
        (prefix / "best.pt.sha256").unlink()
        self.weights_path.unlink()
        parent_bytes = self.snapshot(prefix)
        self.write_config(epochs=2)
        self.prepare.reset_mock()
        resumed_report = self.train(resumed, resume=prefix / "last.pt")
        self.assertEqual(resumed_report["start_epoch"], 1)
        self.assertEqual(resumed_report["epochs_completed"], 1)
        self.assertEqual(resumed_report["best_epoch"], 1)
        self.prepare.assert_not_called()
        self.assertEqual(self.snapshot(prefix), parent_bytes)
        first, second = self.checkpoint(continuous), self.checkpoint(resumed)
        for key in ("config", "epoch", "global_step", "model_state_dict", "optimizer_state_dict", "rng_state", "best_epoch", "best_iou"):
            self.assert_nested_equal(first[key], second[key])
        self.assert_nested_equal(self.checkpoint(continuous, "best.pt"), self.checkpoint(resumed, "best.pt"))

    def test_best_weights_bundle_is_not_a_resume_checkpoint(self) -> None:
        prefix = self.root / "prefix"
        self.write_config(epochs=1)
        self.train(prefix)
        self.write_config(epochs=2)
        self.assert_failure(resume=prefix / "best.pt")

    def test_failure_after_a_completed_epoch_retains_a_resumable_checkpoint(self) -> None:
        original_step = torch.optim.AdamW.step
        steps = 0

        def fail_in_second_epoch(optimizer, *args, **kwargs):
            nonlocal steps
            steps += 1
            if steps == 3:
                raise RuntimeError("synthetic second epoch failure")
            return original_step(optimizer, *args, **kwargs)

        before = self.snapshot(self.dataset)
        with patch.object(torch.optim.AdamW, "step", new=fail_in_second_epoch):
            with self.assertRaisesRegex(self.module.BaselineTrainingError, "retained"):
                self.train()
        self.assertFalse(self.output.exists())
        recovered = list(self.root.glob("run.incomplete-*"))
        self.assertEqual(len(recovered), 1)
        checkpoint = self.checkpoint(recovered[0])
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["global_step"], 2)
        failure = json.loads((recovered[0] / "failure-report.json").read_text())
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["resume_from"], "last.pt")
        saved = self.snapshot(recovered[0])
        report = self.train(self.root / "recovered", resume=recovered[0] / "last.pt")
        self.assertEqual(report["start_epoch"], 1)
        self.assertEqual(report["global_step"], 4)
        self.assertEqual(self.snapshot(recovered[0]), saved)
        self.assertEqual(self.snapshot(self.dataset), before)

    def test_resume_rejects_changed_configuration_and_source_snapshot(self) -> None:
        prefix = self.root / "prefix"
        self.write_config(epochs=1)
        self.train(prefix)
        for changes in ({"seed": 7}, {"learning_rate": 0.002}, {"threshold": 0.6}, {"max_train_samples": 3}):
            with self.subTest(changes=changes):
                self.write_config(epochs=2, **changes)
                self.assert_failure(resume=prefix / "last.pt")
        self.write_config(epochs=2)
        self.records[0]["capture_date"] = "2025-06-21"
        self.write_manifest()
        self.assert_failure(resume=prefix / "last.pt")

    def test_resume_checksum_failure_occurs_before_any_torch_load(self) -> None:
        prefix = self.root / "prefix"
        self.write_config(epochs=1)
        self.train(prefix)
        path = prefix / "last.pt"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        self.write_config(epochs=2)
        with patch.object(torch, "load", side_effect=AssertionError("Unverified checkpoint must not load")) as loads:
            self.assert_failure(resume=path)
        loads.assert_not_called()

    def test_existing_run_is_preserved_and_failed_optimizer_leaves_no_partial_run(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing work", encoding="utf-8")
        with self.assertRaises(self.module.BaselineTrainingError):
            self.train()
        self.assertEqual(marker.read_text(), "existing work")
        marker.unlink()
        self.output.rmdir()
        with patch.object(torch.optim.AdamW, "step", side_effect=RuntimeError("synthetic failure")):
            self.assert_failure()

    def test_strict_configuration_rejects_invalid_or_unknown_fields(self) -> None:
        for changes in (
            {"epochs": 0}, {"epochs": True}, {"batch_size": 0},
            {"learning_rate": -1}, {"threshold": 2}, {"horizontal_flip": 2},
            {"seed": -1}, {"max_train_samples": -1}, {"unexpected": "typo"},
        ):
            with self.subTest(changes=changes):
                self.write_config(**changes)
                self.assert_failure()

    def test_32_pixel_crops_reject_singleton_batches_before_weights_or_optimizer(self) -> None:
        self.set_native_dimensions(32, 32)
        before = self.snapshot(self.dataset)
        entries = {path.name for path in self.root.iterdir()}
        for batch_size, train_samples in ((1, 4), (3, 4), (2, 3)):
            with self.subTest(batch_size=batch_size, train_samples=train_samples):
                self.write_config(batch_size=batch_size, max_train_samples=train_samples)
                self.prepare.reset_mock()
                self.factory.reset_mock()
                with patch.object(torch.optim, "AdamW") as optimizer:
                    with self.assertRaises(self.module.BaselineTrainingError) as failed:
                        self.train(download_weights=True)
                self.assertIn("BatchNorm", str(failed.exception))
                self.assertIn("singleton", str(failed.exception))
                self.prepare.assert_not_called()
                self.factory.assert_not_called()
                optimizer.assert_not_called()
                self.assertFalse(self.output.exists())
                self.assertEqual(self.snapshot(self.dataset), before)
                self.assertEqual({path.name for path in self.root.iterdir()}, entries)

    def test_supported_small_crop_batches_and_native_320_singletons_still_train(self) -> None:
        for width, height, batch_size in ((32, 32, 2), (32, 64, 1), (320, 320, 1), (320, 320, 3)):
            with self.subTest(size=(width, height), batch_size=batch_size):
                self.set_native_dimensions(width, height)
                self.write_config(epochs=1, batch_size=batch_size)
                directory = self.root / f"run-{width}x{height}-batch{batch_size}"
                report = self.train(directory)
                self.assertTrue(report["ok"])
                self.assertEqual(report["global_step"], math.ceil(4 / batch_size))

    def test_encoder_cache_cannot_overlap_run_output_even_when_download_requested(self) -> None:
        before = self.snapshot(self.dataset)
        entries = {path.name for path in self.root.iterdir()}
        for cache in (self.output, self.output / "encoder-cache", self.root):
            with self.subTest(cache=cache):
                self.prepare.reset_mock()
                self.factory.reset_mock()
                with patch.object(torch.optim, "AdamW") as optimizer:
                    with self.assertRaisesRegex(self.module.BaselineTrainingError, "Encoder cache"):
                        self.train(weights_dir=cache, download_weights=True)
                self.prepare.assert_not_called()
                self.factory.assert_not_called()
                optimizer.assert_not_called()
                self.assertFalse(self.output.exists())
                self.assertEqual(self.snapshot(self.dataset), before)
                self.assertEqual({path.name for path in self.root.iterdir()}, entries)


if __name__ == "__main__":
    unittest.main()
