# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Training data isolation and the mathematical behavior of masked losses."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

try:
    import torch
    from pvextent.data.training import TrainingDataError, TrainingDataset
    from pvextent.training.losses import masked_bce_dice
except ImportError:
    torch = None


METADATA = '''schema_version = "1.0"
dataset_id = "training-fixture"
dataset_version = "1"
[sources.synthetic]
name = "Generated test pixels"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "CC0-1.0"
attribution = "Synthetic test fixture; no real imagery"
'''


@unittest.skipIf(torch is None, "Install the train extra for training data and loss tests")
class TrainingDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        (self.root / "dataset.toml").write_text(METADATA, encoding="utf-8")
        self.records = []
        for index, (identifier, split, status) in enumerate((
            ("mixed", "train", "upstream"), ("negative", "train", "reviewed"),
            ("ignored", "train", "upstream"), ("validation", "val", "reviewed"),
            ("holdout", "test", "upstream"), ("unlabelled", "unassigned", "unreviewed"),
        )):
            record = {"id": identifier, "image": f"images/{identifier}.png", "mask": f"masks/{identifier}.png",
                      "image_sha256": "", "mask_sha256": "", "width": 8, "height": 8, "gsd_m": 0.1,
                      "source_id": "synthetic", "capture_date": None, "spatial_group": f"group-{identifier}",
                      "building_ids": [f"synthetic:{identifier}"], "split": split, "annotation_status": status,
                      "georeference": None}
            image = Image.new("RGB", (8, 8), (255 - index * 20, 128, index * 20))
            image.save(self.root / record["image"])
            mask = Image.new("L", (8, 8), 255 if identifier in ("ignored", "unlabelled") else 0)
            if identifier == "mixed":
                mask.putpixel((2, 3), 1)
                mask.putpixel((4, 1), 255)
            mask.save(self.root / record["mask"])
            self.rehash(record)
            self.records.append(record)
        self.write_manifest()

    def rehash(self, record) -> None:
        for role in ("image", "mask"):
            record[f"{role}_sha256"] = hashlib.sha256((self.root / record[role]).read_bytes()).hexdigest()

    def write_manifest(self) -> None:
        (self.root / "manifest.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in self.records), encoding="utf-8")

    def test_training_never_opens_or_resolves_val_test_pixels(self) -> None:
        forbidden = {self.root / record[role] for record in self.records
                     if record["split"] != "train" or record["id"] == "ignored"
                     for role in (("image",) if record["id"] == "ignored" else ("image", "mask"))}
        original_open, original_resolve = Path.open, Path.resolve

        def guarded_open(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError(f"Forbidden pixel access: {path}")
            return original_open(path, *args, **kwargs)

        def guarded_resolve(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError(f"Forbidden pixel resolution: {path}")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Path, "resolve", guarded_resolve):
            dataset = TrainingDataset(self.root)
            samples = [dataset[index] for index in range(len(dataset))]
        self.assertEqual({item["id"] for item in samples}, {"mixed", "negative"})
        mixed = next(item for item in samples if item["id"] == "mixed")
        self.assertEqual(mixed["image"].shape, (3, 8, 8))
        self.assertEqual(mixed["image"].dtype, torch.float32)
        self.assertEqual(mixed["target"].shape, (1, 8, 8))
        self.assertEqual(mixed["target"].dtype, torch.float32)
        self.assertEqual(mixed["valid"].dtype, torch.bool)
        self.assertEqual(mixed["image"][0, 0, 0].item(), 1)
        self.assertAlmostEqual(mixed["image"][1, 0, 0].item(), 128 / 255)
        self.assertEqual(mixed["target"][0, 3, 2].item(), 1)
        self.assertEqual(mixed["target"][0, 1, 4].item(), 0)
        self.assertFalse(mixed["valid"][0, 1, 4].item())
        self.assertEqual(mixed["valid"].sum().item(), 63)
        negative = next(item for item in samples if item["id"] == "negative")
        self.assertEqual(negative["target"].sum().item(), 0)
        self.assertEqual(negative["valid"].sum().item(), 64)

    def test_missing_val_test_files_are_not_required_for_training(self) -> None:
        for record in self.records:
            if record["split"] in ("val", "test"):
                for role in ("image", "mask"):
                    (self.root / record[role]).unlink()
        self.assertEqual(len(TrainingDataset(self.root)), 2)

    def test_selection_and_snapshot_fingerprints_are_reproducible(self) -> None:
        first = TrainingDataset(self.root, max_samples=1, seed=123)
        second = TrainingDataset(self.root, max_samples=1, seed=123)
        self.assertEqual(first.records, second.records)
        self.assertEqual(first.selection_fingerprint, second.selection_fingerprint)
        payload = (self.root / "dataset.toml").read_bytes() + b"\0" + (self.root / "manifest.jsonl").read_bytes()
        self.assertEqual(first.dataset_fingerprint, hashlib.sha256(payload).hexdigest())
        choices = {TrainingDataset(self.root, max_samples=1, seed=seed).records[0]["id"] for seed in range(12)}
        self.assertEqual(choices, {"mixed", "negative"})

    def test_unreviewed_assigned_declarations_fail_without_pixel_access(self) -> None:
        self.records[3]["annotation_status"] = "unreviewed"
        self.write_manifest()
        with patch("pvextent.data.training._raster", side_effect=AssertionError("No pixels should be opened")):
            with self.assertRaisesRegex(TrainingDataError, "unreviewed_labels"):
                TrainingDataset(self.root)

    def test_all_ignore_train_records_provide_no_eligible_supervision(self) -> None:
        for record in self.records:
            if record["split"] == "train":
                Image.new("L", (8, 8), 255).save(self.root / record["mask"])
                self.rehash(record)
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "No eligible train records"):
            TrainingDataset(self.root)

    def test_source_tampering_is_rejected_both_at_init_and_getitem(self) -> None:
        dataset = TrainingDataset(self.root)
        selected = dataset.records[0]
        path = self.root / selected["image"]
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(TrainingDataError, "hash mismatch"):
            dataset[0]
        with self.assertRaisesRegex(TrainingDataError, "hash mismatch"):
            TrainingDataset(self.root)

    def test_noncanonical_mask_encoding_and_labels_are_rejected(self) -> None:
        record = self.records[0]
        for mode, value, expected in (("I;16", 1, "8-bit"), ("L", 2, "mask labels"), ("RGB", (0, 0, 0), "single-frame L")):
            with self.subTest(mode=mode, value=value):
                Image.new(mode, (8, 8), value).save(self.root / record["mask"])
                self.rehash(record)
                self.write_manifest()
                with self.assertRaisesRegex(TrainingDataError, expected):
                    TrainingDataset(self.root)

    def test_unsafe_and_missing_train_files_are_rejected(self) -> None:
        record = self.records[0]
        original = record["image"]
        record["image"] = "../outside.png"
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "unsafe image path"):
            TrainingDataset(self.root)
        record["image"] = original
        self.write_manifest()
        (self.root / original).unlink()
        with self.assertRaisesRegex(TrainingDataError, "missing_file"):
            TrainingDataset(self.root)

    def test_cross_split_path_sharing_is_rejected_before_pixel_access(self) -> None:
        self.records[3]["image"] = self.records[0]["image"]
        self.write_manifest()
        with patch("pvextent.data.training._raster", side_effect=AssertionError("No pixels should be opened")):
            with self.assertRaisesRegex(TrainingDataError, "split_leakage"):
                TrainingDataset(self.root)

    def test_inconsistent_gsd_and_training_dimension_bounds_fail(self) -> None:
        self.records[0]["gsd_m"] = 0.2
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "consistent gsd_m"):
            TrainingDataset(self.root)
        self.records[0]["gsd_m"] = 0.1
        self.records[0]["width"] = 1025
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "dimensions exceed"):
            TrainingDataset(self.root)

    def test_strict_json_schema_is_checked_for_nontraining_declarations(self) -> None:
        self.records[4]["unexpected"] = True
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "unknown_fields"):
            TrainingDataset(self.root)
        del self.records[4]["unexpected"]
        self.write_manifest()
        with (self.root / "manifest.jsonl").open("a") as handle:
            handle.write('{"id":"duplicate", "id":"duplicate"}\n')
        with self.assertRaisesRegex(TrainingDataError, "strict UTF-8 JSON"):
            TrainingDataset(self.root)

    def test_invalid_selection_bounds_fail(self) -> None:
        for value in (0, -1, 65, True, 1.5):
            with self.subTest(max_samples=value), self.assertRaises(TrainingDataError):
                TrainingDataset(self.root, max_samples=value)


@unittest.skipIf(torch is None, "Install the train extra for masked loss tests")
class MaskedLossTests(unittest.TestCase):
    def test_only_supervised_pixels_contribute_and_ignored_gradients_are_zero(self) -> None:
        logits = torch.tensor([[[[0.0, 1000.0], [0.0, -1000.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 255.0], [0.0, -123.0]]]])
        valid = torch.tensor([[[[True, False], [True, False]]]])
        result = masked_bce_dice(logits, target, valid)
        self.assertEqual(result["valid_pixels"], 2)
        self.assertAlmostEqual(result["bce"].item(), math.log(2), places=6)
        self.assertAlmostEqual(result["dice_loss"].item(), 1 / 3, places=6)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[~valid].abs().sum().item(), 0)
        changed = target.clone()
        changed[~valid] = float("nan")
        self.assertEqual(masked_bce_dice(logits.detach(), changed, valid)["loss"].item(), result["loss"].item())

    def test_extreme_finite_logits_have_finite_loss_and_gradients(self) -> None:
        logits = torch.tensor([[[[-10000.0, 10000.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        result = masked_bce_dice(logits, target, valid)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertAlmostEqual(result["bce"].item(), 10000)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_ignore_and_malformed_supervision_fail_clearly(self) -> None:
        logits = torch.zeros((1, 1, 2, 2))
        target = torch.zeros_like(logits)
        valid = torch.ones_like(logits, dtype=torch.bool)
        with self.assertRaisesRegex(TrainingDataError, "at least one valid"):
            masked_bce_dice(logits, target, ~valid)
        for label in (2.0, float("nan"), float("inf")):
            with self.subTest(label=label), self.assertRaisesRegex(TrainingDataError, "finite binary"):
                masked_bce_dice(logits, torch.full_like(target, label), valid)
        with self.assertRaisesRegex(TrainingDataError, "finite"):
            masked_bce_dice(torch.full_like(logits, float("inf")), target, valid)
        with self.assertRaisesRegex(TrainingDataError, "shapes"):
            masked_bce_dice(logits, target[:, :, :1], valid)
        with self.assertRaisesRegex(TrainingDataError, "boolean"):
            masked_bce_dice(logits, target, valid.float())


if __name__ == "__main__":
    unittest.main()
