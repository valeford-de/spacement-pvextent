# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Lazy split isolation and full-sized deterministic segmentation views."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

try:
    import torch
    from pvextent.data.segmentation import SegmentationDataset
    from pvextent.data.training import TrainingDataError
except ImportError:
    torch = None


METADATA = '''schema_version = "1.0"
dataset_id = "segmentation-fixture"
dataset_version = "1"
[sources.synthetic]
name = "Synthetic segmentation pixels"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "CC0-1.0"
attribution = "Generated fixture, not real imagery"
'''


@unittest.skipIf(torch is None, "Install the train extra for segmentation data tests")
class SegmentationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        (self.root / "dataset.toml").write_text(METADATA, encoding="utf-8")
        self.records = []
        for index, (identifier, split) in enumerate((
            ("train-z", "train"), ("validation-a", "val"), ("train-a", "train"),
            ("heldout", "test"), ("train-b", "train"), ("pool", "unassigned"),
        )):
            image = f"images/{identifier}.png"
            mask = f"masks/{identifier}.png"
            Image.new("RGB", (32, 32), (255 - index * 10, 128, index * 10)).save(self.root / image)
            labels = Image.new("L", (32, 32), 255 if split == "unassigned" else 0)
            if identifier != "train-z" and split != "unassigned":
                labels.putpixel((2, 3), 1)
                labels.putpixel((4, 1), 255)
            labels.save(self.root / mask)
            record = {"id": identifier, "image": image, "mask": mask,
                      "image_sha256": "", "mask_sha256": "", "width": 32, "height": 32, "gsd_m": 0.1,
                      "source_id": "synthetic", "capture_date": None, "spatial_group": f"group-{identifier}",
                      "building_ids": [f"synthetic:{identifier}"], "split": split,
                      "annotation_status": "unreviewed" if split == "unassigned" else "upstream",
                      "georeference": None}
            self.rehash(record)
            self.records.append(record)
        self.write_manifest()

    def rehash(self, record) -> None:
        for role in ("image", "mask"):
            record[f"{role}_sha256"] = hashlib.sha256((self.root / record[role]).read_bytes()).hexdigest()

    def write_manifest(self) -> None:
        (self.root / "manifest.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in self.records), encoding="utf-8")

    def test_constructor_is_metadata_only_and_views_have_stable_id_order(self) -> None:
        with patch("pvextent.data.segmentation._raster", side_effect=AssertionError("No raster reads during construction")):
            train = SegmentationDataset(self.root)
            validation = SegmentationDataset(self.root, split="val")
        self.assertEqual([item["id"] for item in train.records], ["train-a", "train-b", "train-z"])
        self.assertEqual([item["id"] for item in validation.records], ["validation-a"])
        self.assertEqual(train.dimensions, (32, 32))
        self.assertEqual(train.gsd, 0.1)
        self.assertEqual(train.gsd_m, 0.1)
        self.assertEqual(train.snapshot_declarations, {"train": 3, "val": 1, "test": 1, "unassigned": 1})
        combined = (self.root / "dataset.toml").read_bytes() + b"\0" + (self.root / "manifest.jsonl").read_bytes()
        self.assertEqual(train.dataset_fingerprint, hashlib.sha256(combined).hexdigest())
        self.assertEqual(validation.dataset_fingerprint, train.dataset_fingerprint)

    def test_train_and_validation_never_open_or_resolve_other_split_pixels(self) -> None:
        original_open, original_resolve = Path.open, Path.resolve
        for split in ("train", "val"):
            forbidden = {self.root / record[role] for record in self.records if record["split"] != split
                         for role in ("image", "mask")}

            def guarded_open(path, *args, **kwargs):
                if path in forbidden:
                    raise AssertionError(f"Forbidden raster open: {path}")
                return original_open(path, *args, **kwargs)

            def guarded_resolve(path, *args, **kwargs):
                if path in forbidden:
                    raise AssertionError(f"Forbidden raster resolution: {path}")
                return original_resolve(path, *args, **kwargs)

            with self.subTest(split=split), patch.object(Path, "open", guarded_open), patch.object(Path, "resolve", guarded_resolve):
                dataset = SegmentationDataset(self.root, split=split)
                for index in range(len(dataset)):
                    sample = dataset[index]
                    self.assertEqual(sample["image"].shape, (3, 32, 32))
                    self.assertEqual(sample["image"].dtype, torch.float32)
                    self.assertTrue(bool(((sample["image"] >= 0) & (sample["image"] <= 1)).all()))
                    self.assertEqual(sample["target"].shape, (1, 32, 32))
                    self.assertEqual(sample["target"].dtype, torch.float32)
                    self.assertEqual(sample["valid"].dtype, torch.bool)
        # Missing test files cannot be accessed accidentally by either view.
        heldout = next(record for record in self.records if record["split"] == "test")
        for role in ("image", "mask"):
            (self.root / heldout[role]).unlink()
        self.assertEqual(len(SegmentationDataset(self.root)), 3)

    def test_masks_preserve_negatives_and_exclude_ignore_pixels(self) -> None:
        dataset = SegmentationDataset(self.root)
        positive, negative = dataset[0], dataset[2]
        self.assertEqual(positive["target"][0, 3, 2].item(), 1)
        self.assertFalse(positive["valid"][0, 1, 4].item())
        self.assertEqual(positive["target"][0, 1, 4].item(), 0)
        self.assertEqual(negative["target"].sum().item(), 0)
        self.assertEqual(negative["valid"].sum().item(), 32 * 32)

    def test_full_split_exceeds_smoke_cap_without_predecoding(self) -> None:
        base = deepcopy(next(record for record in self.records if record["id"] == "train-a"))
        self.records = []
        for index in range(70):
            record = deepcopy(base)
            record.update(id=f"sample-{index:03d}", spatial_group=f"group-{index}", building_ids=[f"synthetic:{index}"])
            self.records.append(record)
        self.write_manifest()
        with patch("pvextent.data.segmentation._raster", side_effect=AssertionError("Constructor must remain lazy")):
            dataset = SegmentationDataset(self.root)
        self.assertEqual(len(dataset), 70)
        self.assertEqual(dataset[69]["id"], "sample-069")
        self.assertEqual(len(SegmentationDataset(self.root, max_samples=65)), 65)

    def test_bounded_selection_is_sorted_and_reproducible(self) -> None:
        first = SegmentationDataset(self.root, max_samples=2, seed=1)
        second = SegmentationDataset(self.root, max_samples=2, seed=2)
        self.assertEqual([record["id"] for record in first.records], ["train-a", "train-b"])
        self.assertEqual(first.selection_fingerprint, second.selection_fingerprint)
        self.assertNotEqual(first.selection_fingerprint, SegmentationDataset(self.root).selection_fingerprint)

    def test_hash_and_mask_values_are_checked_lazily_at_every_access(self) -> None:
        dataset = SegmentationDataset(self.root)
        record = dataset.records[0]
        path = self.root / record["image"]
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(TrainingDataError, "hash mismatch"):
            dataset[0]
        self.rehash(record)
        Image.new("L", (32, 32), 2).save(self.root / record["mask"])
        self.rehash(record)
        with self.assertRaisesRegex(TrainingDataError, "mask labels"):
            dataset[0]

    def test_all_ignore_assigned_mask_fails_without_becoming_negative(self) -> None:
        record = next(record for record in self.records if record["id"] == "train-a")
        Image.new("L", (32, 32), 255).save(self.root / record["mask"])
        self.rehash(record)
        self.write_manifest()
        dataset = SegmentationDataset(self.root)
        with self.assertRaisesRegex(TrainingDataError, "all pixels are ignored"):
            dataset[0]

    def test_mixed_native_gsd_dimensions_and_unreviewed_assignments_fail(self) -> None:
        record = next(record for record in self.records if record["id"] == "train-a")
        record["gsd_m"] = 0.2
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "consistent gsd_m"):
            SegmentationDataset(self.root)
        record["gsd_m"] = 0.1
        record["width"] = 64
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "homogeneous dimensions"):
            SegmentationDataset(self.root)
        record["width"] = 32
        record["annotation_status"] = "unreviewed"
        self.write_manifest()
        with self.assertRaisesRegex(TrainingDataError, "unreviewed_labels"):
            SegmentationDataset(self.root)

    def test_declared_test_leakage_fails_before_any_raster_access(self) -> None:
        heldout = next(record for record in self.records if record["split"] == "test")
        heldout["spatial_group"] = "group-train-a"
        self.write_manifest()
        with patch("pvextent.data.segmentation._raster", side_effect=AssertionError("No pixels should be read")):
            with self.assertRaisesRegex(TrainingDataError, "split_leakage"):
                SegmentationDataset(self.root)

    def test_test_unassigned_and_invalid_limits_are_refused(self) -> None:
        for split in ("test", "unassigned", "Train", None, 42):
            with self.subTest(split=split), self.assertRaisesRegex(TrainingDataError, "only 'train' or 'val'"):
                SegmentationDataset(self.root, split=split)
        for limit in (0, -1, 100001, True, 1.5):
            with self.subTest(max_samples=limit), self.assertRaises(TrainingDataError):
                SegmentationDataset(self.root, max_samples=limit)


if __name__ == "__main__":
    unittest.main()
