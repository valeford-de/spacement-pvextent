# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Frozen holdout isolation when adding experimental reviewed negatives."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

try:
    import torch
    from pyproj import Transformer
    from pvextent.data import augmentation
    from pvextent.data.augmentation import TrainingAugmentationError, augment_training_snapshot
    from pvextent.data.training import TrainingDataset
except ImportError:
    torch = None
    Transformer = None


METADATA = '''schema_version = "1.0"
dataset_id = "augmentation-fixture"
dataset_version = "1"
[sources.synthetic]
name = "Generated augmentation test pixels"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "CC0-1.0"
attribution = "Synthetic test fixture; no real imagery"
'''


@unittest.skipIf(torch is None or Transformer is None, "Install the train and geo extras for augmentation tests")
class TrainingAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base = self.root / "base"
        self.negative = self.root / "negatives"
        self.output = self.root / "augmented"
        self.to_lonlat = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        for directory in (self.base, self.negative):
            for name in ("images", "masks", "provenance"):
                (directory / name).mkdir(parents=True, exist_ok=True)
            (directory / "dataset.toml").write_text(METADATA, encoding="utf-8")
        self.points = {
            "base-train": (392000.0, 5709000.0),
            "base-val": (392000.0, 5713000.0),
            "base-test": (397000.0, 5713000.0),
            "base-unassigned": (391000.0, 5709000.0),
        }
        self.base_rows = []
        for identifier, split in (("base-train", "train"), ("base-val", "val"),
                                  ("base-test", "test"), ("base-unassigned", "unassigned")):
            row = self.make_row(identifier, self.points[identifier], split=split,
                                status="unreviewed" if split == "unassigned" else "upstream")
            self.write_image(self.base, row)
            mask = Image.new("L", (32, 32), 255 if split == "unassigned" else 0)
            if split == "train":
                mask.paste(1, (2, 2, 6, 6))
            mask.save(self.base / row["mask"])
            if split in ("val", "test"):
                # Decoding either held-out raster must fail, although its declared hash is correct.
                for role in ("image", "mask"):
                    (self.base / row[role]).write_bytes(f"Intentionally undecodable {identifier} {role}".encode())
            self.rehash(self.base, row)
            self.base_rows.append(row)
        self.write_manifest(self.base, self.base_rows)
        self.write_centers(self.base, self.base_rows, self.points)

    def make_row(self, identifier, point, *, split="unassigned", status="reviewed"):
        x, y = point
        return {
            "id": identifier, "image": f"images/{identifier}.png", "mask": f"masks/{identifier}.png",
            "image_sha256": "", "mask_sha256": "", "width": 32, "height": 32, "gsd_m": 0.1,
            "source_id": "synthetic", "capture_date": "2026-01-01", "spatial_group": f"group-{identifier}",
            "building_ids": [f"synthetic:{identifier}"], "split": split, "annotation_status": status,
            "georeference": {"crs": "EPSG:25832", "transform": [0.1, 0, x - 1.6, 0, -0.1, y + 1.6]},
        }

    def write_image(self, directory, row) -> None:
        colour = tuple(hashlib.sha256(row["id"].encode()).digest()[:3])
        Image.new("RGB", (32, 32), colour).save(directory / row["image"])

    @staticmethod
    def rehash(directory, row) -> None:
        for role in ("image", "mask"):
            row[f"{role}_sha256"] = hashlib.sha256((directory / row[role]).read_bytes()).hexdigest()

    @staticmethod
    def write_manifest(directory, rows) -> None:
        (directory / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def write_centers(self, directory, rows, points) -> None:
        (directory / "provenance" / "samples.jsonl").write_text("".join(
            json.dumps({"id": row["id"], "approximate_center": list(self.to_lonlat.transform(*points[row["id"]])),
                        "center_crs": "EPSG:4326"}, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def negative_row(self, identifier="new-negative", point=(394000.0, 5709000.0), *,
                     replacement=False, status="reviewed", background=256, pv=False):
        if replacement:
            row = copy.deepcopy(next(row for row in self.base_rows if row["id"] == identifier))
            row.update(split="unassigned", annotation_status=status)
            (self.negative / row["image"]).write_bytes((self.base / row["image"]).read_bytes())
        else:
            row = self.make_row(identifier, point, status=status)
            self.write_image(self.negative, row)
        labels = [0] * background + [255] * (32 * 32 - background)
        if pv:
            labels[-1] = 1
        mask = Image.new("L", (32, 32))
        mask.putdata(labels)
        mask.save(self.negative / row["mask"])
        self.rehash(self.negative, row)
        return row

    @staticmethod
    def manifest(directory):
        return [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]

    @staticmethod
    def source_state(directory):
        return {str(path.relative_to(directory)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                                  path.stat().st_size, path.stat().st_mtime_ns)
                for path in directory.rglob("*") if path.is_file()}

    def decisions(self, output=None):
        return {row["id"]: row for row in json.loads(
            ((output or self.output) / "provenance" / "augmentation-decisions.json").read_text())}

    @contextmanager
    def guard_inherited_pixels(self, base=None):
        """Permit full validation only for the separate negative collection."""
        base = base or self.base
        records = self.manifest(base)
        base_pixels = {base / row[role] for row in records for role in ("image", "mask")}
        heldout_names = {Path(row[role]).name for row in records if row["split"] in ("val", "test")
                         for role in ("image", "mask")}
        original_open, original_image_open = Path.open, Image.open
        original_validate = augmentation.validate_dataset
        calls = []

        def forbidden(path):
            value = Path(path).absolute()
            return value in base_pixels or value.name in heldout_names

        def guarded_open(path, *args, **kwargs):
            if forbidden(path):
                raise AssertionError(f"Inherited pixels must not be opened: {path}")
            return original_open(path, *args, **kwargs)

        def guarded_image_open(fp, *args, **kwargs):
            if isinstance(fp, (str, os.PathLike)) and forbidden(fp):
                raise AssertionError(f"Held-out pixels must not be decoded: {fp}")
            return original_image_open(fp, *args, **kwargs)

        def curated_only(directory, *args, **kwargs):
            self.assertEqual(Path(directory).resolve(), self.negative)
            calls.append(Path(directory))
            return original_validate(directory, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Image, "open", guarded_image_open), \
                patch.object(augmentation, "validate_dataset", curated_only):
            yield calls

    def test_safe_addition_and_replacement_preserve_holdout_bytes_and_hardlinks(self) -> None:
        replacement = self.negative_row("base-unassigned", replacement=True)
        added = self.negative_row()
        self.write_manifest(self.negative, [replacement, added])
        before = {directory: self.source_state(directory) for directory in (self.base, self.negative)}
        with self.guard_inherited_pixels() as validations:
            report = augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
            training = TrainingDataset(self.output, max_samples=64)
            items = {training[index]["id"]: training[index] for index in range(len(training))}
        self.assertEqual(validations, [self.negative])
        self.assertEqual(set(items), {"base-train", "base-unassigned", "new-negative"})
        self.assertEqual(items["base-unassigned"]["valid"].sum().item(), 256)
        self.assertEqual(items["new-negative"]["target"].sum().item(), 0)
        self.assertEqual(items["base-train"]["target"].sum().item(), 16)
        self.assertEqual(report["reviewed_negatives_added_to_train"], 2)
        self.assertEqual(report["splits"], {"train": 3, "val": 1, "test": 1})
        self.assertTrue(report["heldout_records_unchanged"])
        self.assertFalse(report["heldout_pixels_opened"])
        output_rows = {row["id"]: row for row in self.manifest(self.output)}
        for row in self.base_rows:
            output_row = output_rows[row["id"]]
            if row["split"] in ("val", "test"):
                self.assertEqual(output_row, row)
            self.assertTrue(os.path.samefile(self.base / row["image"], self.output / output_row["image"]))
            if row["id"] != replacement["id"]:
                self.assertTrue(os.path.samefile(self.base / row["mask"], self.output / output_row["mask"]))
        self.assertTrue(os.path.samefile(self.negative / added["image"], self.output / output_rows[added["id"]]["image"]))
        for row in (replacement, added):
            target = self.output / output_rows[row["id"]]["mask"]
            self.assertFalse(os.path.samefile(self.negative / row["mask"], target))
            self.assertEqual(target.read_bytes(), (self.negative / row["mask"]).read_bytes())
        for directory, state in before.items():
            self.assertEqual(self.source_state(directory), state)
        proposals = [json.loads(line) for line in
                     (self.output / "provenance" / "split-assignments.jsonl").read_text().splitlines()]
        self.assertEqual({item["id"] for item in proposals}, set(output_rows))
        self.assertEqual(next(item["proposed_split"] for item in proposals if item["id"] == added["id"]), "train")

    def test_holdout_identity_overlap_rejected_before_any_negative_pixel_access(self) -> None:
        heldout = next(row for row in self.base_rows if row["split"] == "val")
        original = self.negative_row()
        negative_pixels = {self.negative / original[role] for role in ("image", "mask")}
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if Path(path).absolute() in negative_pixels:
                raise AssertionError(f"Metadata overlap must fail before opening negative pixels: {path}")
            return original_open(path, *args, **kwargs)

        for field in ("id", "image_sha256", "spatial_group", "building_ids"):
            with self.subTest(overlap=field):
                row = copy.deepcopy(original)
                row[field] = copy.deepcopy(heldout[field])
                self.write_manifest(self.negative, [row])
                with patch.object(Path, "open", guarded_open), \
                        patch.object(Image, "open", side_effect=AssertionError("No pixels may be decoded")):
                    with self.assertRaises(TrainingAugmentationError):
                        augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
                self.assertFalse(self.output.exists())

    def test_candidate_just_inside_one_kilometre_buffer_stays_unassigned(self) -> None:
        row = self.negative_row(point=(self.points["base-val"][0] + 999.5, self.points["base-val"][1]))
        self.write_manifest(self.negative, [row])
        with self.guard_inherited_pixels():
            report = augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2", buffer_m=1000)
        decision = self.decisions()[row["id"]]
        self.assertEqual(decision["split"], "unassigned")
        self.assertEqual(decision["exclusion"], "geographic_buffer")
        self.assertAlmostEqual(decision["nearest_heldout_center_m"], 999.5, places=3)
        self.assertEqual(report["reviewed_negatives_added_to_train"], 0)

    def test_original_validation_and_test_proposals_remain_reserved(self) -> None:
        row = self.negative_row("base-unassigned", replacement=True)
        self.write_manifest(self.negative, [row])
        for proposed_split in ("val", "test"):
            with self.subTest(proposed_split=proposed_split):
                assignments = [{"id": base["id"], "proposed_split": proposed_split if base["id"] == row["id"]
                                else base["split"]} for base in self.base_rows]
                (self.base / "provenance" / "split-assignments.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in assignments), encoding="utf-8")
                output = self.root / f"reserved-{proposed_split}"
                with self.guard_inherited_pixels():
                    augment_training_snapshot(self.base, self.negative, output, dataset_version="v2")
                decision = self.decisions(output)[row["id"]]
                self.assertEqual(decision["split"], "unassigned")
                self.assertEqual(decision["exclusion"], "reserved_geographic_block")
                for base in self.base_rows:
                    if base["split"] in ("val", "test"):
                        self.assertIn(base, self.manifest(output))

    def test_reviewed_negative_with_a_pv_pixel_is_rejected(self) -> None:
        row = self.negative_row(pv=True)
        self.write_manifest(self.negative, [row])
        with self.guard_inherited_pixels(), self.assertRaisesRegex(TrainingAugmentationError, "no PV"):
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".augmented-augmentation-*")), [])

    def test_reserved_geographic_block_survives_a_second_augmentation(self) -> None:
        row = self.negative_row("base-unassigned", replacement=True)
        self.write_manifest(self.negative, [row])
        assignments = [{"id": base["id"], "proposed_split": "val" if base["id"] == row["id"]
                        else base["split"]} for base in self.base_rows]
        (self.base / "provenance" / "split-assignments.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in assignments), encoding="utf-8")
        with self.guard_inherited_pixels():
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        second = self.root / "augmented-again"
        with self.guard_inherited_pixels(self.output):
            augment_training_snapshot(self.output, self.negative, second, dataset_version="v3")
        for output in (self.output, second):
            decision = self.decisions(output)[row["id"]]
            self.assertEqual(decision["split"], "unassigned")
            self.assertEqual(decision["exclusion"], "reserved_geographic_block")
            proposals = [json.loads(line) for line in
                         (output / "provenance" / "split-assignments.jsonl").read_text().splitlines()]
            self.assertEqual({item["id"] for item in proposals}, {base["id"] for base in self.base_rows})
            self.assertEqual(next(item["proposed_split"] for item in proposals if item["id"] == row["id"]), "val")

    def test_curated_mask_path_collision_cannot_mutate_an_inherited_holdout(self) -> None:
        row = self.negative_row()
        key = hashlib.sha256(row["id"].encode()).hexdigest()[:24]
        collision = f"masks/curated/{key}.png"
        holdout = next(base for base in self.base_rows if base["split"] == "val")
        original_path = self.base / holdout["mask"]
        collision_path = self.base / collision
        collision_path.parent.mkdir()
        original_path.rename(collision_path)
        holdout["mask"] = collision
        self.write_manifest(self.base, self.base_rows)
        self.write_manifest(self.negative, [row])
        before = {directory: self.source_state(directory) for directory in (self.base, self.negative)}
        with self.guard_inherited_pixels(), self.assertRaises(TrainingAugmentationError):
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".augmented-augmentation-*")), [])
        for directory, state in before.items():
            self.assertEqual(self.source_state(directory), state)

    def test_reviewed_negative_needs_at_least_256_background_pixels(self) -> None:
        row = self.negative_row(background=255)
        self.write_manifest(self.negative, [row])
        with self.guard_inherited_pixels(), self.assertRaisesRegex(TrainingAugmentationError, "256"):
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        self.assertFalse(self.output.exists())

    def test_unreviewed_all_ignore_candidate_is_excluded_from_training(self) -> None:
        row = self.negative_row(status="unreviewed", background=0)
        self.write_manifest(self.negative, [row])
        with self.guard_inherited_pixels():
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
            training = TrainingDataset(self.output, max_samples=64)
        self.assertEqual([record["id"] for record in training.records], ["base-train"])
        decision = self.decisions()[row["id"]]
        self.assertEqual(decision["split"], "unassigned")
        self.assertEqual(decision["exclusion"], "not_reviewed")
        self.assertEqual(decision["background_pixels"], 0)
        self.assertEqual(decision["ignore_pixels"], 1024)

    def test_existing_output_is_never_overwritten(self) -> None:
        self.write_manifest(self.negative, [self.negative_row()])
        self.output.mkdir()
        (self.output / "keep.txt").write_text("Existing publication", encoding="utf-8")
        before = self.source_state(self.output)
        with patch.object(Image, "open", side_effect=AssertionError("Existing output must fail before pixels")):
            with self.assertRaisesRegex(TrainingAugmentationError, "already exists"):
                augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        self.assertEqual(self.source_state(self.output), before)

    def test_late_bad_provenance_input_removes_staging_without_mutating_sources(self) -> None:
        self.write_manifest(self.negative, [self.negative_row()])
        (self.negative / "provenance" / "linked.txt").symlink_to(self.negative / "dataset.toml")
        before = {directory: self.source_state(directory) for directory in (self.base, self.negative)}
        with self.guard_inherited_pixels(), self.assertRaisesRegex(TrainingAugmentationError, "Provenance links"):
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".augmented-augmentation-*")), [])
        for directory, state in before.items():
            self.assertEqual(self.source_state(directory), state)

    def test_new_candidate_without_georeference_is_rejected(self) -> None:
        row = self.negative_row()
        row["georeference"] = None
        self.write_manifest(self.negative, [row])
        with self.guard_inherited_pixels(), self.assertRaisesRegex(TrainingAugmentationError, "georeference"):
            augment_training_snapshot(self.base, self.negative, self.output, dataset_version="v2")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
