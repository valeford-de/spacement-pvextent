# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Spatial splitting checks using generated pixels and synthetic coordinates."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from pvextent.data import validate_dataset
from pvextent.data.pv4ger import DatasetImportError

try:
    import pyproj
except ImportError:
    pyproj = None


@unittest.skipIf(pyproj is None, "Install the geo extra for geographic split tests")
class DataSplittingTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.data import splitting

        self.module = splitting
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "split"
        for directory in ("images", "masks", "provenance"):
            (self.source / directory).mkdir(parents=True, exist_ok=True)
        (self.source / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = "synthetic-spatial-fixture"\n'
            'dataset_version = "1"\n\n[sources.demo]\n'
            'name = "Synthetic spatial fixture"\nurl = "https://example.invalid/fixture"\n'
            'version = "1"\nimage_license = "CC0-1.0"\n'
            'annotation_license = "CC0-1.0"\nattribution = "Synthetic fixture"\n',
            encoding="utf-8",
        )
        self.to_geographic = pyproj.Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        self.to_metric = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        self.records = []
        self.centers = []
        for index in range(60):
            identifier = f"synthetic-{index:03d}"
            image = self.source / "images" / f"{identifier}.png"
            mask = self.source / "masks" / f"{identifier}.png"
            Image.new("RGB", (16, 16), (index + 1, 60, 140)).save(image)
            label = Image.new("L", (16, 16), 0)
            label.putpixel((index % 16, index // 16), 1)
            label.save(mask)
            self.records.append({
                "id": identifier,
                "image": image.relative_to(self.source).as_posix(),
                "mask": mask.relative_to(self.source).as_posix(),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "width": 16, "height": 16, "gsd_m": 0.1, "source_id": "demo",
                "capture_date": "2025-06-20", "georeference": None,
                "spatial_group": f"synthetic-area-{index:03d}", "building_ids": [],
                "split": "unassigned", "annotation_status": "upstream",
            })
            self.centers.append({
                "id": identifier,
                "source_id": f"fixture-{index:03d}",
                "upstream_split": "train",
                "center_crs": "EPSG:4326",
                "approximate_center": list(self.to_geographic.transform(
                    305000 + (index % 10) * 20000,
                    5605000 + (index // 10) * 20000,
                )),
            })
        self.write_source()

    def write_source(self) -> None:
        (self.source / "manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in self.records),
            encoding="utf-8",
        )
        self.write_centers()

    def write_centers(self) -> None:
        (self.source / "provenance" / "samples.jsonl").write_text(
            "".join(json.dumps(center, sort_keys=True) + "\n" for center in self.centers), encoding="utf-8"
        )

    def set_center(self, index: int, x: float, y: float) -> None:
        self.centers[index]["approximate_center"] = list(self.to_geographic.transform(x, y))

    def replace_mask(self, index: int, value: int) -> None:
        record = self.records[index]
        path = self.source / str(record["mask"])
        Image.new("L", (16, 16), value).save(path)
        record["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def snapshot(directory: Path) -> dict[str, bytes]:
        return {
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()
        }

    def split(self, **kwargs: object) -> dict[str, object]:
        return self.module.split_dataset(self.source, self.output, **kwargs)

    def output_records(self, directory: Path | None = None) -> list[dict[str, object]]:
        return [json.loads(line) for line in ((directory or self.output) / "manifest.jsonl").read_text().splitlines()]

    def assert_failure(self, **kwargs: object) -> None:
        before = self.snapshot(self.source)
        entries_before = {path.name for path in self.root.iterdir()}
        with self.assertRaises(DatasetImportError):
            self.split(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.snapshot(self.source), before)
        self.assertEqual({path.name for path in self.root.iterdir()}, entries_before)

    def assert_assigned_separation(self, records: list[dict[str, object]], minimum_m: float) -> None:
        coordinates = {
            center["id"]: self.to_metric.transform(*center["approximate_center"])
            for center in self.centers
        }
        for index, first in enumerate(records):
            if first["split"] == "unassigned":
                continue
            for second in records[index + 1:]:
                if second["split"] in ("unassigned", first["split"]):
                    continue
                distance = math.dist(coordinates[first["id"]], coordinates[second["id"]])
                self.assertGreaterEqual(distance + 1e-6, minimum_m, (first["id"], second["id"]))

    def test_reproducible_splits_preserve_input_pixels_and_provenance(self) -> None:
        before = self.snapshot(self.source)
        report = self.split()
        self.assertIsInstance(report, dict)
        json.dumps(report, allow_nan=False)
        self.assertEqual(self.snapshot(self.source), before)
        self.assertEqual(report["source_dataset_fingerprint"], validate_dataset(self.source).stats["dataset_fingerprint"])
        self.assertEqual(
            report["source_coordinates_sha256"],
            hashlib.sha256((self.source / "provenance" / "samples.jsonl").read_bytes()).hexdigest(),
        )
        records = self.output_records()
        self.assertEqual(len(records), 60)
        counts = Counter(record["split"] for record in records)
        for split in ("train", "val", "test"):
            self.assertGreater(counts[split], 0)
        validated = validate_dataset(self.output)
        self.assertTrue(validated.ok, validated.to_dict())
        for record in records:
            source_record = next(value for value in self.records if value["id"] == record["id"])
            for role in ("image", "mask"):
                self.assertEqual(
                    (self.output / str(record[role])).read_bytes(),
                    (self.source / str(source_record[role])).read_bytes(),
                )
            self.assertEqual(record["annotation_status"], source_record["annotation_status"])
        self.assertEqual(
            (self.output / "provenance" / "samples.jsonl").read_bytes(),
            (self.source / "provenance" / "samples.jsonl").read_bytes(),
        )
        self.assert_assigned_separation(records, 1000.0)
        second = self.root / "split-again"
        self.module.split_dataset(self.source, second)
        self.assertEqual((self.output / "manifest.jsonl").read_bytes(), (second / "manifest.jsonl").read_bytes())
        self.assertEqual(
            validated.stats["dataset_fingerprint"], validate_dataset(second).stats["dataset_fingerprint"]
        )
        changed_seed = self.root / "different-seed"
        self.module.split_dataset(self.source, changed_seed, seed=20261021)
        self.assertNotEqual(
            {record["id"]: record["split"] for record in records},
            {record["id"]: record["split"] for record in self.output_records(changed_seed)},
        )

    def test_original_import_and_validation_reports_are_frozen_verbatim_with_hashes(self) -> None:
        original_validation = validate_dataset(self.source).to_dict()
        payloads = {
            "import-report.json": b'{ "ok": true, "records": 60, "fixture": "original import" }\n',
            "validation-report.json": (json.dumps(original_validation, indent=1) + "\n").encode(),
        }
        for name, payload in payloads.items():
            (self.source / name).write_bytes(payload)
        before = self.snapshot(self.source)
        report = self.split()
        expected_hashes = {
            name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
        }
        self.assertEqual(report["source_reports_sha256"], expected_hashes)
        persisted_report = json.loads((self.output / "split-report.json").read_text())
        self.assertEqual(persisted_report["source_reports_sha256"], expected_hashes)
        for name, payload in payloads.items():
            preserved = self.output / "provenance" / ("split-input-" + name)
            self.assertEqual(preserved.read_bytes(), payload)
        frozen_validation = json.loads((self.output / "validation-report.json").read_text())
        self.assertEqual(frozen_validation["stats"]["dataset_fingerprint"], report["dataset_fingerprint"])
        self.assertNotEqual(
            frozen_validation["stats"]["dataset_fingerprint"],
            original_validation["stats"]["dataset_fingerprint"],
        )
        self.assertEqual(self.snapshot(self.source), before)
        self.assertTrue(validate_dataset(self.output).ok)

    def test_integer_configuration_overflow_is_a_clean_import_error(self) -> None:
        huge = 10 ** 1000
        for parameters in (
            {"block_size_m": huge}, {"buffer_m": huge}, {"ratios": (huge, 0.1, 0.1)},
        ):
            with self.subTest(parameter=next(iter(parameters))):
                self.assert_failure(**parameters)

    def test_related_groups_buildings_and_image_duplicates_are_joined_transitively(self) -> None:
        self.records[0]["spatial_group"] = self.records[1]["spatial_group"]
        self.records[1]["building_ids"] = ["synthetic-building"]
        self.records[2]["building_ids"] = ["synthetic-building"]
        image = self.source / str(self.records[2]["image"])
        (self.source / str(self.records[3]["image"])).write_bytes(image.read_bytes())
        self.records[3]["image_sha256"] = self.records[2]["image_sha256"]
        self.write_source()
        self.split()
        output = {record["id"]: record for record in self.output_records()}
        first_component = [output[record["id"]] for record in self.records[:4]]
        self.assertEqual(len({record["split"] for record in first_component}), 1)
        self.assertNotEqual(first_component[0]["split"], "unassigned")
        self.assertTrue(validate_dataset(self.output).ok)

    def test_buffer_excludes_whole_components_and_preserves_test_priority(self) -> None:
        self.set_center(0, 309900, 5605000)
        self.set_center(1, 310100, 5605000)
        self.set_center(2, 309800, 5605000)
        self.records[2]["spatial_group"] = self.records[0]["spatial_group"]
        self.write_source()
        before = self.snapshot(self.source)
        ids = [record["id"] for record in self.records[:3]]
        for seed, proposed_left in ((1, "train"), (49, "val")):
            with self.subTest(seed=seed, proposed_left=proposed_left):
                baseline_dir = self.root / f"buffer50-{seed}"
                buffered_dir = self.root / f"buffer1000-{seed}"
                self.module.split_dataset(self.source, baseline_dir, seed=seed, buffer_m=50)
                baseline = {record["id"]: record for record in self.output_records(baseline_dir)}
                self.assertEqual(baseline[ids[0]]["split"], proposed_left)
                self.assertEqual(baseline[ids[1]]["split"], "test")
                self.assertEqual(baseline[ids[2]]["split"], proposed_left)
                report = self.module.split_dataset(self.source, buffered_dir, seed=seed, buffer_m=1000)
                buffered = {record["id"]: record for record in self.output_records(buffered_dir)}
                self.assertEqual(buffered[ids[0]]["split"], "unassigned")
                self.assertEqual(buffered[ids[2]]["split"], "unassigned")
                self.assertEqual(buffered[ids[1]]["split"], "test")
                self.assertEqual(report["excluded_counts"]["geographic_buffer"], 2)
                self.assertGreaterEqual(min(report["minimum_cross_split_center_distance_m"].values()), 1000)
                self.assert_assigned_separation(list(buffered.values()), 1000)
                self.assertTrue(validate_dataset(buffered_dir).ok)
        self.assertEqual(self.snapshot(self.source), before)

    def test_unreviewed_unlabelled_and_ignored_samples_remain_quarantined(self) -> None:
        self.replace_mask(0, 0)
        self.records[1]["annotation_status"] = "unreviewed"
        self.replace_mask(2, 255)
        self.records[2]["annotation_status"] = "reviewed"
        self.replace_mask(3, 0)
        self.records[3]["annotation_status"] = "reviewed"
        self.write_source()
        self.split()
        output = {record["id"]: record for record in self.output_records()}
        for record in self.records[:3]:
            self.assertEqual(output[record["id"]]["split"], "unassigned")
            self.assertEqual(output[record["id"]]["annotation_status"], record["annotation_status"])
        self.assertIn(output[self.records[3]["id"]]["split"], ("train", "val", "test"))
        self.assertEqual(output[self.records[3]["id"]]["annotation_status"], "reviewed")
        self.assertTrue(validate_dataset(self.output).ok)

    def test_frozen_assigned_inputs_cannot_be_silently_reshuffled(self) -> None:
        self.records[0]["split"] = "train"
        self.write_source()
        self.assert_failure()

    def test_invalid_input_pixels_are_rejected_before_creating_output(self) -> None:
        (self.source / str(self.records[0]["image"])).write_bytes(b"corrupted pixels")
        self.assert_failure()

    def test_coordinate_provenance_must_cover_exactly_the_input_ids(self) -> None:
        original = list(self.centers)
        cases = (
            original[:-1],
            original + [original[0]],
            original + [{**original[0], "id": "unknown-sample"}],
        )
        for centers in cases:
            with self.subTest(records=len(centers)):
                self.centers = centers
                self.write_centers()
                self.assert_failure()

    def test_coordinates_require_supported_crs_finite_numbers_and_valid_bounds(self) -> None:
        original = dict(self.centers[0])
        cases = (
            {"center_crs": "EPSG:25832"},
            {"approximate_center": [181, 51]},
            {"approximate_center": [6, 91]},
            {"approximate_center": [float("nan"), 51]},
            {"approximate_center": [6, float("inf")]},
            {"approximate_center": [True, 51]},
            {"approximate_center": ["6", 51]},
            {"approximate_center": [6, 51, 0]},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                self.centers[0] = {**original, **changes}
                self.write_centers()
                self.assert_failure()

    def test_invalid_ratios_block_sizes_and_buffers_are_rejected(self) -> None:
        cases = (
            {"ratios": (0.8, 0.2)}, {"ratios": (0.8, 0.2, 0.2)},
            {"ratios": (0.8, -0.1, 0.3)}, {"ratios": (1, 0, 0)},
            {"ratios": (float("nan"), 0.1, 0.1)},
            {"block_size_m": 0}, {"block_size_m": -1}, {"block_size_m": 99},
            {"block_size_m": float("inf")}, {"block_size_m": True},
            {"buffer_m": -1}, {"buffer_m": 49}, {"buffer_m": float("nan")}, {"buffer_m": True},
            {"block_size_m": 100, "buffer_m": 100},
            {"seed": True}, {"seed": -1}, {"seed": 2 ** 32},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                self.assert_failure(**parameters)

    def test_existing_output_is_preserved(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing work", encoding="utf-8")
        with self.assertRaises(DatasetImportError):
            self.split()
        self.assertEqual(marker.read_text(), "existing work")
        self.assertEqual({path.name for path in self.output.iterdir()}, {"keep.txt"})

    def test_output_cannot_be_nested_in_the_input(self) -> None:
        nested = self.source / "new-split"
        before = self.snapshot(self.source)
        with self.assertRaises(DatasetImportError):
            self.module.split_dataset(self.source, nested)
        self.assertFalse(nested.exists())
        self.assertEqual(self.snapshot(self.source), before)

    def test_too_few_geographic_groups_fail_without_publishing_partial_dataset(self) -> None:
        for record in self.records:
            record["spatial_group"] = "all-one-component"
        self.write_source()
        self.assert_failure()


if __name__ == "__main__":
    unittest.main()
