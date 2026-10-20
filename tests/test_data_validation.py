# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Data-contract checks using synthetic, temporary image/mask pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib
from unittest.mock import patch

from PIL import Image

from pvextent.data import validate_dataset
from pvextent.data import validation


class DataValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dataset"
        self.root.mkdir()
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        (self.root / "dataset.toml").write_text(
            'schema_version = "1.0"\n'
            'dataset_id = "synthetic-review"\n'
            'dataset_version = "1"\n'
            '\n[sources.demo]\n'
            'name = "Synthetic test images"\n'
            'url = "https://example.invalid/synthetic"\n'
            'version = "1"\n'
            'image_license = "CC0-1.0"\n'
            'annotation_license = "CC0-1.0"\n'
            'attribution = "Synthetic test fixture"\n',
            encoding="utf-8",
        )
        self.sequence = 0

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(self, **changes: object) -> dict[str, object]:
        self.sequence += 1
        identifier = f"sample-{self.sequence}"
        image_path = self.root / "images" / f"{identifier}.png"
        mask_path = self.root / "masks" / f"{identifier}.png"
        Image.new("RGB", (8, 8), (self.sequence, 60, 140)).save(image_path)
        mask = Image.new("L", (8, 8), 0)
        mask.putpixel((3, 3), 1)
        mask.putpixel((4, 3), 255)
        mask.save(mask_path)
        result: dict[str, object] = {
            "id": identifier,
            "image": image_path.relative_to(self.root).as_posix(),
            "mask": mask_path.relative_to(self.root).as_posix(),
            "image_sha256": self.digest(image_path),
            "mask_sha256": self.digest(mask_path),
            "width": 8,
            "height": 8,
            "gsd_m": 0.1,
            "source_id": "demo",
            "capture_date": "2025-06-20",
            "spatial_group": f"area-{self.sequence}",
            "building_ids": [f"building-{self.sequence}"],
            "split": "unassigned",
            "annotation_status": "reviewed",
            "georeference": {
                "crs": "EPSG:25832",
                "transform": [0.1, 0, 300000, 0, -0.1, 5700000],
            },
        }
        result.update(changes)
        return result

    def write(self, *records: dict[str, object]) -> None:
        text = "".join(json.dumps(record) + "\n" for record in records)
        (self.root / "manifest.jsonl").write_text(text, encoding="utf-8")

    def replace_mask(
        self, record: dict[str, object], *, mode: str = "L", value: object = 0,
        size: tuple[int, int] = (8, 8),
    ) -> None:
        path = self.root / str(record["mask"])
        Image.new(mode, size, value).save(path)
        record["mask_sha256"] = self.digest(path)

    def assert_invalid(self, *records: dict[str, object]) -> None:
        self.write(*records)
        report = validate_dataset(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(report.errors)

    def test_pv_background_and_ignore_pixels_are_valid(self) -> None:
        self.write(self.record())
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())
        self.assertFalse(report.errors)

    def test_reviewed_negative_masks_may_have_identical_hashes(self) -> None:
        first, second = self.record(), self.record()
        self.replace_mask(first, value=0)
        self.replace_mask(second, value=0)
        self.assertEqual(first["mask_sha256"], second["mask_sha256"])
        self.write(first, second)
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())

    def test_unknown_capture_date_georeference_and_buildings_are_explicit(self) -> None:
        self.write(self.record(capture_date=None, georeference=None, building_ids=[]))
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())
        self.assertTrue(report.warnings)

    def test_unsupported_label_is_not_silently_normalized(self) -> None:
        record = self.record()
        self.replace_mask(record, value=2)
        self.assert_invalid(record)

    def test_palette_rgb_and_sixteen_bit_masks_are_rejected(self) -> None:
        for mode, value in (("P", 1), ("RGB", (1, 1, 1)), ("I;16", 1)):
            with self.subTest(mode=mode):
                record = self.record()
                self.replace_mask(record, mode=mode, value=value)
                self.assert_invalid(record)

    def test_input_must_really_be_rgb(self) -> None:
        for mode, value in (("L", 20), ("RGBA", (20, 60, 140, 255))):
            with self.subTest(mode=mode):
                record = self.record()
                image_path = self.root / str(record["image"])
                Image.new(mode, (8, 8), value).save(image_path)
                record["image_sha256"] = self.digest(image_path)
                self.assert_invalid(record)

    def test_encoded_png_depth_must_be_eight_bits_even_if_pillow_expands_it(self) -> None:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))

        for role, depth, color_type, row_bytes in (
            ("image", 16, 2, b'\0' * (8 * 3 * 2)),
            ("mask", 4, 0, b'\0' * 4),
        ):
            with self.subTest(role=role, depth=depth):
                record = self.record()
                path = self.root / str(record[role])
                header = struct.pack('>IIBBBBB', 8, 8, depth, color_type, 0, 0, 0)
                path.write_bytes(
                    b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
                    + chunk(b'IDAT', zlib.compress((b'\0' + row_bytes) * 8))
                    + chunk(b'IEND', b'')
                )
                record[f"{role}_sha256"] = self.digest(path)
                self.assert_invalid(record)

    def test_jpeg_rotation_requires_explicit_preprocessing(self) -> None:
        record = self.record()
        path = self.root / "images" / "rotated.jpg"
        image = Image.new("RGB", (8, 8), (20, 60, 140))
        exif = image.getexif()
        exif[274] = 1
        image.save(path, exif=exif)
        record["image"] = path.relative_to(self.root).as_posix()
        record["image_sha256"] = self.digest(path)
        self.write(record)
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())
        exif[274] = 6
        image.save(path, exif=exif)
        record["image_sha256"] = self.digest(path)
        self.assert_invalid(record)

    def test_declared_and_actual_dimensions_must_match(self) -> None:
        self.assert_invalid(self.record(width=7))
        record = self.record()
        self.replace_mask(record, size=(9, 8))
        self.assert_invalid(record)

    def test_both_image_and_mask_hashes_are_verified(self) -> None:
        for field in ("image_sha256", "mask_sha256"):
            with self.subTest(field=field):
                record = self.record()
                original = str(record[field])
                record[field] = ("0" if original[0] != "0" else "1") + original[1:]
                self.assert_invalid(record)

    def test_invalid_metadata_and_unknown_source_are_reported(self) -> None:
        self.assert_invalid(self.record(source_id="missing-source"))
        self.write(self.record())
        (self.root / "dataset.toml").write_text('[sources.demo\n', encoding="utf-8")
        report = validate_dataset(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(report.errors)

    def test_invalid_json_and_non_utf8_manifest_are_reported(self) -> None:
        self.write(self.record())
        manifest = self.root / "manifest.jsonl"
        for data in (b'{"id":\n', b'\xff\xfe\n'):
            with self.subTest(data=data):
                manifest.write_bytes(data)
                report = validate_dataset(self.root)
                self.assertFalse(report.ok)
                self.assertTrue(report.errors)

    def test_numeric_values_reject_booleans_nonfinite_and_nonpositive_values(self) -> None:
        for field, value in (
            ("width", True), ("height", False), ("gsd_m", True),
            ("gsd_m", float("nan")), ("gsd_m", float("inf")),
            ("gsd_m", 0), ("gsd_m", -0.1),
        ):
            with self.subTest(field=field, value=value):
                self.assert_invalid(self.record(**{field: value}))

    def test_invalid_container_types_become_errors_instead_of_exceptions(self) -> None:
        for field, value in (
            ("id", []), ("image", {}), ("mask", []),
            ("image_sha256", {}), ("mask_sha256", []),
            ("width", []), ("height", {}), ("gsd_m", {}),
            ("source_id", []), ("capture_date", {}),
            ("spatial_group", []), ("building_ids", [{}]),
            ("split", []), ("annotation_status", {}),
            ("georeference", []),
        ):
            with self.subTest(field=field):
                self.assert_invalid(self.record(**{field: value}))

    def test_overflowed_numbers_and_duplicate_json_keys_are_rejected(self) -> None:
        record = self.record()
        encoded = json.dumps(record)
        invalid_lines = (
            encoded.replace('"gsd_m": 0.1', '"gsd_m": 1e10000'),
            encoded.replace('"id": "sample-1"', '"id": "sample-1", "id": "another"'),
            encoded.replace('"crs": "EPSG:25832"', '"crs": "EPSG:25832", "crs": "EPSG:4326"'),
        )
        for line in invalid_lines:
            with self.subTest(line=line):
                (self.root / "manifest.jsonl").write_text(line + "\n", encoding="utf-8")
                report = validate_dataset(self.root)
                self.assertFalse(report.ok)
                self.assertTrue(report.errors)

    def test_deeply_nested_metadata_is_reported_without_crashing(self) -> None:
        self.write(self.record())
        nested = "unexpected = " + "{nested=" * 1200 + "1" + "}" * 1200
        (self.root / "dataset.toml").write_text(nested + "\n", encoding="utf-8")
        report = validate_dataset(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(report.errors)

    def test_deeply_nested_json_is_reported_without_crashing(self) -> None:
        self.write(self.record())
        nested = "[" * 1200 + "0" + "]" * 1200
        (self.root / "manifest.jsonl").write_text(nested + "\n", encoding="utf-8")
        report = validate_dataset(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(report.errors)

    def test_corrupted_image_and_mask_bytes_are_reported(self) -> None:
        for role in ("image", "mask"):
            with self.subTest(role=role):
                record = self.record()
                path = self.root / str(record[role])
                path.write_bytes(b"This is not an encoded image.")
                record[f"{role}_sha256"] = self.digest(path)
                self.assert_invalid(record)

    def test_configured_resource_limits_are_enforced(self) -> None:
        for constant, limit in (
            ("MAX_METADATA_BYTES", 1), ("MAX_MANIFEST_BYTES", 1),
            ("MAX_LINE_BYTES", 1), ("MAX_RECORDS", 0),
            ("MAX_FILE_BYTES", 1), ("MAX_PIXELS", 63),
        ):
            with self.subTest(constant=constant):
                self.write(self.record())
                with patch.object(validation, constant, limit):
                    report = validate_dataset(self.root)
                self.assertFalse(report.ok)
                self.assertTrue(report.errors)

    def test_actual_pixel_count_is_bounded_before_dimension_mismatch(self) -> None:
        self.write(self.record(width=7))
        with patch.object(validation, "MAX_PIXELS", 63):
            report = validate_dataset(self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any(issue.code == "pixel_limit" for issue in report.errors))

    def test_error_limit_stops_before_processing_remaining_records(self) -> None:
        self.write(self.record(gsd_m=0), self.record(gsd_m=0))
        with patch.object(validation, "MAX_ISSUES", 1):
            report = validate_dataset(self.root)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 1)
        self.assertTrue(report.stats["stopped_early"])

    def test_capture_date_requires_real_calendar_date_and_iso_format(self) -> None:
        for value in ("2025-02-30", "20250620", "20.06.2025", "2025-06-20T12:00:00Z"):
            with self.subTest(value=value):
                self.assert_invalid(self.record(capture_date=value))

    def test_paths_cannot_escape_dataset_root(self) -> None:
        record = self.record()
        outside = self.root.parent / "outside.png"
        outside.write_bytes((self.root / str(record["image"])).read_bytes())
        for path in ("../outside.png", str(outside)):
            with self.subTest(path=path):
                self.assert_invalid({**record, "image": path})

    def test_symlinks_cannot_escape_dataset_root(self) -> None:
        record = self.record()
        outside = self.root.parent / "outside.png"
        outside.write_bytes((self.root / str(record["image"])).read_bytes())
        link = self.root / "images" / "linked.png"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError) as error:
            self.skipTest(str(error))
        record["image"] = "images/linked.png"
        self.assert_invalid(record)

    def test_all_ignore_mask_is_valid_only_while_unassigned(self) -> None:
        record = self.record()
        self.replace_mask(record, value=255)
        self.write(record)
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())
        self.assertTrue(report.warnings)
        self.assert_invalid({**record, "split": "train"})

    def test_ids_must_be_unique(self) -> None:
        self.assert_invalid(self.record(), self.record(id="sample-1"))

    def test_identical_images_cannot_cross_assigned_splits(self) -> None:
        first = self.record(split="train")
        second = self.record(split="test")
        (self.root / str(second["image"])).write_bytes(
            (self.root / str(first["image"])).read_bytes()
        )
        second["image_sha256"] = first["image_sha256"]
        self.assert_invalid(first, second)

    def test_identical_images_in_one_split_are_visible_as_warning(self) -> None:
        first = self.record(split="train")
        second = self.record(split="train")
        (self.root / str(second["image"])).write_bytes(
            (self.root / str(first["image"])).read_bytes()
        )
        second["image_sha256"] = first["image_sha256"]
        self.write(first, second)
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())
        self.assertTrue(report.warnings)

    def test_geographic_groups_cannot_cross_assigned_splits(self) -> None:
        self.assert_invalid(
            self.record(split="train", spatial_group="shared-area"),
            self.record(split="val", spatial_group="shared-area"),
        )

    def test_repeat_buildings_cannot_cross_assigned_splits(self) -> None:
        self.assert_invalid(
            self.record(split="train", building_ids=["shared-building"]),
            self.record(split="test", building_ids=["shared-building"]),
        )

    def test_unassigned_records_do_not_create_false_split_leakage(self) -> None:
        self.write(
            self.record(split="train", spatial_group="shared-area", building_ids=["same"]),
            self.record(spatial_group="shared-area", building_ids=["same"]),
        )
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())

    def test_upstream_labels_are_distinguished_from_unreviewed_labels(self) -> None:
        self.write(self.record(split="train", annotation_status="upstream"))
        report = validate_dataset(self.root)
        self.assertTrue(report.ok, report.to_dict())
        self.assertTrue(report.warnings)
        self.assert_invalid(self.record(split="train", annotation_status="unreviewed"))

    def test_georeference_rejects_degenerate_and_nonfinite_transforms(self) -> None:
        for transform in (
            [0, 0, 300000, 0, 0, 5700000],
            [0.1, 0, 300000, 0, -0.1],
            [0.1, 0, float("nan"), 0, -0.1, 5700000],
            [1e308, 1e308, 0, 1e308, -1e308, 0],
        ):
            with self.subTest(transform=transform):
                self.assert_invalid(self.record(georeference={"crs": "EPSG:25832", "transform": transform}))

    def test_empty_dataset_is_not_reported_as_success(self) -> None:
        self.assert_invalid()

    def test_unknown_record_fields_and_duplicate_building_ids_are_rejected(self) -> None:
        self.assert_invalid(self.record(unexpected_field="probably a misspelling"))
        self.assert_invalid(self.record(building_ids=["same", "same"]))

    def test_report_is_machine_readable_and_json_serializable(self) -> None:
        self.write(self.record(capture_date=None, georeference=None))
        report = validate_dataset(self.root)
        document = report.to_dict()
        self.assertTrue({"ok", "errors", "warnings", "stats"}.issubset(document))
        self.assertTrue(document["ok"])
        self.assertIsInstance(document["errors"], list)
        self.assertIsInstance(document["warnings"], list)
        self.assertEqual(json.loads(json.dumps(document, allow_nan=False))["ok"], report.ok)


if __name__ == "__main__":
    unittest.main()
