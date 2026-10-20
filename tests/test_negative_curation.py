# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Two independent AI reviews yield only agreed background, without holdout reads."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from PIL import Image

from pvextent.data.curation import NegativeCurationError, curate_negative_collection
from pvextent.data.validation import validate_dataset


METADATA = '''schema_version = "1.0"
dataset_id = "negative-curation-fixture"
dataset_version = "source-1"
[sources.synthetic]
name = "Generated test pixels"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC-BY-4.0"
annotation_license = "MIT"
attribution = "Original synthetic imagery attribution"
'''


def rectangle(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def tree_bytes(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@unittest.skipIf(importlib.util.find_spec("torch") is None, "Install the train extra for metadata readers")
class NegativeCurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        for name in ("images", "masks", "provenance"):
            (self.source / name).mkdir()
        (self.source / "dataset.toml").write_text(METADATA, encoding="utf-8")
        (self.source / "provenance" / "upstream-LICENSE").write_text("Original imagery terms\n", encoding="utf-8")
        (self.source / "provenance" / "original-MIT-LICENSE").write_text("Original source package MIT terms\n", encoding="utf-8")
        self.records = []
        for index, (identifier, split) in enumerate((("approved", "unassigned"), ("deferred", "unassigned"),
                                                     ("small", "unassigned"), ("frozen-test", "test"))):
            record = {"id": identifier, "image": f"images/{identifier}.png", "mask": f"masks/{identifier}.png",
                      "image_sha256": "", "mask_sha256": "", "width": 32, "height": 32, "gsd_m": 0.1,
                      "source_id": "synthetic", "capture_date": None, "spatial_group": f"group-{identifier}",
                      "building_ids": [f"synthetic:{identifier}"], "split": split,
                      "annotation_status": "upstream" if split == "test" else "unreviewed", "georeference": None}
            if split == "test":
                # This frozen source can be checked as metadata, but opening its
                # pixels would both violate isolation and fail PNG decoding.
                (self.source / record["image"]).write_bytes(b"unopened frozen test image")
                (self.source / record["mask"]).write_bytes(b"unopened frozen test mask")
            else:
                Image.new("RGB", (32, 32), (index * 30, 90, 170)).save(self.source / record["image"])
                # No source mask is needed to create these new annotations.
                (self.source / record["mask"]).write_bytes(b"source masks must stay unopened: " + identifier.encode())
            self.rehash(record)
            self.records.append(record)
        self.write_manifest()
        coordinates = [{"id": row["id"], "center_crs": "EPSG:4326",
                        "approximate_center": [7 + index * 0.1, 51]}
                       for index, row in enumerate(self.records)]
        (self.source / "provenance" / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in coordinates), encoding="utf-8")
        self.reviews = []
        self.review_paths = [self.root / "review-a.json", self.root / "review-b.json"]
        for reviewer in ("independent-a", "independent-b"):
            samples = []
            for record in self.records[:3]:
                samples.append({"id": record["id"], "image_sha256": record["image_sha256"],
                                "full_image_inspected": True, "decision": "approve-negative",
                                "background_polygons": [rectangle(0, 0, 23, 23)],
                                "notes": "Conservative visible background region; no prediction used."})
            self.reviews.append({"review_schema_version": "1.0", "reviewer": reviewer,
                                 "reviewer_type": "ai", "model_predictions_used": False, "samples": samples})
        self.reviews[1]["samples"][0]["background_polygons"] = [rectangle(8, 8, 31, 31)]
        self.reviews[1]["samples"][1]["decision"] = "defer"
        self.reviews[1]["samples"][2]["background_polygons"] = [rectangle(9, 9, 31, 31)]
        self.write_reviews()
        self.output = self.root / "curated"

    def rehash(self, record):
        for role in ("image", "mask"):
            record[f"{role}_sha256"] = hashlib.sha256((self.source / record[role]).read_bytes()).hexdigest()

    def write_manifest(self):
        (self.source / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.records), encoding="utf-8")

    def write_reviews(self):
        for path, review in zip(self.review_paths, self.reviews):
            path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")

    def curate(self, **kwargs):
        options = {"dataset_version": "ai-experiment-1", "allow_ai_review": True}
        options.update(kwargs)
        return curate_negative_collection([self.source], self.review_paths, self.output, **options)

    def assert_clean_failure(self, expected=None, **kwargs):
        with self.assertRaisesRegex(NegativeCurationError, expected or "."):
            self.curate(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".curated-curation-*")), [])

    def test_intersection_threshold_defer_and_ignore_are_exact(self):
        report = self.curate()
        records = {row["id"]: row for row in map(json.loads, (self.output / "manifest.jsonl").read_text().splitlines())}
        self.assertEqual(report["reviewed_ids"], ["approved"])
        self.assertEqual(report["unreviewed_ids"], ["deferred", "small"])
        self.assertEqual(report["background_pixels"], 256)
        self.assertEqual(report["ignored_pixels"], 3 * 1024 - 256)
        self.assertEqual(report["pv_pixels"], 0)
        decisions = {row["id"]: row for row in report["decisions"]}
        self.assertEqual(decisions["small"]["intersection_pixels"], 225)
        self.assertEqual(decisions["small"]["reason"], "insufficient_consensus")
        self.assertEqual(decisions["deferred"]["reason"], "reviewer_deferred")
        for identifier, record in records.items():
            self.assertEqual(record["split"], "unassigned")
            with Image.open(self.output / record["mask"]) as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(set(mask.get_flattened_data()), {0, 255} if identifier == "approved" else {255})
                if identifier == "approved":
                    self.assertEqual(mask.histogram()[0], 256)
                    for y in range(32):
                        for x in range(32):
                            self.assertEqual(mask.getpixel((x, y)), 0 if 8 <= x <= 23 and 8 <= y <= 23 else 255)
        self.assertTrue(validate_dataset(self.output).ok)

    def test_provenance_preserves_bytes_rights_coordinates_and_explicit_ai_origin(self):
        before = tree_bytes(self.source)
        review_bytes = [path.read_bytes() for path in self.review_paths]
        report = self.curate()
        self.assertEqual(tree_bytes(self.source), before)
        self.assertEqual([path.read_bytes() for path in self.review_paths], review_bytes)
        self.assertEqual(report["coordinate_records"], 3)
        self.assertTrue(report["ai_review_authorized"])
        self.assertFalse(report["human_review_claimed"])
        self.assertFalse(report["model_predictions_used"])
        metadata = tomllib.loads((self.output / "dataset.toml").read_text())
        self.assertEqual(metadata["dataset_id"], "pvextent-negative-curation")
        self.assertEqual(metadata["dataset_version"], "ai-experiment-1")
        for source in metadata["sources"].values():
            self.assertEqual(source["image_license"], "CC-BY-4.0")
            self.assertEqual(source["annotation_license"], "Apache-2.0")
            self.assertIn("Original synthetic imagery attribution", source["attribution"])
            self.assertIn("Two independent AI visual review passes", source["attribution"])
            self.assertIn("no human review claimed", source["attribution"])
        for entry, raw in zip(report["reviewers"], review_bytes):
            self.assertEqual((self.output / entry["file"]).read_bytes(), raw)
            self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
        source = report["sources"][0]
        location = (self.output / source["metadata_file"]).parent
        self.assertEqual((location / "dataset.toml").read_bytes(), before["dataset.toml"])
        for notice, digest in source["notice_hashes"].items():
            self.assertEqual((location / notice).read_bytes(), before[f"provenance/{notice}"])
            self.assertEqual(digest, hashlib.sha256((location / notice).read_bytes()).hexdigest())
        original = [json.loads(line) for line in (self.output / "provenance/original-source-records.jsonl").read_text().splitlines()]
        self.assertEqual({row["id"] for row in original}, {"approved", "deferred", "small"})
        for row in original:
            self.assertEqual(row["record"], next(record for record in self.records if record["id"] == row["id"]))
        for row in map(json.loads, (self.output / "manifest.jsonl").read_text().splitlines()):
            old = next(record for record in self.records if record["id"] == row["id"])
            self.assertEqual((self.output / row["image"]).read_bytes(), before[old["image"]])
            self.assertEqual(row["image_sha256"], old["image_sha256"])
        fingerprint = hashlib.sha256((self.output / "dataset.toml").read_bytes() + b"\0" +
                                     (self.output / "manifest.jsonl").read_bytes()).hexdigest()
        self.assertEqual(report["dataset_fingerprint"], fingerprint)
        self.assertEqual(json.loads((self.output / "curation-report.json").read_text()), report)

    def test_frozen_test_and_all_original_masks_are_never_opened_or_resolved(self):
        forbidden = {self.source / row[role] for row in self.records for role in ("image", "mask")
                     if role == "mask" or row["split"] == "test"}
        original_open, original_resolve = Path.open, Path.resolve

        def guarded_open(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError(f"Forbidden source pixel access: {path}")
            return original_open(path, *args, **kwargs)

        def guarded_resolve(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError(f"Forbidden source pixel resolution: {path}")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Path, "resolve", guarded_resolve):
            self.assertTrue(self.curate()["ok"])

    def test_authorization_and_two_distinct_prediction_free_ai_reviewers_are_required(self):
        for authorization in (False, None, 1, "true"):
            with self.subTest(authorization=authorization):
                self.assert_clean_failure("explicit allow_ai_review", allow_ai_review=authorization)
        original = copy.deepcopy(self.reviews)
        for field, value in (("reviewer_type", "human"), ("model_predictions_used", True),
                             ("reviewer", " Independent-A ")):
            with self.subTest(field=field):
                self.reviews = copy.deepcopy(original)
                self.reviews[1][field] = value
                self.write_reviews()
                self.assert_clean_failure()
        self.reviews = copy.deepcopy(original)
        self.reviews[0]["samples"][0]["full_image_inspected"] = False
        self.write_reviews()
        self.assert_clean_failure("inspection flag")

    def test_review_membership_hashes_and_assigned_splits_fail_before_any_pixel_access(self):
        original_reviews = copy.deepcopy(self.reviews)
        for modification, expected in (("hash", "SHA-256"), ("membership", "same sample IDs"),
                                       ("unknown", "absent"), ("assigned", "Only unassigned")):
            with self.subTest(modification=modification):
                self.reviews = copy.deepcopy(original_reviews)
                self.records[0]["split"] = "unassigned"
                self.records[0]["annotation_status"] = "unreviewed"
                if modification == "hash":
                    self.reviews[1]["samples"][0]["image_sha256"] = "0" * 64
                elif modification == "membership":
                    self.reviews[1]["samples"].pop()
                elif modification == "unknown":
                    for review in self.reviews:
                        review["samples"][0]["id"] = "unknown"
                else:
                    self.records[0]["split"] = "test"
                    self.records[0]["annotation_status"] = "upstream"
                self.write_manifest()
                self.write_reviews()
                with patch("pvextent.data.curation._image_payload", side_effect=AssertionError("No pixels before preflight")):
                    self.assert_clean_failure(expected)

    def test_reviews_reject_duplicate_ids_keys_nan_unknown_fields_and_bad_flags(self):
        original = copy.deepcopy(self.reviews)
        cases = ("duplicate-id", "duplicate-key", "nan", "extra-field", "boolean-inspection")
        for kind in cases:
            with self.subTest(kind=kind):
                self.reviews = copy.deepcopy(original)
                if kind == "duplicate-id":
                    self.reviews[0]["samples"].append(self.reviews[0]["samples"][0])
                elif kind == "extra-field":
                    self.reviews[0]["approval"] = True
                elif kind == "boolean-inspection":
                    self.reviews[0]["samples"][0]["full_image_inspected"] = 1
                self.write_reviews()
                if kind == "duplicate-key":
                    raw = self.review_paths[0].read_text().replace('"reviewer_type": "ai"', '"reviewer_type": "ai", "reviewer_type": "ai"')
                    self.review_paths[0].write_text(raw)
                elif kind == "nan":
                    raw = self.review_paths[0].read_text().replace('"model_predictions_used": false', '"model_predictions_used": NaN')
                    self.review_paths[0].write_text(raw)
                self.assert_clean_failure()

    def test_polygons_reject_out_of_grid_noninteger_and_self_intersecting_geometry(self):
        original = copy.deepcopy(self.reviews)
        invalid = (
            [[0, 0], [32, 0], [31, 31]],
            [[0, 0], [31.0, 0], [31, 31]],
            [[False, 0], [31, 0], [31, 31]],
            [[0, 0], [10, 10], [20, 20]],
            [[0, 0], [31, 31], [0, 31], [31, 0]],
            [[0, 0], [31, 0], [31, 31], [0, 0], [0, 31]],
            [[0, 0], [20, 0], [10, 0], [10, 20], [0, 20]],
            [[0, 0], [31, 0]],
        )
        for polygon in invalid:
            with self.subTest(polygon=polygon):
                self.reviews = copy.deepcopy(original)
                self.reviews[0]["samples"][0]["background_polygons"] = [polygon]
                self.write_reviews()
                with patch("pvextent.data.curation._image_payload", side_effect=AssertionError("No pixels for invalid polygon")):
                    self.assert_clean_failure()

    def test_closed_polygon_and_union_within_each_reviewer_are_supported(self):
        for review in self.reviews:
            review["samples"] = review["samples"][:1]
        self.reviews[0]["samples"][0]["background_polygons"] = [
            rectangle(0, 0, 7, 31), rectangle(8, 0, 15, 31) + [[8, 0]]]
        self.reviews[1]["samples"][0]["background_polygons"] = [rectangle(0, 0, 31, 31)]
        self.write_reviews()
        report = self.curate()
        self.assertEqual(report["background_pixels"], 512)
        self.assertEqual(report["reviewed_samples"], 1)

    def test_duplicate_source_ids_fail_without_reading_any_images(self):
        duplicate = self.root / "duplicate"
        duplicate.mkdir()
        (duplicate / "dataset.toml").write_bytes((self.source / "dataset.toml").read_bytes())
        (duplicate / "manifest.jsonl").write_bytes((self.source / "manifest.jsonl").read_bytes())
        with patch("pvextent.data.curation._image_payload", side_effect=AssertionError("No images for duplicate IDs")):
            with self.assertRaisesRegex(NegativeCurationError, "unique across"):
                curate_negative_collection([self.source, duplicate], self.review_paths, self.output,
                                           dataset_version="v1", allow_ai_review=True)
        self.assertFalse(self.output.exists())

    def test_tampered_reviewed_image_fails_and_removes_all_staging_output(self):
        path = self.source / self.records[1]["image"]
        path.write_bytes(path.read_bytes() + b"changed since review")
        self.assert_clean_failure("SHA-256")

    def test_invalid_native_image_cannot_be_approved_by_rehashing_a_review(self):
        record = self.records[0]
        for mode, size in (("RGBA", (32, 32)), ("RGB", (31, 32))):
            with self.subTest(mode=mode, size=size):
                Image.new(mode, size).save(self.source / record["image"])
                self.rehash(record)
                for review in self.reviews:
                    review["samples"][0]["image_sha256"] = record["image_sha256"]
                self.write_manifest()
                self.write_reviews()
                self.assert_clean_failure("Selected image is invalid")

    def test_existing_output_and_source_overlap_are_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_bytes(b"existing result")
        with self.assertRaisesRegex(NegativeCurationError, "already exists"):
            self.curate()
        self.assertEqual(sentinel.read_bytes(), b"existing result")
        before = tree_bytes(self.source)
        with self.assertRaisesRegex(NegativeCurationError, "separate"):
            curate_negative_collection([self.source], self.review_paths, self.source / "new-child",
                                       dataset_version="v1", allow_ai_review=True)
        self.assertEqual(tree_bytes(self.source), before)

    def test_bounded_regular_review_files_and_dataset_version_are_required(self):
        original = self.review_paths[0].read_bytes()
        with patch("pvextent.data.curation.MAX_REVIEW_BYTES", len(original) - 1):
            self.assert_clean_failure("bounded regular file")
        self.review_paths[0].unlink()
        real = self.root / "real-review.json"
        real.write_bytes(original)
        self.review_paths[0].symlink_to(real)
        self.assert_clean_failure("bounded regular file")
        for version in ("", " ", None, True, "x" * 257):
            with self.subTest(version=version):
                self.assert_clean_failure("dataset_version", dataset_version=version)


if __name__ == "__main__":
    unittest.main()
