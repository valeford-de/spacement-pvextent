# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Training review acquisition retains separation without holdout namespace."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

import test_negative_holdout as fixtures

try:
    from pvextent.data import negative_holdout, nrw_training
    from pvextent.data.training import _read_manifest
except ImportError:
    nrw_training = None


@unittest.skipIf(fixtures.Transformer is None or nrw_training is None, "Install train and geo extras")
class TrainingCandidateTests(unittest.TestCase):
    write_reference = fixtures.NegativeHoldoutTests.write_reference
    write_protocol = fixtures.NegativeHoldoutTests.write_protocol
    image_payload = staticmethod(fixtures.NegativeHoldoutTests.image_payload)
    fetch = fixtures.NegativeHoldoutTests.fetch

    def setUp(self):
        fixtures.NegativeHoldoutTests.setUp(self)
        self.rules["distance_reference_splits"] = ["val", "test", "unassigned"]
        self.write_protocol()

    def acquire(self, exclusions=None):
        return nrw_training.fetch_training_candidates(exclusions or [self.source], self.output,
                                                      protocol_path=self.protocol)

    def pilot(self):
        directory = self.root / "reserved-pilot"
        (directory / "provenance").mkdir(parents=True)
        (directory / "dataset.toml").write_text(fixtures.METADATA.replace(
            'dataset_id = "holdout-exclusion-fixture"', 'dataset_id = "nrw-negative-holdout-candidates"'))
        rows, centers = [], []
        for index in range(20):
            split = "test" if index < 7 else "unassigned"
            row = {**self.rows[0], "id": f"pilot-{index}", "image": f"images/pilot-{index}.png",
                   "mask": f"masks/pilot-{index}.png", "image_sha256": hashlib.sha256(f"pilot-{index}".encode()).hexdigest(),
                   "spatial_group": f"reserved-pilot-{index}", "building_ids": [f"synthetic:pilot-{index}"],
                   "split": split, "annotation_status": "reviewed" if split == "test" else "unreviewed"}
            rows.append(row)
            centers.append({"id": row["id"], "approximate_center": list(self.inverse.transform(330000, 5600000 + index*200)),
                            "center_crs": "EPSG:4326"})
        (directory / "manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        (directory / "provenance/samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in centers))
        return directory, rows, centers

    @contextmanager
    def forbid_pixels(self, sources):
        original_open, original_resolve, original_image_open = Path.open, Path.resolve, Image.open
        original_validate = negative_holdout.validate_dataset

        def forbidden(path):
            path = Path(path).absolute()
            return any(path.is_relative_to(source / folder) for source in sources for folder in ("images", "masks"))

        def guarded_open(path, *args, **kwargs):
            if forbidden(path):
                raise AssertionError("Reserved raster opened: " + str(path))
            return original_open(path, *args, **kwargs)

        def guarded_resolve(path, *args, **kwargs):
            if forbidden(path):
                raise AssertionError("Reserved raster path resolved: " + str(path))
            return original_resolve(path, *args, **kwargs)

        def guarded_decode(fp, *args, **kwargs):
            if isinstance(fp, (str, os.PathLike)) and forbidden(fp):
                raise AssertionError("Reserved raster decoded: " + str(fp))
            return original_image_open(fp, *args, **kwargs)

        def new_only(directory, *args, **kwargs):
            self.assertNotIn(Path(directory).resolve(), sources)
            return original_validate(directory, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Path, "resolve", guarded_resolve), \
                patch.object(Image, "open", guarded_decode), patch.object(negative_holdout, "validate_dataset", new_only):
            yield

    def test_training_namespace_all_ignore_native_bytes_and_source_notices(self):
        with self.forbid_pixels([self.source]):
            report = self.acquire()
        metadata, rows, fingerprint = _read_manifest(self.output)
        self.assertEqual(metadata["dataset_id"], "nrw-training-candidates")
        self.assertEqual(report["dataset_fingerprint"], fingerprint)
        self.assertEqual(report["purpose"], "experimental-training-review-pool")
        self.assertEqual(report["intended_use"], "training-review")
        self.assertFalse(report["training_use_allowed"])
        self.assertFalse(report["model_predictions_used"])
        self.assertFalse(report["exclusion_rasters_opened"])
        self.assertEqual((report["pv_pixels"], report["background_pixels"]), (0, 0))
        for row in rows:
            self.assertEqual(row["spatial_group"], "nrw-training-dortmund")
            self.assertEqual((row["split"], row["annotation_status"]), ("unassigned", "unreviewed"))
            with Image.open(self.output / row["mask"]) as mask:
                self.assertEqual(mask.getextrema(), (255, 255))
            transform = row["georeference"]["transform"]
            bounds = (transform[2], transform[5]-32, transform[2]+32, transform[5])
            self.assertEqual((self.output / row["image"]).read_bytes(), self.image_payload(bounds))
        notes = (self.output / "provenance/SOURCES.md").read_text()
        self.assertIn("possible future training", notes)
        self.assertIn("Datenlizenz Deutschland - Zero 2.0", notes)
        self.assertEqual((self.output / "provenance/protocol.json").read_bytes(), self.protocol.read_bytes())

    def test_training_and_entire_twenty_case_pilot_are_metadata_only_exclusions(self):
        pilot, _, _ = self.pilot()
        sources = [self.source, pilot]
        before = {path: path.read_bytes() for source in sources for path in source.rglob("*") if path.is_file()}
        with self.forbid_pixels(sources):
            report = self.acquire(sources)
        self.assertEqual(report["reference_snapshot_count"], 2)
        self.assertEqual(report["reference_center_count"], 24)
        self.assertEqual(report["identity_reference_records"], 24)
        self.assertEqual(report["distance_reference_records"], 23)
        self.assertEqual(report["distance_reference_splits"], ["val", "test", "unassigned"])
        self.assertEqual(report["references"][1]["split_counts"], {"test": 7, "unassigned": 13})
        self.assertEqual({path: path.read_bytes() for source in sources for path in source.rglob("*") if path.is_file()}, before)

    def test_near_training_centre_is_allowed_but_its_identity_still_excluded(self):
        self.features = [fixtures.building(1)]
        self.points["reference-train"] = (392080, 5711600)
        self.write_reference()
        with self.forbid_pixels([self.source]):
            report = self.acquire()
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["identity_reference_records"], 4)
        self.assertEqual(report["distance_reference_records"], 3)
        # An unrelated train centre may be close. Its building identity may not
        # be reacquired, even though it is outside the distance reference set.
        self.output = self.root / "identity-overlap"
        self.rows[0]["building_ids"] = ["geobasis-nrw:alkis:" + self.features[0]["id"]]
        self.write_reference()
        self.calls.clear()
        with self.forbid_pixels([self.source]), self.assertRaises(nrw_training.TrainingCandidatesError):
            self.acquire()
        self.assertFalse(fixtures.NegativeHoldoutTests.maps(self.calls))

    def test_near_validation_test_or_unassigned_centre_is_rejected(self):
        self.features = [fixtures.building(1)]
        for split in ("val", "test", "unassigned"):
            with self.subTest(split=split):
                old = self.points["reference-" + split]
                self.points["reference-" + split] = (392080, 5711600)
                self.write_reference()
                self.calls.clear()
                with self.forbid_pixels([self.source]), self.assertRaises(nrw_training.TrainingCandidatesError):
                    self.acquire()
                self.assertFalse(fixtures.NegativeHoldoutTests.maps(self.calls))
                self.assertFalse(self.output.exists())
                self.points["reference-" + split] = old

    def test_protocol_must_explicitly_bind_the_wrappers_distance_split_rule(self):
        for value in (None, ["train", "val", "test", "unassigned"], ["val", "test"], ["test", "val", "unassigned"]):
            with self.subTest(value=value):
                if value is None:
                    self.rules.pop("distance_reference_splits", None)
                else:
                    self.rules["distance_reference_splits"] = value
                self.write_protocol()
                with self.assertRaises(nrw_training.TrainingCandidatesError):
                    self.acquire()
        self.assertEqual(self.calls, [])

    def test_any_pilot_case_excludes_nearby_candidate_before_imagery(self):
        pilot, _, centers = self.pilot()
        self.features = [fixtures.building(1)]
        for index in (0, 6, 7, 19):
            with self.subTest(index=index):
                original = centers[index]["approximate_center"]
                centers[index]["approximate_center"] = list(self.inverse.transform(392080, 5711600))
                (pilot / "provenance/samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in centers))
                self.calls.clear()
                with self.forbid_pixels([self.source, pilot]), self.assertRaises(nrw_training.TrainingCandidatesError):
                    self.acquire([self.source, pilot])
                self.assertFalse(fixtures.NegativeHoldoutTests.maps(self.calls))
                self.assertFalse(self.output.exists())
                centers[index]["approximate_center"] = original

    def test_existing_output_refused_and_training_error_is_clean(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("keep")
        with self.assertRaises(nrw_training.TrainingCandidatesError):
            self.acquire()
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.calls, [])

    def test_invalid_metadata_never_creates_training_labels_or_backfills(self):
        self.rules["regions"][0]["limit"] = 1
        self.write_protocol()
        self.capture = "unknown"
        with self.assertRaises(nrw_training.TrainingCandidatesError):
            self.acquire()
        self.assertEqual(len([url for url, _ in self.calls if "GetFeatureInfo" in url]), 1)
        self.assertFalse(fixtures.NegativeHoldoutTests.maps(self.calls))
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".holdout-fetch-*")))


if __name__ == "__main__":
    unittest.main()
