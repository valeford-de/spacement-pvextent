# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Local review exports preserve pixels, provenance, and pending decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from pvextent.data import validate_dataset
from pvextent.data.pv4ger import DatasetImportError
from pvextent.data.review import export_review_bundle


class ReviewBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source, self.output = self.base / "dataset", self.base / "review"
        for name in ("images", "masks"):
            (self.source / name).mkdir(parents=True)
        (self.source / "dataset.toml").write_text(
            'schema_version="1.0"\ndataset_id="review-fixture"\ndataset_version="1"\n'
            '[sources.synthetic]\nname="Synthetic fixtures"\nurl="https://example.invalid/data"\n'
            'version="1"\nimage_license="CC0-1.0"\nannotation_license="CC0-1.0"\n'
            'attribution="Synthetic fixture"\n', encoding="utf-8")
        self.records = []

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def add(self, identifier: str, mask_values: tuple[int, ...] = (0, 1, 255), **changes: object) -> dict:
        image_path, mask_path = f"images/{identifier}.png", f"masks/{identifier}.png"
        Image.new("RGB", (3, 1), (20, 60, len(self.records) + 10)).save(self.source / image_path)
        mask = Image.new("L", (3, 1))
        mask.putdata(mask_values)
        mask.save(self.source / mask_path)
        record = {
            "id": identifier, "image": image_path, "mask": mask_path,
            "image_sha256": self.digest(self.source / image_path),
            "mask_sha256": self.digest(self.source / mask_path),
            "width": 3, "height": 1, "gsd_m": 0.1, "source_id": "synthetic",
            "capture_date": None, "spatial_group": "test-area", "building_ids": [],
            "split": "unassigned", "annotation_status": "upstream", "georeference": None,
        }
        record.update(changes)
        self.records.append(record)
        self.flush()
        return record

    def flush(self) -> None:
        (self.source / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")

    def review_manifest(self) -> dict:
        return json.loads((self.output / "review-manifest.json").read_text(encoding="utf-8"))

    def test_input_is_immutable_and_fingerprint_and_exact_pixels_are_preserved(self) -> None:
        self.add("one")
        before = {path.relative_to(self.source): self.digest(path) for path in self.source.rglob("*") if path.is_file()}
        fingerprint = validate_dataset(self.source).stats["dataset_fingerprint"]
        report = export_review_bundle(self.source, self.output)
        after = {path.relative_to(self.source): self.digest(path) for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(report["dataset_fingerprint"], fingerprint)
        document = self.review_manifest()
        self.assertEqual(document["dataset_fingerprint"], fingerprint)
        sample = document["samples"][0]
        for field in ("image", "mask"):
            self.assertEqual((self.output / sample[field]).read_bytes(), (self.source / sample["source_record"][field]).read_bytes())
            self.assertFalse((self.output / sample[field]).is_symlink())
        self.assertEqual(sample["source_record"]["annotation_status"], "upstream")
        self.assertEqual(sample["decision"], "pending")

    def test_all_no_pv_masks_are_included_before_twelve_deterministic_positives(self) -> None:
        for number in reversed(range(15)):
            self.add(f"positive-{number:02d}")
        for number in range(14):
            self.add(f"zero-{number:02d}", (0, 0, 0))
        self.add("ignore", (255, 255, 255))
        self.add("mixed-no-pv", (0, 255, 0))
        report = export_review_bundle(self.source, self.output)
        samples = self.review_manifest()["samples"]
        self.assertEqual(report["no_pv_label_candidates"], 16)
        self.assertEqual(report["positive_examples"], 12)
        self.assertEqual(samples[0]["id"], "ignore")
        self.assertTrue(all(sample["no_pv_label_candidate"] for sample in samples[:16]))
        self.assertEqual([sample["id"] for sample in samples[16:]], [f"positive-{number:02d}" for number in range(12)])
        self.assertTrue(all(sample["decision"] == "pending" for sample in samples))

    def test_label_view_and_overlay_distinguish_pv_background_and_ignore(self) -> None:
        self.add("one")
        export_review_bundle(self.source, self.output)
        sample = self.review_manifest()["samples"][0]
        with Image.open(self.output / sample["labels"]) as labels:
            self.assertEqual(list(labels.get_flattened_data()), [(0, 0, 0), (0, 200, 150), (255, 179, 71)])
        with Image.open(self.output / sample["overlay"]) as overlay:
            self.assertEqual(overlay.getpixel((0, 0)), (20, 60, 10))
            self.assertNotEqual(overlay.getpixel((1, 0)), (20, 60, 10))
            self.assertNotEqual(overlay.getpixel((2, 0)), (20, 60, 10))
        with Image.open(self.output / sample["mask"]) as mask:
            self.assertEqual(list(mask.get_flattened_data()), [0, 1, 255])

    def test_explicit_sample_ids_preserve_requested_order(self) -> None:
        self.add("first")
        self.add("second", (0, 0, 0))
        self.add("third")
        report = export_review_bundle(self.source, self.output, sample_ids=["third", "first"])
        self.assertEqual(report["selection"], "explicit-sample-ids")
        self.assertEqual([sample["id"] for sample in self.review_manifest()["samples"]], ["third", "first"])

    def test_unknown_duplicate_or_empty_explicit_selection_is_rejected(self) -> None:
        self.add("first")
        for selected in (["missing"], ["first", "first"], [], "first"):
            with self.subTest(selected=selected), self.assertRaises(DatasetImportError):
                export_review_bundle(self.source, self.output, sample_ids=selected)
            self.assertFalse(self.output.exists())

    def test_html_escapes_source_metadata_and_starts_pending(self) -> None:
        attack = '</script><script>alert("source")</script>'
        self.add("candidate", (0, 0, 0), spatial_group=attack)
        export_review_bundle(self.source, self.output)
        html = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(attack, html)
        self.assertIn("&lt;/script&gt;&lt;script&gt;", html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertIn('<strong class="status">pending</strong>', html)
        self.assertIn("No visible PV (requires full image review)", html)
        self.assertIn("full_image_inspected", html)
        self.assertIn("new Date().toISOString()", html)
        self.assertIn("URL.createObjectURL", html)
        self.assertNotIn('<img src="http', html)
        self.assertNotIn('fetch(', html)
        self.assertTrue(all(sample["decision"] == "pending" for sample in self.review_manifest()["samples"]))

    def test_invalid_dataset_is_not_exported(self) -> None:
        record = self.add("one")
        (self.source / record["image"]).write_bytes(b"invalid changed image")
        with self.assertRaisesRegex(DatasetImportError, "invalid dataset"):
            export_review_bundle(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_target_and_source_overlap_are_rejected(self) -> None:
        self.add("one")
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(DatasetImportError):
            export_review_bundle(self.source, self.output)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        with self.assertRaises(DatasetImportError):
            export_review_bundle(self.source, self.source / "review")

    def test_failed_publish_leaves_no_partial_export(self) -> None:
        self.add("one")
        with patch("pvextent.data.review._publish", side_effect=OSError("publish failure")):
            with self.assertRaisesRegex(DatasetImportError, "publish failure"):
                export_review_bundle(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob(".review-review-*")), [])

    def test_source_notices_are_copied(self) -> None:
        self.add("one")
        (self.source / "provenance").mkdir()
        (self.source / "provenance" / "original-MIT-LICENSE").write_text("Synthetic license notice", encoding="utf-8")
        export_review_bundle(self.source, self.output)
        self.assertEqual((self.output / "provenance" / "original-MIT-LICENSE").read_text(encoding="utf-8"), "Synthetic license notice")
        self.assertIn("provenance/original-MIT-LICENSE", self.review_manifest()["source_notices"])


if __name__ == "__main__":
    unittest.main()
