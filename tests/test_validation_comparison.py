# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Report-only paired comparisons preserve selection and integer pixel counts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pvextent.evaluation import comparison


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    def ratio(a, b):
        return a / b if b else None

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "valid_pixels": tp + fp + fn + tn,
            "iou": ratio(tp, tp + fp + fn),
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "pixel_accuracy": ratio(tp + tn, tp + fp + fn + tn)}


def crop(identifier: str, counts: tuple[int, int, int, int]) -> dict:
    result = metrics(*counts)
    tp, fp, fn, tn = counts
    return {**result, "id": identifier, "split": "val", "source_id": "fixture",
            "prediction": f"predictions/{identifier}.png",
            "target_pv_pixels": tp + fn, "predicted_pv_pixels": tp + fp,
            "target_pv_area_m2": (tp + fn) * 0.01,
            "predicted_pv_area_m2": (tp + fp) * 0.01,
            "foreground_fraction": (tp + fn) / result["valid_pixels"],
            "target_touches_crop_edge": False, "error_pixels": fp + fn,
            "boundary_band_pixels": 10, "boundary_error_pixels": 0,
            "interior_error_pixels": fp + fn, "unclassified_error_pixels": 0}


class ValidationComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.output = self.root / "comparison"
        self.reference.mkdir()
        self.candidate.mkdir()
        self.write_fixtures()

    def write_analysis(self, directory: Path, rows: list[dict]) -> None:
        payload = (json.dumps(rows, sort_keys=True) + "\n").encode()
        (directory / "samples.json").write_bytes(payload)
        summary = metrics(*(sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")))
        # Input summaries may carry rounded displayed ratios; outputs must use counts.
        for key in ("iou", "dice", "precision", "recall", "pixel_accuracy"):
            if summary[key] is not None:
                summary[key] = round(summary[key], 3)
        summary["samples"] = len(rows)
        report = {"ok": True, "purpose": "frozen-validation-error-analysis",
                  "samples": len(rows), "test_used": False, "checkpoint_epoch": 3,
                  "checkpoint_sha256": ("a" if directory == self.reference else "b") * 64,
                  "dataset_fingerprint": ("c" if directory == self.reference else "d") * 64,
                  "val_selection_fingerprint": "e" * 64, "threshold": 0.5,
                  "image_dimensions": [320, 320], "gsd_m": 0.1, "device": "mps",
                  "package_versions": {"torch": "2.14.1", "timm": "1.0.30"},
                  "same_device_and_libraries": True,
                  "matches_checkpoint_validation_counts": True,
                  "samples_sha256": digest(payload), "summary": summary}
        self.write_report(directory, report)

    def write_fixtures(self) -> None:
        self.write_analysis(self.reference, [crop("improved", (6, 4, 4, 86)),
                                             crop("worsened", (9, 1, 1, 89)),
                                             crop("tied", (2, 0, 0, 98)),
                                             crop("undefined", (0, 0, 0, 100))])
        self.write_analysis(self.candidate, [crop("improved", (8, 2, 2, 88)),
                                             crop("worsened", (7, 5, 3, 85)),
                                             crop("tied", (2, 0, 0, 98)),
                                             crop("undefined", (0, 0, 0, 100))])

    @staticmethod
    def write_report(directory: Path, report: dict) -> None:
        (directory / "analysis-report.json").write_text(json.dumps(report, sort_keys=True) + "\n")

    @staticmethod
    def read_report(directory: Path) -> dict:
        return json.loads((directory / "analysis-report.json").read_text())

    @staticmethod
    def read_rows(directory: Path) -> list[dict]:
        return json.loads((directory / "samples.json").read_text())

    def replace_sample_payload(self, payload: bytes) -> None:
        (self.candidate / "samples.json").write_bytes(payload)
        report = self.read_report(self.candidate)
        report["samples_sha256"] = digest(payload)
        self.write_report(self.candidate, report)

    def compare(self):
        return comparison.compare_validation_analyses(self.reference, self.candidate, self.output)

    def assert_failure(self) -> None:
        inputs = {str(path): path.read_bytes() for directory in (self.reference, self.candidate)
                  for path in directory.iterdir()}
        entries = {path.name for path in self.root.iterdir()}
        with self.assertRaises(comparison.ValidationComparisonError):
            self.compare()
        self.assertFalse(self.output.exists())
        self.assertEqual({str(path): path.read_bytes() for directory in (self.reference, self.candidate)
                          for path in directory.iterdir()}, inputs)
        self.assertEqual({path.name for path in self.root.iterdir()}, entries)

    def test_paired_directions_and_global_metrics_come_from_integer_counts(self) -> None:
        result = self.compare()
        self.assertTrue(result["ok"])
        self.assertEqual(result["purpose"], "paired-validation-comparison")
        self.assertEqual(result["samples"], 4)
        self.assertFalse(result["test_used"])
        self.assertEqual(result["paired_iou"], {"improved": 1, "worse": 1, "tied": 1,
                                                "undefined_in_either": 1})
        old = result["reference"]["metrics"]
        new = result["candidate"]["metrics"]
        self.assertEqual(tuple(old[key] for key in ("tp", "fp", "fn", "tn")), (17, 5, 5, 373))
        self.assertEqual(tuple(new[key] for key in ("tp", "fp", "fn", "tn")), (17, 7, 5, 371))
        self.assertEqual(old["iou"], 17 / 27)
        self.assertEqual(new["iou"], 17 / 29)
        delta = new["iou"] - old["iou"]
        self.assertEqual(result["delta"]["counts"]["fp"], 2)
        self.assertAlmostEqual(result["delta"]["ratios"]["iou"], delta)
        self.assertAlmostEqual(result["delta"]["percentage_points"]["iou"], 100 * delta)
        self.assertEqual(result["rankings"]["most_improved"], ["improved"])
        self.assertEqual(result["rankings"]["most_worsened"], ["worsened"])
        # Native crop size does not turn ignored pixels into background supervision.
        self.assertEqual(old["valid_pixels"], 400)
        self.assertEqual(new["valid_pixels"], 400)
        self.assertEqual(result["paired_samples_sha256"], digest((self.output / "paired-samples.json").read_bytes()))
        for role, directory in (("reference", self.reference), ("candidate", self.candidate)):
            self.assertEqual(result[role]["analysis_report_sha256"], digest((directory / "analysis-report.json").read_bytes()))
        json.dumps(result, allow_nan=False)

    def test_reversing_models_reverses_global_difference(self) -> None:
        result = comparison.compare_validation_analyses(self.candidate, self.reference, self.output)
        self.assertGreater(result["delta"]["ratios"]["iou"], 0)
        self.assertEqual(result["delta"]["counts"]["fp"], -2)

    def test_both_empty_pv_masks_remain_undefined(self) -> None:
        self.write_analysis(self.reference, [crop("empty", (0, 0, 0, 100))])
        self.write_analysis(self.candidate, [crop("empty", (0, 0, 0, 100))])
        result = self.compare()
        self.assertIsNone(result["reference"]["metrics"]["iou"])
        self.assertIsNone(result["delta"]["ratios"]["iou"])
        self.assertIsNone(result["delta"]["percentage_points"]["iou"])
        self.assertEqual(result["paired_iou"]["undefined_in_either"], 1)
        self.assertEqual(result["paired_iou"]["tied"], 0)

    def test_only_json_is_opened_and_inputs_are_unchanged(self) -> None:
        before = {str(path): path.read_bytes() for directory in (self.reference, self.candidate)
                  for path in directory.iterdir()}
        original = Path.open
        opened = []

        def json_only(path, *args, **kwargs):
            self.assertEqual(path.suffix, ".json", f"Unexpected raster/checkpoint access: {path}")
            opened.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", json_only):
            self.compare()
        self.assertTrue(opened)
        self.assertEqual({str(path): path.read_bytes() for directory in (self.reference, self.candidate)
                          for path in directory.iterdir()}, before)

    def test_incompatible_validation_contracts_are_rejected(self) -> None:
        for key, value in (("val_selection_fingerprint", "f" * 64), ("threshold", 0.6),
                           ("image_dimensions", [640, 320]), ("gsd_m", 0.2),
                           ("device", "cpu"), ("package_versions", {"torch": "different"})):
            with self.subTest(key=key):
                self.write_fixtures()
                report = self.read_report(self.candidate)
                report[key] = value
                self.write_report(self.candidate, report)
                self.assert_failure()

    def test_unverified_or_test_analysis_is_rejected(self) -> None:
        for key, value in (("ok", False), ("test_used", True),
                           ("purpose", "test-evaluation"),
                           ("matches_checkpoint_validation_counts", False),
                           ("same_device_and_libraries", False)):
            with self.subTest(key=key):
                self.write_fixtures()
                report = self.read_report(self.candidate)
                report[key] = value
                self.write_report(self.candidate, report)
                self.assert_failure()

    def test_tampered_sample_hash_is_rejected(self) -> None:
        with (self.candidate / "samples.json").open("ab") as stream:
            stream.write(b" ")
        self.assert_failure()

    def test_different_or_duplicate_ids_are_rejected(self) -> None:
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                self.write_fixtures()
                rows = self.read_rows(self.candidate)
                rows[-1]["id"] = rows[0]["id"] if duplicate else "another-crop"
                self.write_analysis(self.candidate, rows)
                self.assert_failure()

    def test_changed_labels_or_valid_pixel_count_are_rejected(self) -> None:
        for change in ("positive", "valid"):
            with self.subTest(change=change):
                self.write_fixtures()
                rows = self.read_rows(self.candidate)
                # Each altered row is internally consistent but no longer the same target.
                counts = (9, 2, 2, 87) if change == "positive" else (8, 2, 2, 89)
                rows[0] = crop("improved", counts)
                self.write_analysis(self.candidate, rows)
                self.assert_failure()

    def test_inconsistent_integer_counts_are_rejected(self) -> None:
        for scope in ("row", "summary"):
            with self.subTest(scope=scope):
                self.write_fixtures()
                if scope == "row":
                    rows = self.read_rows(self.candidate)
                    rows[0]["valid_pixels"] -= 1
                    self.replace_sample_payload(json.dumps(rows).encode())
                else:
                    report = self.read_report(self.candidate)
                    report["summary"]["fp"] += 1
                    self.write_report(self.candidate, report)
                self.assert_failure()

    def test_duplicate_json_keys_nan_and_boolean_counts_are_rejected(self) -> None:
        for invalid in ("duplicate", "nan", "boolean"):
            with self.subTest(invalid=invalid):
                self.write_fixtures()
                if invalid == "duplicate":
                    path = self.candidate / "analysis-report.json"
                    text = path.read_text()
                    path.write_text(text.replace('"ok": true', '"ok": true, "ok": true'))
                elif invalid == "nan":
                    path = self.candidate / "analysis-report.json"
                    path.write_text(path.read_text().replace('"threshold": 0.5', '"threshold": NaN'))
                else:
                    rows = self.read_rows(self.candidate)
                    rows[0]["fp"] = True
                    self.replace_sample_payload(json.dumps(rows).encode())
                self.assert_failure()

    def test_existing_output_and_overlapping_output_are_never_replaced(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(comparison.ValidationComparisonError):
            self.compare()
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual({path.name for path in self.output.iterdir()}, {"keep.txt"})
        with self.assertRaises(comparison.ValidationComparisonError):
            comparison.compare_validation_analyses(self.reference, self.candidate, self.reference / "nested")
        self.assertFalse((self.reference / "nested").exists())

    def test_publish_failure_removes_staged_output_and_preserves_inputs(self) -> None:
        with patch.object(comparison, "_publish", side_effect=OSError("synthetic publish failure")):
            self.assert_failure()


if __name__ == "__main__":
    unittest.main()
