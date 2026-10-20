# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Known-count diagnostics for masked PV errors, boundaries and crop strata."""

from __future__ import annotations

import json
import unittest

try:
    import torch
    from pvextent.evaluation.error_metrics import aggregate_statistics, sample_statistics
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the train extra for error metric tests")
class ErrorMetricTests(unittest.TestCase):
    def vertical_boundary(self):
        target = torch.zeros((9, 9), dtype=torch.bool)
        target[:, 4:] = True
        prediction = target.clone()
        prediction[4, 0] = True
        prediction[4, 4] = False
        prediction[4, 8] = False
        return prediction, target, torch.ones_like(target)

    def test_known_counts_areas_and_fixed_three_pixel_boundary_band(self) -> None:
        prediction, target, valid = self.vertical_boundary()
        row = sample_statistics(prediction, target, valid, gsd_m=0.1)
        self.assertEqual({key: row[key] for key in ("tp", "fp", "fn", "tn")},
                         {"tp": 43, "fp": 1, "fn": 2, "tn": 35})
        self.assertEqual(row["valid_pixels"], 81)
        self.assertEqual(row["target_pv_pixels"], 45)
        self.assertEqual(row["predicted_pv_pixels"], 44)
        self.assertAlmostEqual(row["target_pv_area_m2"], 0.45)
        self.assertAlmostEqual(row["predicted_pv_area_m2"], 0.44)
        self.assertAlmostEqual(row["foreground_fraction"], 45 / 81)
        self.assertTrue(row["target_touches_crop_edge"])
        # Columns 1..6 have both classes in their actual 7x7 neighborhood.
        self.assertEqual(row["boundary_band_pixels"], 54)
        self.assertEqual(row["boundary_error_pixels"], 1)
        self.assertEqual(row["interior_error_pixels"], 2)
        self.assertEqual(row["unclassified_error_pixels"], 0)
        self.assertEqual(row["error_pixels"], 3)
        self.assertAlmostEqual(row["iou"], 43 / 46)
        self.assertAlmostEqual(row["dice"], 86 / 89)

    def test_ignore_neighborhoods_exclude_boundary_and_interior_classification(self) -> None:
        prediction, target, valid = self.vertical_boundary()
        valid[4, 4] = False
        prediction[4, 5] = False
        prediction[0, 4] = False
        row = sample_statistics(prediction, target, valid, gsd_m=0.2)
        self.assertEqual(row["error_pixels"], 4)
        self.assertEqual(row["boundary_band_pixels"], 12)
        self.assertEqual(row["boundary_error_pixels"], 1)
        self.assertEqual(row["interior_error_pixels"], 2)
        self.assertEqual(row["unclassified_error_pixels"], 1)
        self.assertEqual(row["error_pixels"], sum(row[key] for key in (
            "boundary_error_pixels", "interior_error_pixels", "unclassified_error_pixels")))
        # Neither source nor predictions at ignored pixels affect diagnostics.
        target[4, 4] = ~target[4, 4]
        prediction[4, 4] = ~prediction[4, 4]
        self.assertEqual(row, sample_statistics(prediction, target, valid, gsd_m=0.2))

    def test_crop_padding_does_not_invent_background_ignore_or_a_semantic_boundary(self) -> None:
        for size in ((1, 1), (2, 3), (9, 9)):
            with self.subTest(size=size):
                target = torch.ones(size, dtype=torch.bool)
                valid = torch.ones_like(target)
                prediction = target.clone()
                prediction[0, 0] = False
                row = sample_statistics(prediction, target, valid, gsd_m=1)
                self.assertTrue(row["target_touches_crop_edge"])
                self.assertEqual(row["boundary_band_pixels"], 0)
                self.assertEqual(row["boundary_error_pixels"], 0)
                self.assertEqual(row["interior_error_pixels"], 1)
                self.assertEqual(row["unclassified_error_pixels"], 0)

    def test_actual_ignored_neighbors_differ_from_space_outside_the_crop(self) -> None:
        target = torch.ones((3, 3), dtype=torch.bool)
        valid = torch.zeros_like(target)
        valid[1, 1] = True
        row = sample_statistics(torch.zeros_like(target), target, valid, gsd_m=1)
        self.assertEqual(row["valid_pixels"], 1)
        self.assertEqual(row["unclassified_error_pixels"], 1)
        self.assertEqual(row["boundary_error_pixels"], 0)
        self.assertEqual(row["interior_error_pixels"], 0)
        self.assertFalse(row["target_touches_crop_edge"])

    def test_chebyshev_radius_includes_diagonal_neighbor_at_three_pixels(self) -> None:
        target = torch.zeros((9, 9), dtype=torch.bool)
        target[4, 4] = True
        prediction = target.clone()
        prediction[1, 1] = True
        prediction[0, 0] = True
        row = sample_statistics(prediction, target, torch.ones_like(target), gsd_m=1)
        self.assertEqual(row["boundary_band_pixels"], 49)
        self.assertEqual(row["boundary_error_pixels"], 1)
        self.assertEqual(row["interior_error_pixels"], 1)
        self.assertFalse(row["target_touches_crop_edge"])

    def test_aggregation_uses_global_counts_and_reports_macro_denominators(self) -> None:
        small = torch.ones((2, 2), dtype=torch.bool)
        large = torch.zeros((1, 100), dtype=torch.bool)
        large[0, 0] = True
        negative = torch.zeros((3, 3), dtype=torch.bool)
        rows = [
            sample_statistics(small, small, torch.ones_like(small), gsd_m=1),
            sample_statistics(torch.ones_like(large), large, torch.ones_like(large), gsd_m=1),
            sample_statistics(negative, negative, torch.ones_like(negative), gsd_m=1),
        ]
        report = aggregate_statistics(iter(rows))
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["valid_pixels"], 113)
        self.assertEqual((report["tp"], report["fp"], report["fn"], report["tn"]), (5, 99, 0, 9))
        self.assertAlmostEqual(report["iou"], 5 / 104)
        self.assertAlmostEqual(report["macro_iou"], 0.505)
        self.assertAlmostEqual(report["median_iou"], 0.505)
        self.assertAlmostEqual(report["min_iou"], 0.01)
        self.assertEqual(report["defined_iou_samples"], 2)
        self.assertEqual(report["undefined_iou_samples"], 1)
        self.assertNotEqual(report["iou"], report["macro_iou"])
        json.dumps(report, allow_nan=False)

    def test_area_bins_use_total_source_pv_area_and_exact_ten_and_fifty_boundaries(self) -> None:
        rows = []
        for size in ((3, 3), (1, 10), (7, 7), (5, 10)):
            target = torch.ones(size, dtype=torch.bool)
            rows.append(sample_statistics(target, target, torch.ones_like(target), gsd_m=1))
        target = torch.zeros((3, 3), dtype=torch.bool)
        target[1, 1] = True
        rows.append(sample_statistics(target, target, torch.ones_like(target), gsd_m=1))
        report = aggregate_statistics(rows)
        self.assertEqual(report["target_pv_area_m2"], 119)
        self.assertEqual({key: value["samples"] for key, value in report["area_strata"].items()},
                         {"lt_10_m2": 2, "10_to_lt_50_m2": 2, "ge_50_m2": 1})
        self.assertEqual(report["crop_edge_strata"]["touches_edge"]["samples"], 4)
        self.assertEqual(report["crop_edge_strata"]["does_not_touch_edge"]["samples"], 1)
        self.assertIn("per crop", report["area_strata_basis"])
        self.assertIn("not individual array", report["area_strata_basis"])
        for stratum in (*report["area_strata"].values(), *report["crop_edge_strata"].values()):
            self.assertEqual(stratum["iou"], 1)

    def test_boundary_aggregate_totals_and_shares_reconcile_all_errors(self) -> None:
        prediction, target, valid = self.vertical_boundary()
        first = sample_statistics(prediction, target, valid, gsd_m=1)
        valid[4, 4] = False
        prediction[4, 5] = False
        prediction[0, 4] = False
        second = sample_statistics(prediction, target, valid, gsd_m=1)
        report = aggregate_statistics([first, second])
        self.assertEqual(report["boundary_band_pixels"], 66)
        self.assertEqual(report["error_pixels"], 7)
        self.assertEqual(report["boundary_error_pixels"], 2)
        self.assertEqual(report["interior_error_pixels"], 4)
        self.assertEqual(report["unclassified_error_pixels"], 1)
        self.assertAlmostEqual(report["boundary_error_share"], 2 / 7)
        self.assertAlmostEqual(report["boundary_error_rate"], 2 / 66)
        self.assertAlmostEqual(sum(report[key] for key in (
            "boundary_error_share", "interior_error_share", "unclassified_error_share")), 1)

    def test_empty_aggregates_and_undefined_negative_iou_remain_null(self) -> None:
        empty = aggregate_statistics([])
        self.assertEqual(empty["samples"], 0)
        self.assertIsNone(empty["iou"])
        self.assertIsNone(empty["macro_iou"])
        self.assertIsNone(empty["median_iou"])
        self.assertIsNone(empty["min_iou"])
        self.assertIsNone(empty["boundary_error_share"])
        self.assertEqual(empty["undefined_iou_samples"], 0)
        for stratum in empty["area_strata"].values():
            self.assertEqual(stratum["samples"], 0)
            self.assertIsNone(stratum["iou"])
        target = torch.zeros((2, 2), dtype=torch.bool)
        report = aggregate_statistics([sample_statistics(target, target, ~target, gsd_m=1)])
        self.assertEqual(report["undefined_iou_samples"], 1)
        self.assertEqual(report["defined_iou_samples"], 0)
        self.assertIsNone(report["macro_iou"])
        self.assertEqual(report["pixel_accuracy"], 1)
        json.dumps(empty, allow_nan=False)

    def test_malformed_inputs_fail_before_computation(self) -> None:
        target = torch.ones((2, 2), dtype=torch.bool)
        cases = (
            ([], target, target), (target.float(), target, target),
            (target, target.long(), target), (target, target, target.float()),
            (target[None], target, target), (target, target[:, :1], target),
            (target, target, ~target),
            (torch.empty((0, 2), dtype=torch.bool),) * 3,
            (target, torch.ones((2, 2), dtype=torch.bool, device="meta"), target),
        )
        for values in cases:
            with self.subTest(shapes=[getattr(value, "shape", None) for value in values]):
                with self.assertRaises(ValueError):
                    sample_statistics(*values, gsd_m=1)
        for gsd in (0, -1, True, float("nan"), float("inf"), 10 ** 1000, 1e308, 1e-300):
            with self.subTest(gsd=gsd), self.assertRaises(ValueError):
                sample_statistics(target, target, target, gsd_m=gsd)

    def test_aggregate_rejects_inconsistent_counts_and_nonfinite_areas(self) -> None:
        target = torch.ones((2, 2), dtype=torch.bool)
        row = sample_statistics(target, target, target, gsd_m=1)
        for changes in (
            {"tp": True}, {"valid_pixels": 3}, {"error_pixels": 1},
            {"boundary_error_pixels": 1}, {"target_pv_area_m2": float("nan")},
            {"target_pv_area_m2": 0}, {"target_touches_crop_edge": "yes"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                aggregate_statistics([{**row, **changes}])


if __name__ == "__main__":
    unittest.main()
