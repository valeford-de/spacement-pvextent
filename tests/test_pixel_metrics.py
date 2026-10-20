# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Known pixel confusion counts, ignore handling and dataset aggregation."""

from __future__ import annotations

import json
import unittest

try:
    import torch
    from pvextent.data.training import TrainingDataError
    from pvextent.evaluation.pixels import PixelConfusion
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the train extra for pixel metrics tests")
class PixelConfusionTests(unittest.TestCase):
    def fixture(self):
        logits = torch.tensor([[[[3.0, 3.0, -3.0, -3.0], [3.0, -3.0, 3.0, -3.0]]]])
        target = torch.tensor([[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 255.0]]]])
        valid = target != 255
        return logits, target, valid

    def test_exact_confusion_and_manually_computed_ratios(self) -> None:
        metrics = PixelConfusion()
        metrics.update(*self.fixture())
        result = metrics.to_dict()
        self.assertEqual({key: result[key] for key in ("tp", "fp", "fn", "tn", "valid_pixels")},
                         {"tp": 3, "fp": 1, "fn": 1, "tn": 2, "valid_pixels": 7})
        self.assertAlmostEqual(result["iou"], 3 / 5)
        self.assertAlmostEqual(result["dice"], 6 / 8)
        self.assertAlmostEqual(result["precision"], 3 / 4)
        self.assertAlmostEqual(result["recall"], 3 / 4)
        self.assertAlmostEqual(result["pixel_accuracy"], 5 / 7)
        self.assertEqual(metrics.samples, 1)
        json.dumps(result, allow_nan=False)

    def test_aggregation_matches_whole_data_with_unequal_valid_pixel_counts(self) -> None:
        logits, target, valid = self.fixture()
        whole, batches = PixelConfusion(), PixelConfusion()
        whole.update(logits, target, valid)
        # Unequal spatial chunks also exercise unequal valid-pixel weighting.
        batches.update(logits[:, :, :, :1], target[:, :, :, :1], valid[:, :, :, :1])
        batches.update(logits[:, :, :, 1:], target[:, :, :, 1:], valid[:, :, :, 1:])
        self.assertEqual(whole.to_dict(), batches.to_dict())

    def test_probability_exactly_half_is_positive_and_threshold_is_configurable(self) -> None:
        logits = torch.zeros((1, 1, 1, 2))
        target = torch.tensor([[[[1, 0]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        metrics = PixelConfusion()
        metrics.update(logits, target, valid)
        self.assertEqual((metrics.tp, metrics.fp, metrics.fn, metrics.tn), (1, 1, 0, 0))
        strict = PixelConfusion()
        strict.update(logits, target, valid, threshold=0.6)
        self.assertEqual((strict.tp, strict.fp, strict.fn, strict.tn), (0, 0, 1, 1))

    def test_empty_and_all_negative_pv_ratios_are_null(self) -> None:
        empty = PixelConfusion().to_dict()
        for key in ("iou", "dice", "precision", "recall", "pixel_accuracy"):
            self.assertIsNone(empty[key])
        target = torch.zeros((1, 1, 2, 2))
        metrics = PixelConfusion()
        metrics.update(torch.full_like(target, -1000), target, torch.ones_like(target, dtype=torch.bool))
        result = metrics.to_dict()
        self.assertEqual(result["tn"], 4)
        self.assertEqual(result["pixel_accuracy"], 1.0)
        for key in ("iou", "dice", "precision", "recall"):
            self.assertIsNone(result[key])
        json.dumps(result, allow_nan=False)

    def test_all_ignored_and_arbitrary_ignored_labels_add_no_counts(self) -> None:
        logits, target, valid = self.fixture()
        first, second = PixelConfusion(), PixelConfusion()
        first.update(logits, target, valid)
        target[~valid] = float("nan")
        second.update(logits, target, valid)
        self.assertEqual(first.to_dict(), second.to_dict())
        second.update(logits, torch.full_like(target, float("inf")), torch.zeros_like(valid))
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_metrics_do_not_mutate_inputs_or_autograd_state(self) -> None:
        logits, target, valid = self.fixture()
        logits.requires_grad_()
        original = logits.detach().clone()
        metrics = PixelConfusion()
        self.assertTrue(torch.is_grad_enabled())
        metrics.update(logits, target, valid)
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(logits.requires_grad)
        self.assertIsNone(logits.grad)
        self.assertTrue(torch.equal(logits.detach(), original))
        self.assertTrue(all(type(getattr(metrics, key)) is int for key in ("tp", "fp", "fn", "tn", "samples")))

    def test_invalid_values_shapes_and_thresholds_fail_without_partial_update(self) -> None:
        logits, target, valid = self.fixture()
        metrics = PixelConfusion()
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5", 10**1000):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(TrainingDataError, "threshold"):
                metrics.update(logits, target, valid, threshold=threshold)
        for value in (2, float("nan"), float("inf")):
            bad = target.clone()
            bad[valid] = value
            with self.subTest(value=value), self.assertRaisesRegex(TrainingDataError, "finite binary"):
                metrics.update(logits, bad, valid)
        with self.assertRaisesRegex(TrainingDataError, "finite"):
            metrics.update(torch.full_like(logits, float("inf")), target, valid)
        with self.assertRaisesRegex(TrainingDataError, "shapes"):
            metrics.update(logits, target[:, :, :1], valid)
        with self.assertRaisesRegex(TrainingDataError, "boolean"):
            metrics.update(logits, target, valid.float())
        self.assertEqual(metrics.to_dict()["valid_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
