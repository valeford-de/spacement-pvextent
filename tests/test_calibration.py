# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for probability calibration and threshold optimization."""

import unittest
import numpy as np
import torch

from pvextent.inference.calibration import (
    compute_calibration_curve,
    compute_nll_and_brier,
    find_optimal_thresholds,
    fit_platt_scaling,
    fit_temperature_scaling,
)


class TestCalibration(unittest.TestCase):
    def test_nll_and_brier(self):
        p = np.array([0.9, 0.1])
        y = np.array([1, 0])
        res = compute_nll_and_brier(p, y)
        self.assertAlmostEqual(res["brier"], 0.01, places=3)
        self.assertGreater(res["nll"], 0.0)

    def test_ece_perfect_calibration(self):
        # 100 samples with prob 0.8 where exactly 80 are positive
        p = np.full(100, 0.8)
        y = np.array([1] * 80 + [0] * 20)
        res = compute_calibration_curve(p, y, n_bins=10)
        self.assertAlmostEqual(res["ece"], 0.0, places=4)

    def test_ece_severe_miscalibration(self):
        # Model predicts 0.99 for balanced 50/50 outcomes
        p = np.full(100, 0.99)
        y = np.array([1] * 50 + [0] * 50)
        res = compute_calibration_curve(p, y, n_bins=10)
        self.assertGreater(res["ece"], 0.4)

    def test_fit_temperature_scaling_overconfident(self):
        # Logits +10.0 and -10.0 for balanced labels (extreme overconfidence)
        logits = torch.tensor([10.0] * 50 + [-10.0] * 50)
        targets = torch.tensor([1.0] * 25 + [0.0] * 25 + [1.0] * 25 + [0.0] * 25)
        # Optimal temperature should scale down logits (T >> 1)
        t = fit_temperature_scaling(logits, targets)
        self.assertGreater(t, 1.5)

    def test_fit_platt_scaling(self):
        logits = torch.randn(200)
        # True probabilities driven by 2.0 * z + 0.5
        true_p = torch.sigmoid(2.0 * logits + 0.5)
        targets = (torch.rand(200) < true_p).float()
        a, b = fit_platt_scaling(logits, targets)
        self.assertGreater(a, 0.5)
        self.assertTrue(np.isfinite(b))

    def test_find_optimal_thresholds(self):
        # Generate probabilities where positive targets follow a step function at 0.4
        p = np.linspace(0.0, 1.0, 100)
        y = (p >= 0.4)
        res = find_optimal_thresholds(p, y, n_steps=50)
        self.assertIn("optimal_threshold_iou", res)
        self.assertIn("optimal_threshold_f1", res)
        self.assertGreaterEqual(res["max_iou"], 0.95)
        # Optimal threshold should be around 0.4
        self.assertAlmostEqual(res["optimal_threshold_iou"], 0.4, delta=0.05)


if __name__ == "__main__":
    unittest.main()
