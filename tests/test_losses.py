# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for binary segmentation losses (BCE-Dice and Focal-Lovasz)."""

from __future__ import annotations

import unittest
import torch

from pvextent.training.losses import masked_bce_dice, masked_focal_lovasz, TrainingDataError


class LossesTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        self.target = (torch.rand(2, 1, 32, 32) > 0.5).float()
        self.valid = torch.ones(2, 1, 32, 32, dtype=torch.bool)

    def test_masked_bce_dice_finite_gradients(self) -> None:
        res = masked_bce_dice(self.logits, self.target, self.valid)
        self.assertTrue(torch.isfinite(res["loss"]))
        res["loss"].backward()
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_masked_focal_lovasz_finite_gradients(self) -> None:
        res = masked_focal_lovasz(self.logits, self.target, self.valid)
        self.assertTrue(torch.isfinite(res["loss"]))
        self.assertTrue(torch.isfinite(res["focal_loss"]))
        self.assertTrue(torch.isfinite(res["lovasz_loss"]))
        self.assertEqual(res["valid_pixels"], 2 * 32 * 32)
        res["loss"].backward()
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_masked_focal_lovasz_excludes_ignored_pixels(self) -> None:
        # Construct valid mask with only half pixels valid
        valid = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
        valid[:, :, :16, :] = True
        
        # Corrupt logits and target in invalid region
        logits_corrupt = self.logits.clone().detach().requires_grad_(True)
        with torch.no_grad():
            logits_corrupt[:, :, 16:, :] = 999.0
            target_corrupt = self.target.clone()
            target_corrupt[:, :, 16:, :] = 0.0

        res1 = masked_focal_lovasz(self.logits, self.target, valid)
        res2 = masked_focal_lovasz(logits_corrupt, target_corrupt, valid)
        self.assertAlmostEqual(res1["loss"].item(), res2["loss"].item(), places=5)

    def test_rejects_empty_valid_mask(self) -> None:
        valid_empty = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
        with self.assertRaises(TrainingDataError):
            masked_focal_lovasz(self.logits, self.target, valid_empty)

    def test_rejects_non_finite_logits(self) -> None:
        nan_logits = self.logits.clone()
        nan_logits[0, 0, 0, 0] = float("nan")
        with self.assertRaises(TrainingDataError):
            masked_focal_lovasz(nan_logits, self.target, self.valid)


if __name__ == "__main__":
    unittest.main()
