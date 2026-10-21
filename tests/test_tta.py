# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Test-Time Augmentation (TTA)."""

import unittest
import torch

from pvextent.inference.tta import (
    apply_tta_transforms,
    invert_tta_prediction,
    predict_with_tta,
)


class DummyModel(torch.nn.Module):
    """Dummy model returning input channel 0 as logits."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:1, :, :] * 2.0


class TestTTA(unittest.TestCase):
    def test_roundtrip_all_transforms(self):
        # Create an asymmetric test image (C=1, H=4, W=4)
        x = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]]
        ])

        batch, names = apply_tta_transforms(x, mode="d4")
        self.assertEqual(len(names), 8)
        self.assertEqual(batch.shape, (8, 1, 4, 4))

        for i, name in enumerate(names):
            t = batch[i:i + 1]
            inv = invert_tta_prediction(t, name)
            diff = torch.max(torch.abs(inv - x)).item()
            self.assertAlmostEqual(diff, 0.0, places=5, msg=f"Transform {name} did not roundtrip.")

    def test_modes(self):
        x = torch.zeros((1, 3, 16, 16))
        for mode, expected_k in [("none", 1), ("flips", 4), ("rot4", 4), ("d4", 8)]:
            b, names = apply_tta_transforms(x, mode=mode)
            self.assertEqual(len(names), expected_k)
            self.assertEqual(b.shape[0], expected_k)

    def test_predict_with_tta_execution(self):
        model = DummyModel()
        x = torch.randn((3, 32, 32))

        res = predict_with_tta(model, x, mode="d4")
        self.assertIn("probabilities", res)
        self.assertIn("uncertainty", res)
        self.assertEqual(res["transforms_used"], 8)
        self.assertEqual(res["probabilities"].shape, (32, 32))
        self.assertEqual(res["uncertainty"].shape, (32, 32))
        self.assertTrue(torch.all(res["probabilities"] >= 0.0))
        self.assertTrue(torch.all(res["probabilities"] <= 1.0))
        self.assertTrue(torch.all(res["uncertainty"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
