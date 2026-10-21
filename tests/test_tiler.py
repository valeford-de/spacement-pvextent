# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sliding-window tiled inference and 2D Hann window blending."""

import unittest
import numpy as np
import torch

from pvextent.inference.tiler import (
    compute_tile_slices,
    generate_hann_window,
    pad_to_stride,
    predict_tiled,
    unpad,
)


class DummyModel(torch.nn.Module):
    """Dummy segmentation model returning sigmoid probability based on red channel."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits = 2.0 * red_channel - 1.0
        return x[:, 0:1, :, :] * 2.0 - 1.0


class TestTiler(unittest.TestCase):
    def test_hann_window_geometry(self):
        h, w = 100, 100
        win = generate_hann_window(h, w)
        self.assertEqual(win.shape, (h, w))
        # Center should be close to 1.0
        center_val = win[50, 50]
        self.assertGreater(center_val, 0.95)
        # Corners should be small
        corner_val = win[0, 0]
        self.assertLess(corner_val, 0.05)
        # Symmetry
        np.testing.assert_allclose(win, np.flip(win, axis=0), atol=1e-4)
        np.testing.assert_allclose(win, np.flip(win, axis=1), atol=1e-4)

    def test_compute_tile_slices(self):
        # 1000px with tile 400 and stride 200 (50% overlap)
        slices = compute_tile_slices(total_size=1000, tile_size=400, stride=200)
        self.assertGreater(len(slices), 1)
        self.assertEqual(slices[0].start, 0)
        self.assertEqual(slices[0].stop, 400)
        self.assertEqual(slices[-1].stop, 1000)
        # Ensure complete coverage
        covered = np.zeros(1000, dtype=bool)
        for s in slices:
            covered[s] = True
        self.assertTrue(np.all(covered))

    def test_pad_to_stride_and_unpad(self):
        # Non-multiple of 32
        x = torch.randn(1, 3, 350, 420)
        padded, pad_spec = pad_to_stride(x, stride=32)
        h_pad, w_pad = padded.shape[-2:]
        self.assertEqual(h_pad % 32, 0)
        self.assertEqual(w_pad % 32, 0)
        self.assertEqual(h_pad, 352)
        self.assertEqual(w_pad, 448)

        restored = unpad(padded, pad_spec)
        self.assertEqual(restored.shape, x.shape)
        torch.testing.assert_close(restored, x)

    def test_predict_tiled_arbitrary_resolution(self):
        model = DummyModel()
        # Large arbitrary resolution image (650 x 750)
        img = np.random.uniform(0.0, 1.0, (650, 750, 3)).astype(np.float32)

        res = predict_tiled(
            model=model,
            image=img,
            tile_size=400,
            overlap=0.50,
            tta_mode="none",
            device="cpu",
        )

        self.assertIn("probabilities", res)
        self.assertIn("uncertainty", res)
        probs = res["probabilities"]
        self.assertEqual(probs.shape, (650, 750))
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertGreater(res["tiles_processed"], 1)

    def test_predict_tiled_small_image(self):
        model = DummyModel()
        # Small non-multiple of 32 (120 x 140)
        img = np.random.uniform(0.0, 1.0, (120, 140, 3)).astype(np.float32)

        res = predict_tiled(
            model=model,
            image=img,
            tile_size=400,
            device="cpu",
        )
        self.assertEqual(res["probabilities"].shape, (120, 140))
        self.assertEqual(res["tiles_processed"], 1)


if __name__ == "__main__":
    unittest.main()
