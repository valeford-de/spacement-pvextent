# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Multispectral 4-Band Analysis, NIR/CIR composites, and Spectral PV Signatures."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from pvextent.cad.spectral import (
    NDVI_VEGETATION_THRESHOLD,
    SILICON_PV_NDVI_MAX,
    calculate_ndre,
    calculate_ndvi,
    calculate_pvsi,
    compose_cir_composite,
    expand_conv1_for_nir,
    extract_polygon_spectral_stats,
)
from pvextent.cad.vectorize import vectorize_mask


class TestMultispectralAnalysis(unittest.TestCase):
    def test_calculate_ndvi(self):
        # Living vegetation: high NIR, low red -> high positive NDVI
        nir_veg = np.array([[0.80]], dtype=np.float32)
        red_veg = np.array([[0.10]], dtype=np.float32)
        ndvi_veg = calculate_ndvi(nir_veg, red_veg)
        self.assertAlmostEqual(float(ndvi_veg[0, 0]), (0.8 - 0.1) / (0.8 + 0.1), places=3)
        self.assertGreater(float(ndvi_veg[0, 0]), NDVI_VEGETATION_THRESHOLD)

        # Crystalline silicon PV: moderate NIR, low red -> low positive NDVI
        nir_pv = np.array([[0.15]], dtype=np.float32)
        red_pv = np.array([[0.12]], dtype=np.float32)
        ndvi_pv = calculate_ndvi(nir_pv, red_pv)
        self.assertAlmostEqual(float(ndvi_pv[0, 0]), (0.15 - 0.12) / (0.15 + 0.12), places=3)
        self.assertLessEqual(float(ndvi_pv[0, 0]), SILICON_PV_NDVI_MAX)

        # Water: extreme NIR absorption -> negative NDVI
        nir_water = np.array([[0.02]], dtype=np.float32)
        red_water = np.array([[0.08]], dtype=np.float32)
        ndvi_water = calculate_ndvi(nir_water, red_water)
        self.assertLess(float(ndvi_water[0, 0]), 0.0)

    def test_calculate_pvsi(self):
        nir = np.array([[0.20]], dtype=np.float32)
        blue = np.array([[0.08]], dtype=np.float32)
        pvsi = calculate_pvsi(nir, blue)
        self.assertAlmostEqual(float(pvsi[0, 0]), (0.20 - 0.08) / (0.20 + 0.08), places=3)

    def test_calculate_ndre(self):
        nir = np.array([[0.60]], dtype=np.float32)
        re = np.array([[0.20]], dtype=np.float32)
        ndre = calculate_ndre(nir, re)
        self.assertAlmostEqual(float(ndre[0, 0]), (0.60 - 0.20) / (0.60 + 0.20), places=3)

    def test_compose_cir_composite(self):
        red = np.zeros((20, 20), dtype=np.uint8)
        green = np.zeros((20, 20), dtype=np.uint8)
        nir = np.full((20, 20), 200, dtype=np.uint8)

        cir = compose_cir_composite(red, green, nir)
        self.assertEqual(cir.size, (20, 20))
        arr = np.array(cir)
        # Channel 0 is NIR (200), Channel 1 is Red (0), Channel 2 is Green (0)
        self.assertEqual(arr[0, 0, 0], 200)
        self.assertEqual(arr[0, 0, 1], 0)
        self.assertEqual(arr[0, 0, 2], 0)

    def test_extract_polygon_spectral_stats_silicon_pv(self):
        # Create synthetic 100x100 scene with a silicon solar panel array
        # Silicon panel: dark visible (R=0.08, G=0.09, B=0.12), moderate NIR (0.16)
        rgb = np.full((100, 100, 3), [20, 23, 30], dtype=np.uint8)
        nir = np.full((100, 100), 40, dtype=np.uint8)

        poly = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        stats = extract_polygon_spectral_stats(poly, rgb, nir)

        self.assertTrue(stats["multispectral_nir_available"])
        self.assertEqual(stats["spectral_verdict"], "silicon_photovoltaic_confirmed")
        self.assertFalse(stats["is_spectral_false_positive"])
        self.assertLessEqual(stats["mean_ndvi"], SILICON_PV_NDVI_MAX)
        self.assertGreaterEqual(stats["nir_red_ratio"], 0.70)

    def test_extract_polygon_spectral_stats_green_roof_veto(self):
        # Green roof (Dachbegrünung): R=0.10, G=0.25, B=0.08, extreme NIR=0.75
        rgb = np.full((100, 100, 3), [25, 64, 20], dtype=np.uint8)
        nir = np.full((100, 100), 190, dtype=np.uint8)

        poly = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        stats = extract_polygon_spectral_stats(poly, rgb, nir)

        self.assertEqual(stats["spectral_verdict"], "vegetated_roof_veto")
        self.assertTrue(stats["is_spectral_false_positive"])
        self.assertGreater(stats["mean_ndvi"], NDVI_VEGETATION_THRESHOLD)

    def test_extract_polygon_spectral_stats_water_puddle_veto(self):
        # Water puddle: low visible, extreme NIR absorption < 0.04
        rgb = np.full((100, 100, 3), [10, 10, 12], dtype=np.uint8)
        nir = np.full((100, 100), 5, dtype=np.uint8)  # 5/255 = 0.0196 < 0.04

        poly = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        stats = extract_polygon_spectral_stats(poly, rgb, nir)

        self.assertEqual(stats["spectral_verdict"], "water_puddle_veto")
        self.assertTrue(stats["is_spectral_false_positive"])

    def test_vectorize_mask_with_nir_and_veto(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Array 1: Silicon PV at [10..40, 10..40]
        mask[10:40, 10:40] = 1
        # Array 2: Green roof false positive at [60..90, 60..90]
        mask[60:90, 60:90] = 1

        rgb = np.full((100, 100, 3), [50, 50, 50], dtype=np.uint8)
        nir = np.full((100, 100), 50, dtype=np.uint8)

        # Set spectral properties for Array 1 (Silicon)
        rgb[10:40, 10:40] = [20, 20, 25]
        nir[10:40, 10:40] = 35

        # Set spectral properties for Array 2 (Vegetation / Dachbegrünung)
        rgb[60:90, 60:90] = [25, 60, 20]
        nir[60:90, 60:90] = 200

        # Run 1: Without veto -> both detected, Array 2 tagged review_recommended
        res_no_veto = vectorize_mask(mask, rgb=rgb, nir=nir, veto_vegetated_roofs=False)
        self.assertEqual(len(res_no_veto["features"]), 2)
        self.assertTrue(res_no_veto["summary"]["multispectral_nir_enabled"])

        # Run 2: With veto -> vegetated array filtered out completely
        res_with_veto = vectorize_mask(mask, rgb=rgb, nir=nir, veto_vegetated_roofs=True)
        self.assertEqual(len(res_with_veto["features"]), 1)
        self.assertEqual(res_with_veto["features"][0]["properties"]["spectral_verdict"], "silicon_photovoltaic_confirmed")

    def test_expand_conv1_for_nir(self):
        try:
            import torch
            # Mock 3-channel Conv1 weight: (64, 3, 7, 7)
            w3 = torch.randn(64, 3, 7, 7)
            state = {"conv1.weight": w3}
            expanded = expand_conv1_for_nir(None, state)
            w4 = expanded["conv1.weight"]
            self.assertEqual(w4.shape, (64, 4, 7, 7))
            # Verify 4th channel is average of channel 0 (Red) and channel 1 (Green)
            expected_nir = 0.5 * (w3[:, 0:1, :, :] + w3[:, 1:2, :, :])
            self.assertTrue(torch.allclose(w4[:, 3:4, :, :], expected_nir))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
