# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for physical module grid fitting and CAD layout packing."""

import unittest
import numpy as np

from pvextent.cad.packing import fit_module_grid, point_in_polygon
from pvextent.cad.vectorize import vectorize_mask


class TestPacking(unittest.TestCase):
    def test_point_in_polygon(self):
        # 10x10 square: (0,0) to (10,10)
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertTrue(point_in_polygon(5.0, 5.0, square))
        self.assertFalse(point_in_polygon(15.0, 5.0, square))
        self.assertFalse(point_in_polygon(-1.0, 5.0, square))

    def test_fit_module_grid(self):
        # 10m x 4m rectangle at GSD 0.10m -> 100px x 40px
        # (100 px, 40 px) with 0 degree rotation
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (0.0, 40.0)]
        res = fit_module_grid(poly, dominant_angle_deg=0.0, gsd_m=0.10)

        self.assertIn("module_count", res)
        self.assertIn("orientation", res)
        self.assertIn("modules_px", res)
        # In a 10m x 4m area (40 m²), standard modules 1.76m x 1.04m should fit ~15 to 20 panels
        self.assertGreaterEqual(res["module_count"], 12)
        self.assertLessEqual(res["module_count"], 24)
        self.assertEqual(len(res["modules_px"]), res["module_count"])
        # Each module has 4 corner vertices
        self.assertEqual(len(res["modules_px"][0]), 4)

    def test_vectorize_mask_integration(self):
        # Create a mask with a 60x30 rectangular array (6m x 3m = 18 m² at GSD 0.1)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:80] = 1

        geojson = vectorize_mask(mask, gsd_m=0.10, min_area_m2=1.5, fit_modules=True)
        self.assertEqual(len(geojson["features"]), 1)
        feat = geojson["features"][0]
        props = feat["properties"]

        self.assertIn("fitted_module_count", props)
        self.assertIn("fitted_capacity_kwp", props)
        self.assertIn("fitted_modules_px", props)
        self.assertGreater(props["fitted_module_count"], 5)
        self.assertIn("total_fitted_modules", geojson["summary"])
        self.assertEqual(geojson["summary"]["total_fitted_modules"], props["fitted_module_count"])


if __name__ == "__main__":
    unittest.main()
