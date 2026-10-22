# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for 3D roof elevation fitting, tilt/azimuth extraction, and solar yield modeling."""

import math
import unittest
import numpy as np

from pvextent.cad.elevation import (
    azimuth_to_cardinal,
    calculate_3d_surface_area,
    calculate_annual_solar_yield,
    fit_roof_plane_from_dsm,
)
from pvextent.cad.packing import fit_module_grid
from pvextent.cad.vectorize import vectorize_mask


class TestElevationAndYield(unittest.TestCase):
    def test_calculate_3d_surface_area(self):
        # 0° flat roof: no expansion
        res_0 = calculate_3d_surface_area(100.0, 0.0)
        self.assertEqual(res_0["projected_area_m2"], 100.0)
        self.assertEqual(res_0["true_3d_area_m2"], 100.0)
        self.assertEqual(res_0["area_expansion_factor"], 1.0)

        # 30° pitched roof: 1 / cos(30°) = 1.1547 (+15.47%)
        res_30 = calculate_3d_surface_area(100.0, 30.0)
        self.assertAlmostEqual(res_30["area_expansion_factor"], 1.0 / math.cos(math.radians(30)), places=3)
        self.assertAlmostEqual(res_30["true_3d_area_m2"], 115.47, places=1)

        # 45° pitched roof: 1 / cos(45°) = 1.4142 (+41.42%)
        res_45 = calculate_3d_surface_area(100.0, 45.0)
        self.assertAlmostEqual(res_45["area_expansion_factor"], math.sqrt(2), places=3)
        self.assertAlmostEqual(res_45["true_3d_area_m2"], 141.42, places=1)

    def test_azimuth_to_cardinal(self):
        # Flat roof
        self.assertEqual(azimuth_to_cardinal(180.0, tilt_deg=2.0), "FLAT")
        # Cardinals
        self.assertEqual(azimuth_to_cardinal(0.0, tilt_deg=30.0), "N")
        self.assertEqual(azimuth_to_cardinal(90.0, tilt_deg=30.0), "E")
        self.assertEqual(azimuth_to_cardinal(180.0, tilt_deg=30.0), "S")
        self.assertEqual(azimuth_to_cardinal(270.0, tilt_deg=30.0), "W")
        self.assertEqual(azimuth_to_cardinal(225.0, tilt_deg=30.0), "SW")
        self.assertEqual(azimuth_to_cardinal(135.0, tilt_deg=30.0), "SE")

    def test_calculate_annual_solar_yield(self):
        # 10 kWp South facing at 35°
        south_res = calculate_annual_solar_yield(
            capacity_kwp=10.0,
            tilt_deg=35.0,
            azimuth_deg=180.0,
            ghi_kwh_m2=1050.0,
        )
        self.assertEqual(south_res["azimuth_cardinal"], "S")
        self.assertGreater(south_res["specific_annual_yield_kwh_per_kwp"], 900.0)
        self.assertLess(south_res["specific_annual_yield_kwh_per_kwp"], 1200.0)
        self.assertGreater(south_res["estimated_annual_yield_mwh"], 9.0)
        self.assertGreater(south_res["annual_co2_savings_tons"], 3.0)

        # 10 kWp North facing at 35°
        north_res = calculate_annual_solar_yield(
            capacity_kwp=10.0,
            tilt_deg=35.0,
            azimuth_deg=0.0,
            ghi_kwh_m2=1050.0,
        )
        self.assertEqual(north_res["azimuth_cardinal"], "N")
        # South should yield significantly more than North
        self.assertGreater(south_res["estimated_annual_yield_kwh"], north_res["estimated_annual_yield_kwh"] * 1.3)

        # East and West facing at 30° should have nearly identical yields
        east_res = calculate_annual_solar_yield(10.0, 30.0, 90.0)
        west_res = calculate_annual_solar_yield(10.0, 30.0, 270.0)
        self.assertAlmostEqual(east_res["estimated_annual_yield_kwh"], west_res["estimated_annual_yield_kwh"], places=0)

    def test_fit_roof_plane_from_dsm_south_slope(self):
        # Create a 100x100 DSM grid at GSD 0.10m (10m x 10m area)
        # Perfect 30° South-facing roof:
        # Z increases going North (decreasing y in image coordinates)
        # slope = tan(30°) = 0.57735
        # Z(y, x) = Z0 - slope * (y * gsd)
        gsd = 0.10
        slope_30 = math.tan(math.radians(30.0))
        dsm = np.zeros((100, 100), dtype=np.float32)
        for y in range(100):
            dsm[y, :] = 100.0 - slope_30 * (y * gsd)

        # Define roof polygon from (20, 20) to (80, 80)
        poly = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        fit = fit_roof_plane_from_dsm(dsm, poly, gsd_m=gsd)

        self.assertTrue(fit["is_valid_3d_fit"])
        self.assertAlmostEqual(fit["tilt_deg"], 30.0, delta=1.0)
        self.assertAlmostEqual(fit["azimuth_deg"], 180.0, delta=2.0)
        self.assertEqual(fit["azimuth_cardinal"], "S")
        self.assertGreater(fit["fit_r2"], 0.99)
        self.assertLess(fit["fit_rmse_m"], 0.01)

    def test_fit_roof_plane_from_dsm_east_slope(self):
        # Perfect 45° East-facing roof:
        # Z decreases going East (increasing x)
        # slope = tan(45°) = 1.0
        # Z(y, x) = 50.0 - 1.0 * (x * gsd)
        gsd = 0.10
        dsm = np.zeros((100, 100), dtype=np.float32)
        for x in range(100):
            dsm[:, x] = 50.0 - 1.0 * (x * gsd)

        poly = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
        fit = fit_roof_plane_from_dsm(dsm, poly, gsd_m=gsd)

        self.assertTrue(fit["is_valid_3d_fit"])
        self.assertAlmostEqual(fit["tilt_deg"], 45.0, delta=1.0)
        self.assertAlmostEqual(fit["azimuth_deg"], 90.0, delta=2.0)
        self.assertEqual(fit["azimuth_cardinal"], "E")
        self.assertGreater(fit["fit_r2"], 0.99)

    def test_sloped_module_grid_packing(self):
        # 10m x 4m rectangle (100px x 40px) at GSD 0.10m
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 40.0), (0.0, 40.0)]

        # Flat roof (0°)
        flat = fit_module_grid(poly, dominant_angle_deg=0.0, gsd_m=0.10, tilt_deg=0.0)

        # 45° sloped roof: slope allows more module rows along slope axis v
        sloped_45 = fit_module_grid(poly, dominant_angle_deg=0.0, gsd_m=0.10, tilt_deg=45.0)

        self.assertGreaterEqual(sloped_45["module_count"], flat["module_count"])
        self.assertGreaterEqual(sloped_45["total_capacity_kwp"], flat["total_capacity_kwp"])

    def test_vectorize_mask_with_dsm_and_yield(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:80] = 1  # 4m x 6m PV array (24 m²)

        # 35° South slope DSM
        gsd = 0.10
        slope_35 = math.tan(math.radians(35.0))
        dsm = np.zeros((100, 100), dtype=np.float32)
        for y in range(100):
            dsm[y, :] = 80.0 - slope_35 * (y * gsd)

        geojson = vectorize_mask(mask, gsd_m=gsd, min_area_m2=1.5, dsm=dsm, fit_modules=True)
        self.assertEqual(len(geojson["features"]), 1)
        props = geojson["features"][0]["properties"]

        # Check 3D geometry metrics
        self.assertIn("area_true_3d_m2", props)
        self.assertIn("area_expansion_factor", props)
        self.assertIn("roof_tilt_deg", props)
        self.assertIn("roof_azimuth_deg", props)
        self.assertIn("roof_azimuth_cardinal", props)

        self.assertAlmostEqual(props["roof_tilt_deg"], 35.0, delta=1.5)
        self.assertEqual(props["roof_azimuth_cardinal"], "S")
        self.assertGreater(props["area_true_3d_m2"], props["area_m2"])
        self.assertAlmostEqual(props["area_expansion_factor"], 1.0 / math.cos(math.radians(35.0)), places=2)

        # Check energy yield and carbon savings
        self.assertIn("estimated_annual_yield_kwh", props)
        self.assertIn("estimated_annual_yield_mwh", props)
        self.assertIn("annual_co2_savings_tons", props)
        self.assertGreater(props["estimated_annual_yield_kwh"], 1000.0)
        self.assertGreater(props["annual_co2_savings_tons"], 0.5)

        # Check GeoJSON summary
        summary = geojson["summary"]
        self.assertIn("total_true_3d_area_m2", summary)
        self.assertIn("total_annual_yield_mwh", summary)
        self.assertIn("total_annual_co2_savings_tons", summary)
        self.assertIn("mean_roof_tilt_deg", summary)
        self.assertGreater(summary["total_true_3d_area_m2"], summary["total_pv_area_m2"])


if __name__ == "__main__":
    unittest.main()
