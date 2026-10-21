# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CAD-grade geospatial vectorization and rectilinear regularization."""

import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from pvextent.cad.metrics import (
    calculate_pv_metrics,
    polygon_area_pixels,
    polygon_centroid,
    polygon_perimeter_pixels,
)
from pvextent.cad.regularize import (
    detect_dominant_orientation,
    minimum_bounding_box,
    regularize_manhattan,
    simplify_douglas_peucker,
)
from pvextent.cad.vectorize import (
    marching_squares_contours,
    render_cad_overlay,
    transform_pixel_to_geo,
    vectorize_mask,
)


class TestCadMetrics(unittest.TestCase):
    def test_shoelace_rectangle_area(self):
        rect = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
        self.assertAlmostEqual(polygon_area_pixels(rect), 200.0)

    def test_perimeter(self):
        rect = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
        self.assertAlmostEqual(polygon_perimeter_pixels(rect), 60.0)

    def test_centroid(self):
        rect = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]
        cx, cy = polygon_centroid(rect)
        self.assertAlmostEqual(cx, 10.0)
        self.assertAlmostEqual(cy, 5.0)

    def test_pv_metrics_calculation(self):
        # 1000 pixels at 0.1m GSD -> 10.0 m²
        # At 20.5% efficiency -> 2.05 kWp
        # At 400W per module -> 2.05 / 0.400 = 5.125 -> 5 modules
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)]
        metrics = calculate_pv_metrics(poly, gsd_m=0.1, module_efficiency=0.205)
        self.assertEqual(metrics["area_pixels"], 1000.0)
        self.assertEqual(metrics["area_m2"], 10.0)
        self.assertEqual(metrics["estimated_capacity_kwp"], 2.05)
        self.assertEqual(metrics["estimated_module_count"], 5)


class TestRegularization(unittest.TestCase):
    def test_douglas_peucker_collinear(self):
        # 10 points on a straight line should reduce to start and end
        pts = [(float(i), 0.0) for i in range(11)] + [(10.0, 10.0), (0.0, 10.0)]
        simplified = simplify_douglas_peucker(pts, tolerance=0.1)
        # Should be a clean 4-corner rectangle
        self.assertEqual(len(simplified), 4)

    def test_dominant_orientation_axis_aligned(self):
        rect = [(0.0, 0.0), (50.0, 0.0), (50.0, 20.0), (0.0, 20.0)]
        angle_rad, azimuth_deg = detect_dominant_orientation(rect)
        self.assertAlmostEqual(angle_rad, 0.0, places=3)
        self.assertIn(round(azimuth_deg, 1), [90.0, 270.0, 180.0, 0.0])

    def test_dominant_orientation_rotated(self):
        # Rotate rectangle by 30 degrees (pi/6)
        theta = math.pi / 6.0
        c, s = math.cos(theta), math.sin(theta)
        rect = [(0.0, 0.0), (40.0 * c, 40.0 * s), (40.0 * c - 20.0 * s, 40.0 * s + 20.0 * c), (-20.0 * s, 20.0 * c)]
        angle_rad, _ = detect_dominant_orientation(rect)
        self.assertAlmostEqual(angle_rad, theta, places=2)

    def test_minimum_bounding_box(self):
        # Rotated rectangle: 30 x 10
        theta = 0.5
        c, s = math.cos(theta), math.sin(theta)
        rect = [(0.0, 0.0), (30.0 * c, 30.0 * s), (30.0 * c - 10.0 * s, 30.0 * s + 10.0 * c), (-10.0 * s, 10.0 * c)]
        obb = minimum_bounding_box(rect)
        self.assertAlmostEqual(obb["area"], 300.0, places=1)
        self.assertAlmostEqual(obb["aspect_ratio"], 3.0, places=1)

    def test_manhattan_regularization(self):
        # Jittered rectangle: slight perturbation from 90 degrees
        rect = [
            (0.0, 0.2), (19.8, -0.1), (20.1, 9.9), (0.2, 10.2)
        ]
        reg = regularize_manhattan(rect, tolerance_deg=25.0)
        self.assertGreaterEqual(len(reg), 4)
        area = polygon_area_pixels(reg)
        self.assertAlmostEqual(area, 200.0, delta=10.0)


class TestVectorize(unittest.TestCase):
    def test_marching_squares_synthetic(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        # Put a 20x15 box in the center
        mask[15:30, 10:30] = 1
        rings = marching_squares_contours(mask, min_area_px=50.0)
        self.assertEqual(len(rings), 1)
        # 20 * 15 = 300 pixels
        area = polygon_area_pixels(rings[0])
        self.assertAlmostEqual(area, 300.0, delta=15.0)

    def test_transform_pixel_to_geo(self):
        # DOP10 transform
        transform = [0.1, 0, 400000.0, 0, -0.1, 5700000.0]
        pts = [(10.0, 20.0)]
        geo = transform_pixel_to_geo(pts, transform)
        # X = 400000 + 10 * 0.1 = 400001.0
        # Y = 5700000 + 20 * -0.1 = 5699998.0
        self.assertEqual(geo[0], (400001.0, 5699998.0))

    def test_vectorize_mask_geojson(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:80] = 1  # 40 x 60 = 2400 px
        georef = {
            "crs": "EPSG:25832",
            "transform": [0.1, 0, 500000.0, 0, -0.1, 5700000.0]
        }
        geojson = vectorize_mask(mask, georeference=georef, gsd_m=0.1)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 1)
        props = geojson["features"][0]["properties"]
        self.assertGreater(props["area_m2"], 20.0)
        self.assertGreater(props["estimated_capacity_kwp"], 4.0)
        self.assertTrue(props["is_regularized"])

    def test_render_cad_overlay(self):
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:80] = 1
        geojson = vectorize_mask(mask)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_overlay.png"
            render_cad_overlay(img, geojson, out_path)
            self.assertTrue(out_path.exists())
            with Image.open(out_path) as rendered:
                self.assertEqual(rendered.size, (100, 100))


if __name__ == "__main__":
    unittest.main()
