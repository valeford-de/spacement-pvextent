# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Multi-Temporal (Bi-Temporal) 4D Change Detection and CAD Visualizer."""

from datetime import date
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from pvextent.cad.temporal import (
    CHANGE_TYPE_DECOMMISSIONED,
    CHANGE_TYPE_EXPANSION,
    CHANGE_TYPE_NEW,
    CHANGE_TYPE_STABLE,
    calculate_temporal_deltas,
    detect_temporal_changes,
    polygon_iou_raster,
    render_temporal_cad_overlay,
)


class TestTemporalChangeDetection(unittest.TestCase):
    def test_polygon_iou_raster(self):
        # Degenerate
        self.assertEqual(polygon_iou_raster([(0, 0), (10, 0)], [(0, 0), (10, 10)]), 0.0)

        # Identical boxes
        box_a = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
        self.assertAlmostEqual(polygon_iou_raster(box_a, box_a), 1.0, places=2)

        # Non-overlapping boxes
        box_b = [(50.0, 50.0), (70.0, 50.0), (70.0, 70.0), (50.0, 70.0)]
        self.assertEqual(polygon_iou_raster(box_a, box_b), 0.0)

        # Partially overlapping boxes:
        # box_a: [10..30] x [10..30], area = 20 x 20 = 400
        # box_c: [20..40] x [10..30], area = 20 x 20 = 400
        # intersection: [20..30] x [10..30], area = 10 x 20 = 200
        # union = 400 + 400 - 200 = 600 -> IoU = 200 / 600 = 0.333
        box_c = [(20.0, 10.0), (40.0, 10.0), (40.0, 30.0), (20.0, 30.0)]
        iou = polygon_iou_raster(box_a, box_c)
        self.assertAlmostEqual(iou, 0.333, delta=0.03)

    def test_calculate_temporal_deltas(self):
        t1_props = {
            "area_m2": 20.0,
            "area_true_3d_m2": 23.09,
            "fitted_capacity_kwp": 4.10,
            "fitted_module_count": 10,
            "estimated_annual_yield_mwh": 4.31,
            "annual_co2_savings_tons": 1.64,
        }
        t2_props = {
            "area_m2": 35.0,
            "area_true_3d_m2": 40.41,
            "fitted_capacity_kwp": 7.18,
            "fitted_module_count": 18,
            "estimated_annual_yield_mwh": 7.54,
            "annual_co2_savings_tons": 2.87,
        }
        deltas = calculate_temporal_deltas(t1_props, t2_props)
        self.assertEqual(deltas["delta_area_m2"], 15.0)
        self.assertEqual(deltas["delta_true_3d_area_m2"], 17.32)
        self.assertEqual(deltas["delta_capacity_kwp"], 3.08)
        self.assertEqual(deltas["delta_module_count"], 8)
        self.assertEqual(deltas["delta_annual_yield_mwh"], 3.23)
        self.assertEqual(deltas["delta_co2_savings_tons"], 1.23)

    def test_detect_new_installation(self):
        geojson_t1 = {"type": "FeatureCollection", "features": []}
        geojson_t2 = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [50, 10], [50, 50], [10, 50], [10, 10]]]},
                    "properties": {
                        "feature_id": "pv-array-001",
                        "pixel_polygon": [(10, 10), (50, 10), (50, 50), (10, 50)],
                        "area_m2": 16.0,
                        "fitted_capacity_kwp": 3.28,
                        "fitted_module_count": 8,
                    },
                }
            ],
        }

        changes = detect_temporal_changes(geojson_t1, geojson_t2, date_t1="2024-06-01", date_t2="2025-06-01")
        summary = changes["summary"]
        self.assertEqual(summary["total_events"], 1)
        self.assertEqual(summary["new_installations"], 1)
        self.assertEqual(summary["expansions"], 0)
        self.assertEqual(summary["decommissioned_arrays"], 0)
        self.assertEqual(summary["stable_arrays"], 0)
        self.assertEqual(summary["net_delta_capacity_kwp"], 3.28)
        self.assertEqual(summary["net_delta_module_count"], 8)
        self.assertEqual(summary["interval_days"], 365)

        feat = changes["features"][0]
        self.assertEqual(feat["properties"]["change_type"], CHANGE_TYPE_NEW)
        self.assertEqual(feat["properties"]["delta_capacity_kwp"], 3.28)

    def test_detect_stable_array(self):
        poly = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
        f1 = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[p for p in poly]]},
            "properties": {
                "feature_id": "pv-array-t1-001",
                "pixel_polygon": poly,
                "area_m2": 16.0,
                "fitted_capacity_kwp": 3.28,
                "fitted_module_count": 8,
            },
        }
        # Slightly jittered poly in T2 (e.g. 0.5px difference)
        poly2 = [(10.2, 10.1), (50.1, 10.0), (50.0, 49.9), (10.1, 50.0)]
        f2 = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[p for p in poly2]]},
            "properties": {
                "feature_id": "pv-array-t2-001",
                "pixel_polygon": poly2,
                "area_m2": 16.0,
                "fitted_capacity_kwp": 3.28,
                "fitted_module_count": 8,
            },
        }

        geojson_t1 = {"type": "FeatureCollection", "features": [f1]}
        geojson_t2 = {"type": "FeatureCollection", "features": [f2]}

        changes = detect_temporal_changes(geojson_t1, geojson_t2, date_t1="2024-06-01", date_t2="2025-06-01")
        summary = changes["summary"]
        self.assertEqual(summary["stable_arrays"], 1)
        self.assertEqual(summary["new_installations"], 0)
        self.assertEqual(summary["expansions"], 0)
        self.assertEqual(summary["decommissioned_arrays"], 0)
        self.assertEqual(changes["features"][0]["properties"]["change_type"], CHANGE_TYPE_STABLE)
        self.assertGreater(changes["features"][0]["properties"]["spatial_overlap_iou"], 0.90)

    def test_detect_expansion(self):
        # T1 has 400 px² box
        poly_t1 = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
        f1 = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {
                "feature_id": "pv-t1-01",
                "pixel_polygon": poly_t1,
                "area_m2": 4.0,
                "fitted_capacity_kwp": 0.82,
                "fitted_module_count": 2,
            },
        }
        # T2 has doubled width box (800 px²), overlapping 50%
        poly_t2 = [(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0)]
        f2 = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {
                "feature_id": "pv-t2-01",
                "pixel_polygon": poly_t2,
                "area_m2": 8.0,
                "fitted_capacity_kwp": 1.64,
                "fitted_module_count": 4,
            },
        }

        changes = detect_temporal_changes(
            {"type": "FeatureCollection", "features": [f1]},
            {"type": "FeatureCollection", "features": [f2]},
        )
        feat = changes["features"][0]
        self.assertEqual(feat["properties"]["change_type"], CHANGE_TYPE_EXPANSION)
        self.assertEqual(feat["properties"]["delta_capacity_kwp"], 0.82)
        self.assertEqual(feat["properties"]["delta_module_count"], 2)

    def test_detect_decommissioned(self):
        poly_t1 = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
        f1 = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {
                "feature_id": "pv-t1-retired",
                "pixel_polygon": poly_t1,
                "area_m2": 4.0,
                "fitted_capacity_kwp": 0.82,
                "fitted_module_count": 2,
            },
        }

        changes = detect_temporal_changes(
            {"type": "FeatureCollection", "features": [f1]},
            {"type": "FeatureCollection", "features": []},
        )
        summary = changes["summary"]
        self.assertEqual(summary["decommissioned_arrays"], 1)
        self.assertEqual(summary["net_delta_capacity_kwp"], -0.82)
        self.assertEqual(summary["net_delta_module_count"], -2)
        feat = changes["features"][0]
        self.assertEqual(feat["properties"]["change_type"], CHANGE_TYPE_DECOMMISSIONED)
        self.assertEqual(feat["properties"]["delta_capacity_kwp"], -0.82)

    def test_render_temporal_cad_overlay(self):
        img_t2 = Image.new("RGB", (200, 200), (80, 80, 80))
        img_t1 = Image.new("RGB", (200, 200), (60, 60, 60))

        temporal_geojson = {
            "type": "FeatureCollection",
            "summary": {"survey_epoch_t1": "2024-06-15", "survey_epoch_t2": "2025-06-20"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "properties": {
                        "change_type": CHANGE_TYPE_NEW,
                        "pixel_polygon": [(20, 20), (80, 20), (80, 80), (20, 80)],
                        "centroid_pixel": (50, 50),
                        "fitted_capacity_kwp": 5.0,
                        "fitted_module_count": 12,
                        "delta_capacity_kwp": 5.0,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "properties": {
                        "change_type": CHANGE_TYPE_EXPANSION,
                        "pixel_polygon": [(110, 20), (180, 20), (180, 80), (110, 80)],
                        "centroid_pixel": (145, 50),
                        "fitted_capacity_kwp": 8.0,
                        "delta_capacity_kwp": 3.0,
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "temporal_cad.png"
            render_temporal_cad_overlay(img_t2, temporal_geojson, out_p, image_t1=img_t1)
            self.assertTrue(out_p.exists())
            with Image.open(out_p) as rendered:
                # Side by side should have double width plus spacer
                self.assertEqual(rendered.width, 200 * 2 + 10)
                self.assertEqual(rendered.height, 200)


if __name__ == "__main__":
    unittest.main()
