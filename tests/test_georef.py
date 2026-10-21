# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for georeferencing extraction, worldfile parsing, and CAD table layout estimation."""

import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from pvextent.cad.georef import extract_georeference, parse_worldfile
from pvextent.cad.vectorize import vectorize_mask
import numpy as np


class TestGeoref(unittest.TestCase):
    def test_parse_worldfile(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wf_path = Path(tmp_dir) / "test.tfw"
            # 6-line standard worldfile: A, D, B, E, C, F
            wf_content = "0.10\n0.0\n0.0\n-0.10\n479235.8\n5671379.0\n"
            wf_path.write_text(wf_content, encoding="utf-8")

            transform = parse_worldfile(wf_path)
            self.assertIsNotNone(transform)
            self.assertEqual(len(transform), 6)
            self.assertAlmostEqual(transform[0], 0.10)
            self.assertAlmostEqual(transform[1], 0.0)
            self.assertAlmostEqual(transform[2], 479235.8)
            self.assertAlmostEqual(transform[3], 0.0)
            self.assertAlmostEqual(transform[4], -0.10)
            self.assertAlmostEqual(transform[5], 5671379.0)

    def test_extract_georeference_companion_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "roof.png"
            pgw_path = Path(tmp_dir) / "roof.pgw"
            Image.new("RGB", (100, 100), color="white").save(img_path)
            pgw_path.write_text("0.10\n0.0\n0.0\n-0.10\n500000.0\n5700000.0\n", encoding="utf-8")

            georef = extract_georeference(img_path)
            self.assertIsNotNone(georef)
            self.assertEqual(georef["crs"], "EPSG:25832")
            self.assertEqual(georef["transform"][2], 500000.0)

    def test_vectorize_with_georef_and_min_area(self):
        # 100x100 mask with one 30x10 array (300 px = 3.0 m² at GSD 0.1) and one tiny 5x5 noise patch (25 px = 0.25 m²)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:30] = 1  # 30 x 10 = 300 px
        mask[70:75, 70:75] = 1  # 5 x 5 = 25 px

        georef = {
            "crs": "EPSG:25832",
            "transform": [0.10, 0.0, 480000.0, 0.0, -0.10, 5600000.0],
        }

        # Filter out noise < 1.0 m²
        res = vectorize_mask(mask, georeference=georef, gsd_m=0.10, min_area_m2=1.0)
        features = res["features"]
        self.assertEqual(len(features), 1)  # Tiny patch filtered out!

        feat = features[0]
        props = feat["properties"]
        self.assertIn("estimated_table_layout", props)
        # Check that geo coordinates are UTM (in the hundreds of thousands, not pixel coords)
        first_pt = feat["geometry"]["coordinates"][0][0]
        self.assertGreater(first_pt[0], 400000.0)
        self.assertGreater(first_pt[1], 5000000.0)


if __name__ == "__main__":
    unittest.main()
