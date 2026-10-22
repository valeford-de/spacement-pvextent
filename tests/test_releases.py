# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PVExtent Vidda and Tinden official model releases."""

import unittest

from pvextent.models.releases import (
    MODEL_RELEASES,
    TINDEN_RELEASE_SPEC,
    VIDDA_RELEASE_SPEC,
    get_model_spec,
    list_model_releases,
)


class TestModelReleases(unittest.TestCase):
    def test_list_releases(self):
        releases = list_model_releases()
        self.assertEqual(len(releases), 2)
        names = [r.name for r in releases]
        self.assertIn("PVExtent Vidda", names)
        self.assertIn("PVExtent Tinden", names)

    def test_get_vidda_spec(self):
        vidda = get_model_spec("vidda")
        self.assertEqual(vidda.name, "PVExtent Vidda")
        self.assertEqual(vidda.version, "1.0.0")
        self.assertEqual(vidda.tier, "Standard")
        self.assertGreater(vidda.benchmark_iou, 0.88)
        self.assertFalse(vidda.supports_multispectral_nir)

    def test_get_tinden_spec(self):
        tinden = get_model_spec("tinden")
        self.assertEqual(tinden.name, "PVExtent Tinden")
        self.assertEqual(tinden.version, "1.0.0")
        self.assertIn("High-End", tinden.tier)
        self.assertGreater(tinden.benchmark_iou, 0.89)
        self.assertEqual(tinden.default_tta, "d4")
        self.assertTrue(tinden.supports_multispectral_nir)
        self.assertTrue(tinden.supports_3d_elevation)
        self.assertTrue(tinden.supports_4d_temporal)

    def test_unknown_release_raises(self):
        with self.assertRaises(KeyError):
            get_model_spec("nonexistent_model")


if __name__ == "__main__":
    unittest.main()
