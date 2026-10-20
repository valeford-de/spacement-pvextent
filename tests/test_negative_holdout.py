# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Bounded holdout acquisition without reopening any reference rasters."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from PIL import Image

try:
    from pyproj import Transformer
    from pvextent.data import negative_holdout, nrw
    from pvextent.data.training import _read_manifest
except ImportError:
    Transformer = None


METADATA = '''schema_version = "1.0"
dataset_id = "holdout-exclusion-fixture"
dataset_version = "1"
[sources.synthetic]
name = "Synthetic metadata-only exclusion"
url = "https://example.invalid/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "CC0-1.0"
attribution = "Synthetic test fixture"
'''


def building(identifier: int, x: float = 392080, y: float = 5711600) -> dict:
    return {"type": "Feature", "id": f"DENW{identifier:012d}BL", "geometry": {
        "type": "MultiPolygon", "coordinates": [[[[x-6, y-5], [x+6, y-5],
            [x+6, y+5], [x-6, y+5], [x-6, y-5]]]],
    }, "properties": {"gebnutzbez": "Gebäude", "funktion": "Wohngebäude",
                      "aktualit": "2024-01-01T00:00:00Z"}}


@unittest.skipIf(Transformer is None, "Install train and geo extras for holdout tests")
class NegativeHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source, self.output = self.root / "reference", self.root / "holdout"
        (self.source / "provenance").mkdir(parents=True)
        (self.source / "dataset.toml").write_text(METADATA, encoding="utf-8")
        self.inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        self.rows, self.points = [], {}
        for i, split in enumerate(("train", "val", "test", "unassigned")):
            identifier = "reference-" + split
            point = (300000 + i*1000, 5600000)
            self.points[identifier] = point
            self.rows.append({
                "id": identifier, "image": f"images/{identifier}.png", "mask": f"masks/{identifier}.png",
                "image_sha256": hashlib.sha256((identifier+"-image").encode()).hexdigest(),
                "mask_sha256": hashlib.sha256((identifier+"-mask").encode()).hexdigest(),
                "width": 320, "height": 320, "gsd_m": .1, "source_id": "synthetic",
                "capture_date": "2025-01-01", "georeference": None,
                "spatial_group": "reference-group-"+split, "building_ids": ["synthetic:"+identifier],
                "split": split, "annotation_status": "unreviewed" if split=="unassigned" else "upstream",
            })
        self.write_reference()
        self.protocol = self.root / "protocol.json"
        self.rules = {"schema_version": "1.0", "dataset_version": "holdout-test-v1",
                      "regions": [{"id": "dortmund", "bbox_crs84": [7.435, 51.545, 7.445, 51.555], "limit": 3}],
                      "min_reference_distance_m": 5000+32*math.sqrt(2), "min_candidate_distance_m": 200}
        self.write_protocol()
        self.features = [building(1), building(2, y=5711850), building(3, y=5712100)]
        self.gsd, self.capture = "0.10", "07.03.2025"
        self.calls = []
        self.image_payload_override = None
        patcher = patch.object(nrw, "_fetch", side_effect=self.fetch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_reference(self) -> None:
        (self.source / "manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in self.rows), encoding="utf-8")
        (self.source / "provenance" / "samples.jsonl").write_text("".join(json.dumps({
            "id": row["id"], "approximate_center": list(self.inverse.transform(*self.points[row["id"]])),
            "center_crs": "EPSG:4326"})+"\n" for row in self.rows), encoding="utf-8")

    def write_protocol(self) -> None:
        self.protocol.write_text(json.dumps(self.rules), encoding="utf-8")

    @staticmethod
    def image_payload(bounds):
        image = Image.new("RGB", (320, 320), (int(bounds[1]) % 255, int(bounds[0]) % 255, 110))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def fetch(self, url, maximum):
        self.calls.append((url, maximum))
        query = parse_qs(urlsplit(url).query)
        if url.startswith(nrw.BUILDING_URL):
            payload = {"type": "FeatureCollection", "numberReturned": len(self.features),
                       "numberMatched": len(self.features), "features": self.features}
            return json.dumps(payload).encode(), {"content-crs": "<"+nrw.CRS_URI+">"}
        bounds = list(map(float, query["BBOX"][0].split(",")))
        self.assertEqual(query["CRS"], ["EPSG:25832"])
        self.assertEqual(query["WIDTH"], ["320"])
        self.assertEqual(query["HEIGHT"], ["320"])
        if query["REQUEST"] == ["GetFeatureInfo"]:
            payload = f'''<?xml version="1.0"?><msGMLOutput xmlns:gml="http://www.opengis.net/gml">
<nw_dop_utm_info_layer><nw_dop_utm_info_feature><gml:boundedBy><gml:Box srsName="EPSG:25832">
<gml:coordinates>390000,5700000 400000,5720000</gml:coordinates></gml:Box></gml:boundedBy>
<Kachelname>synthetic-tile</Kachelname><Bildflugdatum>{self.capture}</Bildflugdatum>
<Bodenauflösung>{self.gsd}</Bodenauflösung></nw_dop_utm_info_feature></nw_dop_utm_info_layer></msGMLOutput>'''
            return payload.encode(), {"content-type": "application/vnd.ogc.gml"}
        self.assertEqual(query["REQUEST"], ["GetMap"])
        self.assertEqual(query["LAYERS"], ["nw_dop_rgb"])
        self.assertAlmostEqual(bounds[2]-bounds[0], 32)
        self.assertAlmostEqual(bounds[3]-bounds[1], 32)
        return self.image_payload_override or self.image_payload(bounds), {"content-type": "image/png"}

    def acquire(self):
        return negative_holdout.fetch_negative_holdout_candidates([self.source], self.output, protocol_path=self.protocol)

    @staticmethod
    def maps(calls):
        return [url for url, _ in calls if parse_qs(urlsplit(url).query).get("REQUEST")==["GetMap"]]

    @contextmanager
    def forbid_reference_pixels(self):
        original_open, original_image_open = Path.open, Image.open
        original_validate = negative_holdout.validate_dataset

        def forbidden(path):
            p = Path(path).absolute()
            return p.is_relative_to(self.source / "images") or p.is_relative_to(self.source / "masks")

        def guarded_open(path, *args, **kwargs):
            if forbidden(path):
                raise AssertionError("Reference raster accessed: "+str(path))
            return original_open(path, *args, **kwargs)

        def guarded_image_open(fp, *args, **kwargs):
            if isinstance(fp, (str, os.PathLike)) and forbidden(fp):
                raise AssertionError("Reference raster decoded: "+str(fp))
            return original_image_open(fp, *args, **kwargs)

        def new_only(directory, *args, **kwargs):
            self.assertNotEqual(Path(directory).resolve(), self.source)
            return original_validate(directory, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Image, "open", guarded_image_open), \
                patch.object(negative_holdout, "validate_dataset", new_only):
            yield

    def test_candidates_preserve_native_bytes_and_never_read_reference_pixels(self):
        before = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        with self.forbid_reference_pixels():
            result = self.acquire()
        self.assertTrue(result["ok"])
        self.assertEqual(result["records"], 3)
        metadata, rows, _ = _read_manifest(self.output)
        self.assertEqual(metadata["dataset_id"], "nrw-negative-holdout-candidates")
        self.assertNotIn("purpose", result)
        self.assertNotIn("intended_use", result)
        self.assertNotIn("training_use_allowed", result)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual((row["split"], row["annotation_status"]), ("unassigned", "unreviewed"))
            self.assertEqual((row["width"], row["height"], row["gsd_m"]), (320, 320, .1))
            self.assertEqual(row["capture_date"], "2025-03-07")
            self.assertEqual(row["spatial_group"], "nrw-negative-holdout-dortmund")
            with Image.open(self.output / row["mask"]) as mask:
                self.assertEqual((mask.mode, mask.getextrema()), ("L", (255, 255)))
            bbox = row["georeference"]["transform"]
            bounds = (bbox[2], bbox[5]-32, bbox[2]+32, bbox[5])
            self.assertEqual((self.output / row["image"]).read_bytes(), self.image_payload(bounds))
        self.assertTrue(metadata["dataset_id"])
        self.assertEqual({p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}, before)
        self.assertLessEqual(len(self.calls), 1+3+3)

    def test_every_reference_partition_excludes_nearby_candidate_before_imagery(self):
        self.features = [building(1)]
        for split in ("train", "val", "test", "unassigned"):
            with self.subTest(split=split):
                saved = dict(self.points)
                self.points["reference-"+split] = (392080, 5711600)
                self.write_reference()
                self.calls.clear()
                with self.forbid_reference_pixels(), self.assertRaises(negative_holdout.NegativeHoldoutError):
                    self.acquire()
                self.assertFalse(self.maps(self.calls))
                self.assertFalse(self.output.exists())
                self.points = saved

    def test_building_identity_overlap_is_rejected_before_getmap(self):
        self.features = [building(1)]
        self.rows[0]["building_ids"] = ["geobasis-nrw:alkis:"+self.features[0]["id"]]
        self.write_reference()
        with self.assertRaises(negative_holdout.NegativeHoldoutError):
            self.acquire()
        self.assertFalse(self.maps(self.calls))
        self.assertFalse(self.output.exists())

    def test_image_content_overlap_is_never_published(self):
        self.features = [building(1)]
        payload = self.image_payload(nrw._bounds((392080, 5711600)))
        self.rows[1]["image_sha256"] = hashlib.sha256(payload).hexdigest()
        self.write_reference()
        with self.assertRaises(negative_holdout.NegativeHoldoutError):
            self.acquire()
        self.assertFalse(self.output.exists())

    def test_invalid_native_gsd_or_date_is_excluded_before_getmap(self):
        self.features = [building(1)]
        for gsd, capture in (("0.20", "07.03.2025"), ("0.10", "unknown"), ("0.10", "07.03.2099")):
            with self.subTest(gsd=gsd, capture=capture):
                self.gsd, self.capture = gsd, capture
                self.calls.clear()
                with self.assertRaises(negative_holdout.NegativeHoldoutError):
                    self.acquire()
                self.assertFalse(self.maps(self.calls))
                self.assertFalse(self.output.exists())

    def test_fixed_id_order_and_spacing(self):
        self.features = [building(5, y=5712100), building(4, y=5711620), building(1)]
        result = self.acquire()
        _, rows, _ = _read_manifest(self.output)
        self.assertEqual([r["id"] for r in rows], ["nrw-DENW000000000001BL", "nrw-DENW000000000005BL"])
        self.assertEqual(result["records"], 2)

    def test_failed_metadata_does_not_expand_preselected_region_quota(self):
        self.rules["regions"][0]["limit"] = 1
        self.write_protocol()
        # A later feature could pass source metadata, but lies beyond the
        # fixed one-candidate quota and must never replace its failed member.
        self.features = [building(1), building(2, y=5712100)]
        metadata_attempts = 0
        def first_invalid(url, maximum):
            nonlocal metadata_attempts
            query = parse_qs(urlsplit(url).query)
            if query.get("REQUEST") == ["GetFeatureInfo"]:
                metadata_attempts += 1
                self.capture = "unknown" if metadata_attempts == 1 else "07.03.2025"
                result = self.fetch(url, maximum)
                self.capture = "07.03.2025"
                return result
            return self.fetch(url, maximum)
        with patch.object(nrw, "_fetch", side_effect=first_invalid), self.assertRaises(negative_holdout.NegativeHoldoutError):
            self.acquire()
        metadata_calls = [url for url, _ in self.calls
                          if parse_qs(urlsplit(url).query).get("REQUEST") == ["GetFeatureInfo"]]
        self.assertEqual(len(metadata_calls), 1)
        self.assertFalse(self.maps(self.calls))
        self.assertFalse(self.output.exists())

    def test_existing_output_and_overlapping_output_refused_without_network(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("keep")
        with self.assertRaises(negative_holdout.NegativeHoldoutError):
            self.acquire()
        self.assertEqual(marker.read_text(), "keep")
        for output in (self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaises(negative_holdout.NegativeHoldoutError):
                negative_holdout.fetch_negative_holdout_candidates([self.source], output, protocol_path=self.protocol)
        self.assertEqual(self.calls, [])

    def test_duplicate_protocol_keys_and_nonfinite_numbers_are_rejected(self):
        valid = json.dumps(self.rules)
        malformed = [valid.replace('"schema_version": "1.0"', '"schema_version": "1.0", "schema_version": "1.0"'),
                     valid.replace('"min_candidate_distance_m": 200', '"min_candidate_distance_m": NaN'),
                     valid.replace('"min_candidate_distance_m": 200', '"min_candidate_distance_m": 1e999')]
        for payload in malformed:
            with self.subTest(payload=payload):
                self.protocol.write_text(payload)
                with self.assertRaises(negative_holdout.NegativeHoldoutError):
                    self.acquire()
        self.assertEqual(self.calls, [])

    def test_failed_image_request_has_no_retry_and_cleans_staging(self):
        self.features = [building(1)]
        def fail(url, maximum):
            if parse_qs(urlsplit(url).query).get("REQUEST") == ["GetMap"]:
                self.calls.append((url, maximum))
                raise OSError("synthetic image failure")
            return self.fetch(url, maximum)
        with patch.object(nrw, "_fetch", side_effect=fail), self.assertRaises(negative_holdout.NegativeHoldoutError):
            self.acquire()
        self.assertEqual(len(self.maps(self.calls)), 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".holdout-*")))

    def test_wrong_response_crs_fails_before_metadata_or_image(self):
        def wrong(url, maximum):
            payload, headers = self.fetch(url, maximum)
            if url.startswith(nrw.BUILDING_URL):
                headers["content-crs"] = "<http://www.opengis.net/def/crs/OGC/1.3/CRS84>"
            return payload, headers
        with patch.object(nrw, "_fetch", side_effect=wrong), self.assertRaises(negative_holdout.NegativeHoldoutError):
            self.acquire()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.output.exists())

    def test_invalid_protocol_rejected_before_network(self):
        cases = []
        for key, value in (("schema_version", "2.0"), ("min_reference_distance_m", 5000),
                           ("min_candidate_distance_m", 199), ("min_reference_distance_m", True)):
            rule = copy.deepcopy(self.rules);rule[key] = value;cases.append(rule)
        for change in ({"limit": 9}, {"limit": True}, {"bbox_crs84": [7, 51, 8, 52]}, {"id": "../bad"}):
            rule = copy.deepcopy(self.rules);rule["regions"][0].update(change);cases.append(rule)
        rule = copy.deepcopy(self.rules);rule["regions"] *= 2;cases.append(rule)
        rule = copy.deepcopy(self.rules);rule["unexpected"] = True;cases.append(rule)
        rule = copy.deepcopy(self.rules);rule["distance_reference_splits"] = ["val", "test", "unassigned"];cases.append(rule)
        for rule in cases:
            with self.subTest(rule=rule):
                self.protocol.write_text(json.dumps(rule))
                with self.assertRaises(negative_holdout.NegativeHoldoutError):
                    self.acquire()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
