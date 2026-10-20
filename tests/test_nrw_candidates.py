# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Mocked official responses: no network or labelled roof claims in tests."""

from __future__ import annotations

from datetime import date
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from pvextent.data import validate_dataset
from pvextent.data import nrw, pv4ger
from pvextent.data.validation import ValidationReport

REAL_FETCH = nrw._fetch

try:
    import pyproj
except ImportError:
    pyproj = None


def building(identifier: int, x: float = 392080, y: float = 5711600) -> dict:
    return {"type": "Feature", "id": f"DENW{identifier:012d}BL", "geometry": {
        "type": "MultiPolygon", "coordinates": [[[[x - 6, y - 5], [x + 6, y - 5],
                                                    [x + 6, y + 5], [x - 6, y + 5], [x - 6, y - 5]]]],
    }, "properties": {"gebnutzbez": "Gebäude", "funktion": "Wohngebäude", "aktualit": "2024-01-01T00:00:00Z"}}


def metadata(*, gsd: str = "0.10", capture: str = "07.03.2025") -> bytes:
    return f'''<?xml version="1.0"?>
<msGMLOutput xmlns:gml="http://www.opengis.net/gml">
<nw_dop_utm_info_layer><nw_dop_utm_info_feature>
<gml:boundedBy><gml:Box srsName="EPSG:25832"><gml:coordinates>
392000,5711000 393000,5712000</gml:coordinates></gml:Box></gml:boundedBy>
<Kachelname>32392_5711</Kachelname><Bildflugdatum>{capture}</Bildflugdatum>
<Bodenauflösung>{gsd}</Bodenauflösung></nw_dop_utm_info_feature>
</nw_dop_utm_info_layer></msGMLOutput>'''.encode()


@unittest.skipIf(pyproj is None, "Install the geo extra for NRW candidate tests")
class NRWCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "source", self.root / "candidates"
        (self.source / "provenance").mkdir(parents=True)
        partition = self.source / "provenance" / "default_partition.json"
        partition.write_text(json.dumps({"train": ["6.0,51.0"], "valid": ["6.1,51.1"], "test": ["6.2,51.2"]}))
        payload = partition.read_bytes()
        files = (pv4ger.SourceFile("default_partition.json", len(payload), hashlib.sha256(payload).hexdigest()),)
        self.features = [building(1), building(2, y=5711700), building(3, y=5711800)]
        self.gml = metadata()
        self.calls = []

        def validation(root):
            if Path(root).resolve() == self.source.resolve():
                report = ValidationReport()
                report.stats.update(records=4027, dataset_fingerprint="a" * 64)
                return report
            return validate_dataset(root)

        for patcher in (
            patch.object(nrw, "SOURCE_FILES", files),
            patch.object(pv4ger, "EXPECTED_SPLIT_COUNTS", {"train": 1, "valid": 1, "test": 1}),
            patch.object(nrw, "validate_dataset", side_effect=validation),
            patch.object(nrw, "_fetch", side_effect=self.fetch),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fetch(self, url: str, maximum: int):
        self.calls.append((url, maximum))
        query = parse_qs(urlsplit(url).query)
        if url.startswith(nrw.BUILDING_URL):
            result = {"type": "FeatureCollection", "numberReturned": len(self.features),
                      "numberMatched": len(self.features), "features": self.features}
            return json.dumps(result).encode(), {"content-crs": "<" + nrw.CRS_URI + ">"}
        if query["REQUEST"] == ["GetFeatureInfo"]:
            return self.gml, {"content-type": "application/vnd.ogc.gml"}
        self.assertEqual(query["CRS"], ["EPSG:25832"])
        self.assertEqual(query["LAYERS"], ["nw_dop_rgb"])
        self.assertEqual(query["WIDTH"], ["320"])
        self.assertEqual(query["HEIGHT"], ["320"])
        bounds = list(map(float, query["BBOX"][0].split(",")))
        self.assertAlmostEqual(bounds[2] - bounds[0], 32)
        self.assertAlmostEqual(bounds[3] - bounds[1], 32)
        image = Image.new("RGB", (320, 320), (int(bounds[1]) % 255, 80, 120))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), {"content-type": "image/png"}

    def test_candidates_are_ignored_unreviewed_georeferenced_and_provenanced(self) -> None:
        report = nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=3)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["pv_pixels"], 0)
        self.assertEqual(report["background_pixels"], 0)
        self.assertEqual(report["ignored_pixels"], 3 * 320 * 320)
        self.assertEqual(report["network_requests"], {"building_requests": 1, "metadata_requests": 1, "image_requests": 3})
        self.assertGreater(report["minimum_source_distance_observed_m"], nrw.MIN_SOURCE_DISTANCE_M)
        result = validate_dataset(self.output)
        self.assertTrue(result.ok, result.to_dict())
        self.assertEqual(result.stats["ignored_pixels"], 3 * 320 * 320)
        records = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        for record in records:
            self.assertEqual(record["capture_date"], "2025-03-07")
            self.assertEqual(record["annotation_status"], "unreviewed")
            self.assertEqual(record["split"], "unassigned")
            self.assertTrue(record["building_ids"][0].startswith("geobasis-nrw:alkis:DENW"))
            self.assertEqual(record["georeference"]["crs"], "EPSG:25832")
            transform = record["georeference"]["transform"]
            self.assertEqual(transform[:2], [0.1, 0])
            self.assertEqual(transform[3:5], [0, -0.1])
            with Image.open(self.output / record["mask"]) as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.getextrema(), (255, 255))
        samples = [json.loads(line) for line in (self.output / "provenance" / "samples.jsonl").read_text().splitlines()]
        self.assertTrue(all(item["native_gsd_m"] == 0.1 for item in samples))
        self.assertTrue(all(item["date_basis"] == "single_metadata_tile_contains_patch" for item in samples))
        self.assertTrue((self.output / "provenance" / "buildings.geojson").exists())

    def test_metadata_date_never_uses_retrieval_or_building_currency(self) -> None:
        self.gml = metadata(capture="unknown")
        nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=1)
        record = json.loads((self.output / "manifest.jsonl").read_text())
        self.assertIsNone(record["capture_date"])

    def test_twenty_cm_source_is_not_upsampled_into_a_ten_cm_claim(self) -> None:
        self.gml = metadata(gsd="0.20")
        with self.assertRaisesRegex(nrw.DatasetImportError, "verified native 10 cm"):
            nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=1)
        self.assertFalse(self.output.exists())
        self.assertFalse(any(parse_qs(urlsplit(url).query).get("REQUEST") == ["GetMap"] for url, _ in self.calls))
        self.assertEqual(sorted(item.name for item in self.root.iterdir()), ["source"])

    def test_source_proximity_is_checked_against_all_source_centers(self) -> None:
        project = pyproj.Transformer.from_crs(25832, 4326, always_xy=True)
        close = project.transform(392080, 5711600)
        with patch.object(nrw, "_partition", return_value=({}, {"source-center": close})):
            with self.assertRaisesRegex(nrw.DatasetImportError, "No roof candidates"):
                nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=3)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.output.exists())

    def test_deterministic_sort_spacing_and_accessory_filter(self) -> None:
        garage = building(0)
        garage["properties"]["funktion"] = "Garage"
        self.features = [building(5, y=5711800), building(4, y=5711620), garage, building(1)]
        report = nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=3)
        records = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual([r["id"] for r in records], ["nrw-DENW000000000001BL", "nrw-DENW000000000005BL"])
        self.assertTrue(report["selection_exhausted"])
        self.assertEqual(report["selection_exclusions"]["near_selected_candidate"], 1)

    def test_existing_output_is_preserved_without_network(self) -> None:
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaises(nrw.DatasetImportError):
            nrw.fetch_nrw_roof_candidates(self.source, self.output)
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        self.assertEqual(self.calls, [])

    def test_wrong_building_crs_fails_before_imagery_fetch(self) -> None:
        original = self.fetch

        def wrong_crs(url, maximum):
            payload, headers = original(url, maximum)
            if url.startswith(nrw.BUILDING_URL):
                headers["content-crs"] = "<http://www.opengis.net/def/crs/OGC/1.3/CRS84>"
            return payload, headers

        with patch.object(nrw, "_fetch", side_effect=wrong_crs):
            with self.assertRaisesRegex(nrw.DatasetImportError, "EPSG:25832"):
                nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=1)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.output.exists())

    def test_failed_image_fetch_cleans_staging_without_publishing(self) -> None:
        original = self.fetch

        def fail(url, maximum):
            if parse_qs(urlsplit(url).query).get("REQUEST") == ["GetMap"]:
                raise OSError("fixture service failure")
            return original(url, maximum)

        with patch.object(nrw, "_fetch", side_effect=fail):
            with self.assertRaisesRegex(nrw.DatasetImportError, "fixture service failure"):
                nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=1)
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(item.name for item in self.root.iterdir()), ["source"])

    def test_invalid_limits_do_not_fetch(self) -> None:
        for value in (0, -1, 25, True, 1.5, "3"):
            with self.subTest(limit=value), self.assertRaises(nrw.DatasetImportError):
                nrw.fetch_nrw_roof_candidates(self.source, self.output, limit=value)
        self.assertEqual(self.calls, [])

    def test_metadata_tile_bounds_prevent_borrowing_a_neighbor_date(self) -> None:
        result = nrw._metadata(metadata(), (391990, 5711500, 392022, 5711532), date(2026, 10, 20))
        self.assertIsNone(result["capture_date"])
        self.assertIsNone(result["native_gsd_m"])
        result = nrw._metadata(metadata(capture="07.03.2099"), (392010, 5711500, 392042, 5711532), date(2026, 10, 20))
        self.assertIsNone(result["capture_date"])

    def test_concave_or_holed_footprints_never_get_an_outside_center(self) -> None:
        feature = building(1)
        outer = feature["geometry"]["coordinates"][0][0]
        feature["geometry"]["coordinates"][0].append([
            [392078, 5711598], [392082, 5711598], [392082, 5711602],
            [392078, 5711602], [392078, 5711598],
        ])
        center, area = nrw._roof(feature)
        self.assertTrue(nrw._inside(center, [outer, feature["geometry"]["coordinates"][0][1]]))
        self.assertAlmostEqual(area, 104)


class NRWFetchBoundTests(unittest.TestCase):
    def test_network_failure_identifies_the_public_request_without_retrying(self) -> None:
        with patch.object(nrw, "build_opener") as opener:
            opener.return_value.open.side_effect = ConnectionResetError(54, "Connection reset by peer")
            with self.assertRaisesRegex(nrw.DatasetImportError, "Official source request failed: https://ogc-api.nrw.de"):
                REAL_FETCH(nrw.BUILDING_URL, 32)
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_oversized_response_is_rejected_with_a_bounded_read(self) -> None:
        class Response(BytesIO):
            headers = {"Content-Type": "application/geo+json"}

        with patch.object(nrw, "build_opener") as opener:
            reply = Response(b"x" * 1000)
            opener.return_value.open.return_value = reply
            with self.assertRaisesRegex(nrw.DatasetImportError, "exceeds 32 bytes"):
                REAL_FETCH(nrw.BUILDING_URL, 32)
            self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
