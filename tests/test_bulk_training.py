# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic network-only fixtures: held-out raster files deliberately do not exist."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageDraw

try:
    from pyproj import Transformer
    from pvextent.data import bulk_training as bulk, nrw
    from pvextent.data.training import _read_manifest
except ImportError:
    Transformer = None


METADATA = '''schema_version = "1.0"
dataset_id = "bulk-reference"
dataset_version = "1"
[sources.pv4ger-seg]
name = "Synthetic PV4GER reference"
url = "https://example.invalid/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "MIT"
attribution = "Synthetic test fixture"
'''


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def picture(code=1, *, compress_level=6):
    image = Image.new("RGB", (320, 320), (code % 251, code // 251 % 251, 90))
    ImageDraw.Draw(image).rectangle((80, 80, 120, 120), fill=(200, code % 251, 10))
    output = BytesIO()
    image.save(output, format="PNG", compress_level=compress_level)
    return output.getvalue()


def building(identifier, x=392780, y=5711600):
    return {"type": "Feature", "id": f"DENW{identifier:012d}BL", "geometry": {
        "type": "MultiPolygon", "coordinates": [[[[x-6, y-5], [x+6, y-5],
            [x+6, y+5], [x-6, y+5], [x-6, y-5]]]]},
            "properties": {"gebnutzbez": "Gebäude", "funktion": "Wohngebäude", "aktualit": "2025-01-01"}}


@unittest.skipIf(Transformer is None, "Install train and geo extras for bulk acquisition tests")
class BulkTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.training, self.protected, self.identity = [self.root/name for name in ("training", "pilot", "new-pool")]
        self.output, self.cache, self.protocol = self.root/"output", self.root/"cache", self.root/"protocol.json"
        self.inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        self.rows, self.points = [], {}
        for index, split in enumerate(("train", "train", "val", "test", "unassigned")):
            identifier = "original-"+str(index)
            center = (392080, 5711600+index*300) if split == "train" else (300000+index*1000, 5600000)
            self.rows.append(self.row(identifier, split))
            self.points[identifier] = center
        self.write_snapshot(self.training, self.rows, self.points)
        for index, row in enumerate(self.rows):
            if row["split"] == "train":
                payload = picture(10+index)
                (self.training/"images").mkdir(exist_ok=True)
                (self.training/row["image"]).write_bytes(payload)
                row["image_sha256"] = digest(payload)
        self.write_snapshot(self.training, self.rows, self.points)
        self.write_snapshot(self.protected, [self.row("pilot", "unassigned")], {"pilot": (310000, 5600000)})
        self.write_snapshot(self.identity, [self.row("pool", "unassigned")], {"pool": (390000, 5713000)})
        notice = self.training/"provenance"/"parent"
        notice.mkdir()
        (notice/"original-MIT-LICENSE").write_text("MIT synthetic coordinate notice\n")
        self.rules = {"schema_version": "1.0", "dataset_version": "bulk-test-v1", "seed": 20261021,
            "target_known": 1, "target_geometry": 1, "max_known_candidates": 2,
            "regions": [{"id": "dortmund", "bbox_crs84": [7.43, 51.54, 7.47, 51.58], "limit": 3}],
            "min_reference_distance_m": bulk.MIN_REFERENCE_DISTANCE_M,
            "min_candidate_distance_m": 80, "min_known_distance_m": 200,
            "max_workers": 2, "request_interval_s": .35, "max_attempts": 3, "max_network_requests": 50}
        self.write_protocol()
        self.features = [building(1), building(2, y=5711850), building(3, y=5712100)]
        self.calls, self.progress = [], []
        self.fail_all_maps = False
        self.map_override = None
        self.bad_metadata = False
        self.failures = {}
        fetch_patch = patch.object(nrw, "_fetch", side_effect=self.fetch)
        fetch_patch.start()
        self.addCleanup(fetch_patch.stop)
        sleep_patch = patch.object(bulk.time, "sleep")
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    @staticmethod
    def row(identifier, split):
        return {"id": identifier, "image": "images/"+identifier+".png", "mask": "masks/"+identifier+".png",
            "image_sha256": digest((identifier+"-image").encode()), "mask_sha256": digest((identifier+"-mask").encode()),
            "width": 320, "height": 320, "gsd_m": .1, "source_id": "pv4ger-seg", "capture_date": "2023-01-01",
            "georeference": None, "spatial_group": identifier+"-group", "building_ids": [], "split": split,
            "annotation_status": "unreviewed" if split == "unassigned" else "upstream"}

    @staticmethod
    def write_snapshot(path, rows, points):
        (path/"provenance").mkdir(parents=True, exist_ok=True)
        (path/"dataset.toml").write_text(METADATA)
        (path/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        (path/"provenance"/"samples.jsonl").write_text("".join(json.dumps({"id": r["id"],
            "center_crs": "EPSG:4326", "approximate_center": inverse.transform(*points[r["id"]])})+"\n" for r in rows))

    def write_protocol(self):
        self.protocol.write_text(json.dumps(self.rules))

    def fetch(self, url, maximum):
        self.calls.append((url, maximum))
        if self.failures.get(url, 0):
            self.failures[url] -= 1
            raise OSError("synthetic service timeout")
        query = parse_qs(urlsplit(url).query)
        if url.startswith(nrw.BUILDING_URL):
            return json.dumps({"type": "FeatureCollection", "numberReturned": len(self.features),
                               "features": self.features}).encode(), {"content-crs": "<"+nrw.CRS_URI+">"}
        bounds = list(map(float, query["BBOX"][0].split(",")))
        if query["REQUEST"] == ["GetFeatureInfo"]:
            self.assertEqual((bounds[0]+16) % 1000, 500)
            self.assertEqual((bounds[1]+16) % 1000, 500)
            if self.bad_metadata:
                return b"<ServiceExceptionReport/>", {}
            return ('''<msGMLOutput xmlns:gml="http://www.opengis.net/gml"><nw_dop_utm_info_layer>
<nw_dop_utm_info_feature><gml:boundedBy><gml:Box srsName="EPSG:25832"><gml:coordinates>
390000,5700000 400000,5720000</gml:coordinates></gml:Box></gml:boundedBy>
<Kachelname>synthetic-tile</Kachelname><Bildflugdatum>07.03.2025</Bildflugdatum>
<Bodenauflösung>0.10</Bodenauflösung></nw_dop_utm_info_feature></nw_dop_utm_info_layer></msGMLOutput>''').encode(), {}
        self.assertEqual(query["REQUEST"], ["GetMap"])
        # Selection plan must already be sealed before any new RGB request.
        self.assertTrue(list((self.cache/"runs").glob("*/plan.sha256")))
        if self.fail_all_maps:
            raise OSError("synthetic RGB timeout")
        code = int(bounds[0]*10)+int(bounds[1]*10)
        return self.map_override or picture(code), {"content-type": "image/png"}

    def acquire(self, **kwargs):
        return bulk.fetch_bulk_training(self.training, [self.protected], [self.identity], self.output,
            protocol_path=self.protocol, cache_dir=self.cache, progress=self.progress.append, **kwargs)

    def maps(self):
        return [url for url, _ in self.calls if parse_qs(urlsplit(url).query).get("REQUEST") == ["GetMap"]]

    def test_complete_native_unlabelled_pool_and_raster_guard(self):
        import pvextent.data.training as training_module
        original = training_module._raster
        decoded = []
        def guarded(source, row, role):
            self.assertEqual(source, self.training)
            self.assertEqual(row["split"], "train")
            self.assertEqual(role, "image")
            decoded.append(row["id"])
            return original(source, row, role)
        with patch.object(training_module, "_raster", side_effect=guarded):
            report = self.acquire()
        self.assertTrue(report["complete"])
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["quotas"]["known"]["accepted"], 1)
        self.assertEqual(report["quotas"]["geometry"]["accepted"], 1)
        self.assertEqual(decoded, ["original-0", "original-1"])
        self.assertEqual(report["distance_reference_records"], 4)
        self.assertFalse(report["protected_rasters_opened"])
        self.assertEqual(report["labelled_examples"], 0)
        metadata, rows, _ = _read_manifest(self.output)
        self.assertEqual(metadata["dataset_id"], "nrw-training-bulk")
        for row in rows:
            self.assertRegex(row["id"], r"^nrw-bulk-[a-f0-9]{24}$")
            self.assertEqual(row["split"], "unassigned")
            self.assertEqual(row["annotation_status"], "unreviewed")
            with Image.open(self.output/row["mask"]) as image:
                self.assertEqual(image.getextrema(), (255, 255))
        evidence = json.loads((self.output/"provenance"/"response-evidence.json").read_text())
        self.assertTrue(all(e["fetched_at_utc"] <= report["assembled_at_utc"] for e in evidence))
        self.assertTrue((self.output/"provenance"/"coordinate-source-notices"/"parent"/"original-MIT-LICENSE").exists())
        self.assertEqual(self.progress[-1]["phase"], "published")
        self.assertLessEqual(len(self.maps()), 2)

    def test_resume_uses_cached_responses_and_original_retrieval_times(self):
        self.acquire()
        evidence = json.loads((self.output/"provenance"/"response-evidence.json").read_text())
        first = {e["url"]: e["fetched_at_utc"] for e in evidence}
        import shutil
        shutil.rmtree(self.output)
        count = len(self.calls)
        with patch.object(nrw, "_fetch", side_effect=AssertionError("Resume must use cache")):
            report = self.acquire()
        self.assertEqual(len(self.calls), count)
        self.assertGreater(report["cache_hits_this_assembly"], 0)
        evidence = json.loads((self.output/"provenance"/"response-evidence.json").read_text())
        self.assertEqual(first, {e["url"]: e["fetched_at_utc"] for e in evidence})

    def test_failed_quota_keeps_state_without_publishing_then_explicit_partial(self):
        self.rules["target_geometry"] = 3
        self.write_protocol()
        self.features = self.features[:1]
        with self.assertRaisesRegex(bulk.BulkTrainingError, "quotas incomplete"):
            self.acquire()
        self.assertFalse(self.output.exists())
        run = next((self.cache/"runs").iterdir())
        self.assertTrue((run/"plan.json").exists())
        self.assertTrue((run/"last-incomplete-report.json").exists())
        count = len(self.calls)
        report = self.acquire(allow_partial=True)
        self.assertFalse(report["complete"])
        self.assertEqual(report["records"], 2)
        self.assertEqual(len(self.calls), count)

    def test_network_budget_and_three_attempts_are_durable_on_resume(self):
        self.rules.update(target_geometry=0, max_network_requests=5)
        self.write_protocol()
        self.fail_all_maps = True
        with self.assertRaises(bulk.BulkTrainingError):
            self.acquire()
        self.assertLessEqual(len(self.calls), 5)
        per_url = {}
        for url, _ in self.calls:
            per_url[url] = per_url.get(url, 0)+1
        self.assertLessEqual(max(per_url.values()), 3)
        first = len(self.calls)
        with self.assertRaises(bulk.BulkTrainingError):
            self.acquire()
        self.assertEqual(len(self.calls), first)
        state = json.loads(next((self.cache/"runs").glob("*/state.json")).read_text())
        self.assertEqual(state["network_requests"], first)
        self.assertEqual(sum(state["attempts"].values()), first)

    def test_decoded_rgb_duplicate_rejected_even_differently_encoded_png(self):
        self.map_override = picture(10, compress_level=0)
        self.rules.update(target_geometry=0)
        self.write_protocol()
        with self.assertRaisesRegex(bulk.BulkTrainingError, "quotas incomplete"):
            self.acquire()
        self.assertIn("duplicate_reference_image", self.progress[-1]["rejections"])

    def test_new_decoded_rgb_duplicates_do_not_fill_quotas(self):
        self.map_override = picture(200)
        with self.assertRaisesRegex(bulk.BulkTrainingError, "quotas incomplete"):
            self.acquire()
        report = self.acquire(allow_partial=True)
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["rejections"]["duplicate_new_image"], 3)

    def test_metadata_service_exception_retained_without_rgb_requests(self):
        self.bad_metadata = True
        with self.assertRaises(bulk.BulkTrainingError):
            self.acquire()
        self.assertEqual(self.maps(), [])
        self.assertGreater(self.progress[-1]["rejections"]["native_gsd_date_or_full_crop_coverage_unverified"], 0)
        self.assertTrue(any(path.read_bytes() == b"<ServiceExceptionReport/>" for path in self.cache.glob("*/response.bin")))

    def test_geometry_excludes_protected_identity_old_train_and_existing_pool(self):
        self.rules.update(target_known=0)
        self.write_protocol()
        self.write_snapshot(self.identity, [self.row("pool", "unassigned")], {"pool": (392500, 5712400)})
        self.features = [building(1, *self.points["original-0"]), building(2, 392500, 5712400), building(3)]
        report = self.acquire()
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["rejections"]["geometry_candidate_spacing"], 2)

    def test_no_overwrite_or_changed_reference_binding(self):
        self.acquire()
        with self.assertRaisesRegex(bulk.BulkTrainingError, "already exists"):
            self.acquire()
        import shutil
        shutil.rmtree(self.output)
        self.points["original-0"] = (392090, 5711600)
        self.write_snapshot(self.training, self.rows, self.points)
        count = len(self.calls)
        with self.assertRaisesRegex(bulk.BulkTrainingError, "reference binding differs"):
            self.acquire()
        self.assertEqual(len(self.calls), count)

    def test_corrupt_response_cache_aborts_without_refresh(self):
        self.acquire()
        import shutil
        shutil.rmtree(self.output)
        key = digest(nrw._map_url(nrw._bounds(self.points["original-0"])).encode())
        path = self.cache/key/"response.bin"
        if not path.exists():
            key = digest(nrw._map_url(nrw._bounds(self.points["original-1"])).encode())
            path = self.cache/key/"response.bin"
        path.write_bytes(b"corrupt")
        count = len(self.calls)
        with self.assertRaisesRegex(bulk.BulkTrainingError, "Immutable acquisition cache"):
            self.acquire()
        self.assertEqual(len(self.calls), count)

    def test_protocol_bounds_reject_before_network_or_reference_pixels(self):
        for field, value in (("max_workers", 5), ("max_attempts", 4), ("max_network_requests", 10001),
                              ("min_candidate_distance_m", 79), ("min_reference_distance_m", 5000),
                              ("min_known_distance_m", 199), ("request_interval_s", .1), ("max_known_candidates", 1601)):
            with self.subTest(field=field):
                original = self.rules[field]
                self.rules[field] = value
                self.write_protocol()
                with self.assertRaises(bulk.BulkTrainingError):
                    self.acquire()
                self.rules[field] = original
        self.assertEqual(self.calls, [])

    def test_scheduler_starts_queued_work_while_first_response_is_slow(self):
        third_started = threading.Event()
        first_finished = threading.Event()
        completed, accepted = [], []
        def acquire(candidate):
            if candidate == 0:
                self.assertTrue(third_started.wait(3), "Queued work must start before the oldest response finishes")
                first_finished.set()
            elif candidate == 2:
                self.assertFalse(first_finished.is_set())
                completed.append(candidate)
                third_started.set()
                return candidate
            completed.append(candidate)
            return candidate
        with ThreadPoolExecutor(max_workers=2) as executor:
            for result in bulk._ordered_acquisitions(range(100), executor, acquire,
                    lambda: 5-len(accepted), lambda candidate: True):
                accepted.append(result)
        self.assertEqual(accepted, list(range(5)))
        self.assertEqual(sorted(completed), list(range(5)))  # No lookahead beyond the quota.
        self.assertLess(completed.index(2), completed.index(0))

    def test_scheduler_bounds_retained_results_and_refills_rejected_slots(self):
        accepted, submitted = [], []
        outstanding, maximum = 0, 0
        class ImmediateExecutor:
            def submit(self, acquire, candidate):
                nonlocal outstanding, maximum
                from concurrent.futures import Future
                submitted.append(candidate)
                outstanding += 1
                maximum = max(maximum, outstanding)
                future = Future()
                future.set_result(acquire(candidate))
                return future
        for result in bulk._ordered_acquisitions(range(100), ImmediateExecutor(), lambda candidate: candidate,
                lambda: 35-len(accepted), lambda candidate: True):
            outstanding -= 1
            if result not in (0, 4, 9):
                accepted.append(result)
        self.assertEqual(len(accepted), 35)
        self.assertEqual(submitted, list(range(38)))
        self.assertEqual(maximum, 32)
        self.assertEqual(outstanding, 0)

    def test_acceptance_follows_stable_plan_not_worker_completion(self):
        self.rules.update(target_known=2, target_geometry=0)
        self.write_protocol()
        original = bulk._acquire
        def delayed(candidate, *args):
            if candidate["parent_training_id"] == "original-0":
                time.sleep(.01)
            return original(candidate, *args)
        with patch.object(bulk, "_acquire", side_effect=delayed):
            self.acquire()
        plan = json.loads(next((self.cache/"runs").glob("*/plan.json")).read_text())
        samples = [json.loads(line) for line in (self.output/"provenance"/"samples.jsonl").read_text().splitlines()]
        self.assertEqual([r["id"] for r in samples], [r["id"] for r in plan["known"]])


if __name__ == "__main__":
    unittest.main()
