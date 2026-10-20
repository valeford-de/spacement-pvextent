# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Resumable, prediction-blind acquisition of native NRW training-review images.

A frozen plan combines fresh captures at original PV4GER TRAIN coordinates with
new ALKIS roofs. Existing labels are never copied. Reference rasters are never
opened, except original TRAIN images explicitly guarded for decoded-RGB dedup.
The durable response cache and attempt ledger survive incomplete acquisitions.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from urllib.parse import urlencode

from PIL import Image

from . import nrw
from .acquisition_cache import AcquisitionCacheError, cached_fetch
from .pv4ger import DatasetImportError, _publish
from .snapshots import load_centers
from .validation import _invalid_constant, _unique_object, validate_dataset

MIN_REFERENCE_DISTANCE_M = 5000 + 32 * math.sqrt(2)
PROTOCOL_FIELDS = {"schema_version", "dataset_version", "seed", "target_known", "target_geometry",
                   "max_known_candidates", "regions", "min_reference_distance_m",
                   "min_candidate_distance_m", "min_known_distance_m", "max_workers",
                   "request_interval_s", "max_attempts", "max_network_requests"}
REGION_ID = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
NOTICE_NAMES = {"upstream-LICENSE", "original-MIT-LICENSE", "upstream-README", "SOURCES.md"}


class BulkTrainingError(DatasetImportError):
    """The frozen bulk acquisition cannot be completed or safely published."""


class _RequestFailed(BulkTrainingError):
    pass


class _BudgetExceeded(_RequestFailed):
    pass


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path, maximum=8 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise BulkTrainingError(f"Expected a bounded regular JSON file: {path}")
    payload = path.read_bytes()
    if len(payload) > maximum:
        raise BulkTrainingError("JSON byte bound exceeded.")
    return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _atomic_json(path: Path, value) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _protocol(path: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise BulkTrainingError("Expected a regular protocol JSON file of at most 64 KiB.")
    payload = path.read_bytes()
    if len(payload) > 64 * 1024:
        raise BulkTrainingError("Protocol exceeds its byte bound.")
    value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if (not isinstance(value, dict) or set(value) != PROTOCOL_FIELDS or value["schema_version"] != "1.0"
            or not isinstance(value["dataset_version"], str) or not 1 <= len(value["dataset_version"].strip()) <= 128):
        raise BulkTrainingError("Invalid bulk protocol schema, fields or dataset version.")
    for key, low, high in (("seed", 0, 2**63-1), ("target_known", 0, 2000), ("target_geometry", 0, 1600),
                            ("max_known_candidates", 0, 1600), ("max_workers", 1, 4),
                            ("max_attempts", 1, 3), ("max_network_requests", 1, 10000)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise BulkTrainingError(f"Invalid bounded protocol integer: {key}")
    if not 1 <= value["target_known"] + value["target_geometry"] <= 2000 or value["max_known_candidates"] < value["target_known"]:
        raise BulkTrainingError("Quotas require one to 2000 images and enough known candidate slots.")
    for key, low, high in (("min_reference_distance_m", MIN_REFERENCE_DISTANCE_M, 100000),
                           ("min_candidate_distance_m", 80, 10000), ("min_known_distance_m", 200, 10000),
                           ("request_interval_s", .35, 60)):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not low <= value[key] <= high:
            raise BulkTrainingError(f"Invalid finite protocol bound: {key}")
    regions = value["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 30:
        raise BulkTrainingError("Provide one to thirty bounded regions.")
    ids, boxes = set(), set()
    for region in regions:
        if (not isinstance(region, dict) or set(region) != {"id", "bbox_crs84", "limit"}
                or not isinstance(region["id"], str) or not REGION_ID.fullmatch(region["id"])
                or region["id"] in ids or type(region["limit"]) is not int or not 1 <= region["limit"] <= 100):
            raise BulkTrainingError("Region IDs must be distinct and limits must be integers from one to 100.")
        box = region["bbox_crs84"]
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or not (5.8 <= box[0] < box[2] <= 9.5 and 50.2 <= box[1] < box[3] <= 52.6)
                or box[2]-box[0] > .040000001 or box[3]-box[1] > .040000001 or tuple(box) in boxes):
            raise BulkTrainingError("Regions must be distinct NRW CRS84 boxes at most .04 degrees per axis.")
        ids.add(region["id"])
        boxes.add(tuple(box))
    slots = sum(r["limit"] for r in regions)
    if not value["target_geometry"] <= slots <= 1600:
        raise BulkTrainingError("Geometry slots must cover their quota and total at most 1600.")
    return value, payload


@contextmanager
def _run_lock(run: Path):
    import fcntl
    path = run / "run.lock"
    if path.is_symlink():
        raise BulkTrainingError("Run lock must not be a symbolic link.")
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BulkTrainingError("This protocol is already being acquired in another process.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class _Ledger:
    """Reserve and persist each wire attempt before calling the transport."""

    def __init__(self, run: Path, cache: Path, protocol: dict):
        self.run, self.cache, self.protocol = run, cache, protocol
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.state = _read_json(run / "state.json") if (run / "state.json").exists() else {
            "schema_version": "1.0", "network_requests": 0, "attempts": {}, "created_at_utc": _now()}
        if (not isinstance(self.state, dict) or self.state.get("schema_version") != "1.0"
                or type(self.state.get("network_requests")) is not int or self.state["network_requests"] < 0
                or not isinstance(self.state.get("attempts"), dict)
                or any(not re.fullmatch(r"[0-9a-f]{64}", k) or type(v) is not int or not 1 <= v <= protocol["max_attempts"]
                       for k, v in self.state["attempts"].items())
                or sum(self.state["attempts"].values()) != self.state["network_requests"]
                or self.state["network_requests"] > protocol["max_network_requests"]):
            raise BulkTrainingError("Durable request ledger has an invalid budget or attempt schema.")
        self.cache_hits, self.cache_lookups = 0, 0
        _atomic_json(run / "state.json", self.state)

    def _event(self, value):
        path = self.run / "requests.jsonl"
        if path.is_symlink():
            raise BulkTrainingError("Request log must not be a symbolic link.")
        with path.open("ab") as stream:
            stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def transport(self, url, maximum):
        key = _digest(url.encode())
        last_error = None
        while True:
            with self.lock:
                attempts = self.state["attempts"].get(key, 0)
                if attempts >= self.protocol["max_attempts"]:
                    raise _RequestFailed(f"Explicit per-URL attempt limit reached: {url}") from last_error
                if self.state["network_requests"] >= self.protocol["max_network_requests"]:
                    raise _BudgetExceeded("Frozen total network request budget exhausted.")
                delay = max(0.0, self.next_request-time.monotonic())
                if delay:
                    time.sleep(delay)
                self.next_request = time.monotonic()+self.protocol["request_interval_s"]
                self.state["attempts"][key] = attempts+1
                self.state["network_requests"] += 1
                event = {"url": url, "url_sha256": key, "attempt": attempts+1,
                         "network_request_number": self.state["network_requests"], "started_at_utc": _now()}
                _atomic_json(self.run / "state.json", self.state)
                self._event({**event, "outcome": "reserved"})
            try:
                payload, headers = nrw._fetch(url, maximum)
            except (OSError, ValueError, RuntimeError) as error:
                last_error = error
                with self.lock:
                    self._event({**event, "outcome": "failed", "finished_at_utc": _now(),
                                 "error_type": type(error).__name__, "error": str(error)[:1000]})
            else:
                with self.lock:
                    self._event({**event, "outcome": "received", "finished_at_utc": _now(),
                                 "response_sha256": _digest(payload), "response_bytes": len(payload)})
                return payload, headers

    def fetch(self, url, maximum):
        try:
            result = cached_fetch(url, maximum, cache_dir=self.cache, fetcher=self.transport)
        except AcquisitionCacheError as error:
            # Transport failures are candidate rejections; cache-integrity failures
            # abort, rather than causing an unlogged refresh of immutable evidence.
            if isinstance(error.__cause__, _RequestFailed):
                raise error.__cause__ from error
            raise BulkTrainingError(f"Immutable acquisition cache failed: {error}") from error
        with self.lock:
            self.cache_lookups += 1
            self.cache_hits += int(result[2]["cache_hit"])
        return result


def _references(training: Path, protected: list[Path], identity: list[Path], project):
    from .training import _read_manifest
    paths = [training, *protected, *identity]
    refs, all_rows, training_rows = [], [], None
    protected_points, identity_points, original_train_points = [], [], []
    ids, hashes, buildings, groups = set(), set(), set(), set()
    protected_ids, protected_buildings, protected_hashes = set(), set(), set()
    for index, source in enumerate(paths):
        _, rows, fingerprint = _read_manifest(source)
        centers, coordinates = load_centers(source, rows)
        if index == 0:
            training_rows = rows
        refs.append({"path": str(source), "dataset_fingerprint": fingerprint, "coordinate_sha256": coordinates,
                     "records": len(rows), "role": "training" if index == 0 else "protected" if source in protected else "identity-only",
                     "split_counts": dict(Counter(r["split"] for r in rows))})
        for row in rows:
            point = project.transform(*centers[row["id"]], errcheck=True)
            if not all(math.isfinite(v) for v in point):
                raise BulkTrainingError("Non-finite projected reference centre.")
            if index == 0 and row["split"] == "train":
                original_train_points.append(point)
            protected_row = (index == 0 and row["split"] != "train") or source in protected
            if protected_row:
                protected_points.append(point)
                protected_ids.add(row["id"])
                protected_hashes.add(row["image_sha256"])
                protected_buildings.update(row["building_ids"])
            if source in identity:
                identity_points.append(point)
            ids.add(row["id"])
            hashes.add(row["image_sha256"])
            buildings.update(row["building_ids"])
            groups.add(row["spatial_group"])
            all_rows.append((source, row, point))
    if not protected_points:
        raise BulkTrainingError("At least one metadata-only protected reference centre is required.")
    return {"info": refs, "rows": all_rows, "training_rows": training_rows, "protected_points": protected_points,
            "identity_points": identity_points, "original_train_points": original_train_points, "ids": ids, "hashes": hashes, "buildings": buildings,
            "groups": groups, "protected_ids": protected_ids, "protected_buildings": protected_buildings, "protected_hashes": protected_hashes}


def _near(point, points, distance):
    return any(math.dist(point, other) < distance for other in points)


def _candidate(identifier, kind, center, inverse, *, parent=None, region=None, buildings=None, area=None):
    bounds = nrw._bounds(center)
    actual = (bounds[0]+16, bounds[1]+16)
    return {"id": identifier, "kind": kind, "parent_training_id": parent, "region": region,
            "center_epsg25832": list(actual), "approximate_center": list(inverse.transform(*actual, errcheck=True)),
            "bounds": list(bounds), "building_ids": buildings or [], "building_area_m2": area}


def _known_plan(refs, protocol, inverse, rejections):
    cells = {}
    for source, row, center in refs["rows"]:
        if source != Path(refs["info"][0]["path"]) or row["split"] != "train" or row["source_id"] != "pv4ger-seg":
            continue
        candidate = _candidate("nrw-bulk-" + _digest(("known:"+row["id"]).encode())[:24], "known", center, inverse,
                               parent=row["id"], buildings=row["building_ids"])
        point, lonlat = candidate["center_epsg25832"], candidate["approximate_center"]
        if (not 5.8 <= lonlat[0] <= 9.5 or not 50.2 <= lonlat[1] <= 52.6
                or row["id"] in refs["protected_ids"] or row["image_sha256"] in refs["protected_hashes"]
                or set(row["building_ids"]) & refs["protected_buildings"]
                or candidate["id"] in refs["ids"]):
            rejections["known_source_or_identity_ineligible"] += 1
        elif _near(point, refs["protected_points"], protocol["min_reference_distance_m"]):
            rejections["known_near_protected_reference"] += 1
        elif _near(point, refs["identity_points"], protocol["min_candidate_distance_m"]):
            rejections["known_near_existing_training_pool"] += 1
        else:
            cell = (math.floor(point[0]/10000), math.floor(point[1]/10000))
            cells.setdefault(cell, []).append(candidate)
    seed = str(protocol["seed"])
    ordered = []
    for cell, rows in sorted(cells.items(), key=lambda pair: _digest((seed+repr(pair[0])).encode())):
        ordered.append(deque(sorted(rows, key=lambda r: _digest((seed+r["parent_training_id"]).encode()))))
    selected, centers = [], []
    while ordered and len(selected) < protocol["max_known_candidates"]:
        remaining = []
        for queue in ordered:
            candidate = queue.popleft()
            if _near(candidate["center_epsg25832"], centers, protocol["min_known_distance_m"]):
                rejections["known_seed_spacing"] += 1
            else:
                selected.append(candidate)
                centers.append(candidate["center_epsg25832"])
            if queue:
                remaining.append(queue)
            if len(selected) >= protocol["max_known_candidates"]:
                break
        ordered = remaining
    return selected


def _building_url(region):
    return nrw.BUILDING_URL + "?" + urlencode({"f": "json", "bbox": ",".join(map(str, region["bbox_crs84"])),
        "limit": str(nrw.MAX_FEATURES), "gebnutzbez": "Gebäude", "crs": nrw.CRS_URI})


def _geometry_plan(refs, protocol, inverse, ledger, rejections, emit):
    selected, centers, ids, regions = [], [], set(), []
    for region in protocol["regions"]:
        query = _building_url(region)
        report = {"id": region["id"], "query_url": query, "candidate_slots": 0}
        try:
            payload, headers, evidence = ledger.fetch(query, nrw.MAX_JSON_BYTES)
        except _RequestFailed as error:
            report["request_error"] = str(error)
            rejections["building_request_failed"] += 1
            regions.append(report)
            emit("planning-geometry")
            continue
        document = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        features = document.get("features") if isinstance(document, dict) else None
        if (headers.get("content-crs", "").strip("<>") != nrw.CRS_URI
                or not isinstance(document, dict) or document.get("type") != "FeatureCollection"
                or not isinstance(features, list) or len(features) > nrw.MAX_FEATURES
                or type(document.get("numberReturned")) is not int or document["numberReturned"] != len(features)
                or any(not isinstance(f, dict) or not isinstance(f.get("id"), str) for f in features)
                or len({f["id"] for f in features}) != len(features)):
            raise BulkTrainingError("Unexpected building response schema, CRS, size or duplicate IDs.")
        report.update(response=evidence, returned_features=len(features))
        box = region["bbox_crs84"]
        for feature in sorted(features, key=lambda row: row["id"]):
            roof = nrw._roof(feature)
            if roof is None:
                rejections["geometry_unsuitable"] += 1
                continue
            center, area = roof
            candidate = _candidate("nrw-bulk-"+_digest(("geometry:"+feature["id"]).encode())[:24], "geometry", center, inverse, region=region["id"],
                                   buildings=["geobasis-nrw:alkis:"+feature["id"]], area=area)
            point, lonlat = candidate["center_epsg25832"], candidate["approximate_center"]
            if not box[0] <= lonlat[0] <= box[2] or not box[1] <= lonlat[1] <= box[3]:
                rejections["geometry_outside_region"] += 1
            elif (candidate["id"] in refs["ids"] or "nrw-"+feature["id"] in refs["ids"]
                  or candidate["id"] in ids or set(candidate["building_ids"]) & refs["buildings"]):
                rejections["geometry_reference_identity"] += 1
            elif _near(point, refs["protected_points"], protocol["min_reference_distance_m"]):
                rejections["geometry_near_protected_reference"] += 1
            elif _near(point, centers+refs["identity_points"]+refs["original_train_points"], protocol["min_candidate_distance_m"]):
                rejections["geometry_candidate_spacing"] += 1
            else:
                selected.append(candidate)
                centers.append(point)
                ids.add(candidate["id"])
                report["candidate_slots"] += 1
            if report["candidate_slots"] >= region["limit"]:
                break
        regions.append(report)
        emit("planning-geometry")
    return selected, regions


def _training_rgb_hashes(training, rows):
    from .training import _raster
    hashes, count = set(), 0
    for row in rows:
        # This guard is deliberately adjacent to the only reference raster read.
        if row["split"] != "train":
            continue
        image = _raster(training, row, "image")
        hashes.add(_digest(image.tobytes()))
        count += 1
    return hashes, count


def _acquire(candidate, ledger, old_rgb_hashes, excluded_hashes):
    result = {"candidate": candidate, "accepted": False}
    bounds, center = candidate["bounds"], candidate["center_epsg25832"]
    tile_center = (math.floor(center[0]/1000)*1000+500, math.floor(center[1]/1000)*1000+500)
    metadata_url = nrw._map_url(nrw._bounds(tile_center), metadata=True)
    try:
        payload, _, evidence = ledger.fetch(metadata_url, nrw.MAX_METADATA_BYTES)
        metadata = nrw._metadata(payload, bounds, datetime.now(timezone.utc).date())
        result.update(metadata=metadata, metadata_response=evidence)
        if metadata["native_gsd_m"] != .1 or metadata["capture_date"] is None:
            result["rejection"] = "native_gsd_date_or_full_crop_coverage_unverified"
            return result
        payload, _, image_evidence = ledger.fetch(nrw._map_url(bounds), nrw.MAX_IMAGE_BYTES)
        result["image_response"] = image_evidence
    except _RequestFailed as error:
        result["rejection"], result["request_error"] = "request_failed", str(error)
        return result
    try:
        if len(payload) < 33 or payload[12:16] != b"IHDR" or payload[24] != 8:
            raise ValueError("Expected 8-bit PNG.")
        with Image.open(BytesIO(payload)) as image:
            if (image.format != "PNG" or image.mode != "RGB" or image.size != (320, 320)
                    or getattr(image, "n_frames", 1) != 1 or image.getexif().get(274, 1) != 1):
                raise ValueError("Expected a native, single-frame RGB 320x320 PNG.")
            image.load()
            rgb_hash = _digest(image.tobytes())
            if all(low == high for low, high in image.getextrema()):
                result["rejection"] = "uniform_or_empty_image"
                return result
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        result["rejection"], result["image_error"] = "invalid_native_image", str(error)
        return result
    if _digest(payload) in excluded_hashes or rgb_hash in old_rgb_hashes:
        result["rejection"] = "duplicate_reference_image"
        return result
    result.update(accepted=True, payload=payload, decoded_rgb_sha256=rgb_hash)
    return result


def _ordered_acquisitions(candidates, executor, acquire, remaining, eligible):
    """Feed workers continuously within a 32-result, quota-limited window.

    Futures remain in submission order, including completed later responses.
    The executor can start queued work while the oldest response is slow; only
    the main thread consumes/accepts results and updates the remaining quota.
    """
    candidates, pending, exhausted = iter(candidates), deque(), False
    while True:
        while not exhausted and len(pending) < min(32, remaining()):
            candidate = next(candidates, None)
            if candidate is None:
                exhausted = True
            elif eligible(candidate):
                pending.append(executor.submit(acquire, candidate))
        if not pending:
            return
        yield pending.popleft().result()


def _copy_notices(training, destination):
    copied = []
    provenance = training / "provenance"
    # Only explicitly named text notices are visited, never raster paths.
    for directory, subdirs, names in os.walk(provenance, followlinks=False):
        relative = Path(directory).relative_to(provenance)
        subdirs[:] = sorted(d for d in subdirs if len(relative.parts) < 5 and not (Path(directory)/d).is_symlink())
        for name in sorted(set(names) & NOTICE_NAMES):
            source = Path(directory)/name
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 2*1024*1024:
                raise BulkTrainingError("Upstream coordinate-source notices must be bounded regular text files.")
            payload = source.read_bytes()
            payload.decode("utf-8")
            target = destination / "coordinate-source-notices" / relative / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            copied.append({"file": str(target.relative_to(destination)), "sha256": _digest(payload)})
    return copied


def fetch_bulk_training(training_dir: Path | str, protected_dirs: list[Path | str],
                        identity_only_dirs: list[Path | str], output_dir: Path | str, *,
                        protocol_path: Path | str, cache_dir: Path | str,
                        progress=None, allow_partial: bool = False) -> dict:
    """Acquire fixed known-location/geometry quotas; publish only complete or opted-in partial pools.

    ``progress`` receives dictionaries. A retry/resume reuses the same frozen
    protocol, reference binding, candidate plan, exact responses and attempt
    budget. Every returned record is unassigned, with an all-255 ignore mask.
    """
    stage = None
    try:
        from pyproj import Transformer
        if type(allow_partial) is not bool or not isinstance(protected_dirs, (list, tuple)) or not isinstance(identity_only_dirs, (list, tuple)):
            raise BulkTrainingError("Expected protected/identity snapshot lists and a boolean partial flag.")
        if not 1 <= len(protected_dirs) <= 8 or len(identity_only_dirs) > 8:
            raise BulkTrainingError("Provide one to eight protected and at most eight identity-only snapshots.")
        training = Path(training_dir).resolve(strict=True)
        protected = [Path(p).resolve(strict=True) for p in protected_dirs]
        identity = [Path(p).resolve(strict=True) for p in identity_only_dirs]
        sources = [training, *protected, *identity]
        if len(set(sources)) != len(sources):
            raise BulkTrainingError("Reference snapshot paths must be distinct.")
        output = Path(output_dir).absolute()
        if os.path.lexists(output):
            raise BulkTrainingError(f"Output already exists: {output}")
        output = output.resolve()
        cache = Path(cache_dir).absolute()
        if cache.is_symlink():
            raise BulkTrainingError("Cache root must not be a symbolic link.")
        cache = cache.resolve()
        protocol_at = Path(protocol_path).absolute()
        protocol, protocol_bytes = _protocol(protocol_at)
        if (cache == output or cache.is_relative_to(output) or output.is_relative_to(cache)
                or any(output.is_relative_to(p) or p.is_relative_to(output) or cache.is_relative_to(p) or p.is_relative_to(cache) for p in sources)
                or protocol_at.resolve().is_relative_to(output)):
            raise BulkTrainingError("Output, cache, protocol and reference snapshots must be separate.")
        project = Transformer.from_crs(4326, 25832, always_xy=True)
        inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        refs = _references(training, protected, identity, project)
        cache.mkdir(parents=True, exist_ok=True)
        run = cache / "runs" / _digest(protocol_bytes)
        if (cache / "runs").is_symlink() or run.is_symlink():
            raise BulkTrainingError("Run directories must not be symbolic links.")
        run.mkdir(parents=True, exist_ok=True)
        binding = {"schema_version": "1.0", "protocol_sha256": _digest(protocol_bytes), "references": refs["info"]}
        with _run_lock(run):
            if (run / "binding.json").exists() and _read_json(run / "binding.json") != binding:
                raise BulkTrainingError("Resume reference binding differs from the frozen run.")
            _atomic_json(run / "binding.json", binding)
            (run / "protocol.json").write_bytes(protocol_bytes)
            ledger = _Ledger(run, cache, protocol)
            accepted = Counter(known=0, geometry=0)
            rejections = Counter()
            def emit(phase):
                snapshot = {"phase": phase, "known_accepted": accepted["known"], "geometry_accepted": accepted["geometry"],
                    "target_known": protocol["target_known"], "target_geometry": protocol["target_geometry"],
                    "network_requests": ledger.state["network_requests"], "cache_hits": ledger.cache_hits,
                    "cache_lookups": ledger.cache_lookups, "rejections": dict(rejections), "run_dir": str(run),
                    "updated_at_utc": _now()}
                _atomic_json(run / "progress.json", snapshot)
                if progress is not None:
                    progress(dict(snapshot))
            emit("reading-training-rgb-hashes")
            old_rgb, training_rasters = _training_rgb_hashes(training, refs["training_rows"])
            if (run / "plan.json").exists():
                plan = _read_json(run / "plan.json")
                if (not isinstance(plan, dict) or set(plan) != {"schema_version", "binding", "known", "geometry", "regions", "planning_rejections"}
                        or plan["schema_version"] != "1.0" or plan["binding"] != binding):
                    raise BulkTrainingError("Frozen candidate plan schema or reference binding differs.")
                # Recomputing deterministic known coordinates detects tampering before RGB.
                expected = _known_plan(refs, protocol, inverse, Counter())
                if plan["known"] != expected or len(plan["geometry"]) > sum(r["limit"] for r in protocol["regions"]):
                    raise BulkTrainingError("Frozen candidate plan differs from deterministic bounded selection.")
            else:
                known = _known_plan(refs, protocol, inverse, rejections)
                geometry, regions = _geometry_plan(refs, protocol, inverse, ledger, rejections, emit)
                plan = {"schema_version": "1.0", "binding": binding, "known": known, "geometry": geometry,
                        "regions": regions, "planning_rejections": dict(rejections)}
                _atomic_json(run / "plan.json", plan)
            plan_bytes = (run / "plan.json").read_bytes()
            plan_hash = _digest(plan_bytes)
            if (run / "plan.sha256").exists() and (run / "plan.sha256").read_text().strip() != plan_hash:
                raise BulkTrainingError("Frozen plan hash differs; candidate plan cannot be changed on resume.")
            (run / "plan.sha256").write_text(plan_hash+"\n")
            rejections = Counter(plan["planning_rejections"])
            emit("plan-sealed")  # All candidate slots are durable before the first RGB request.
            output.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix="."+output.name+"-bulk-", dir=output.parent))
            for name in ("images", "masks", "provenance"):
                (stage / name).mkdir()
            records, audit, centers, rgb_hashes, image_hashes = [], [], [], set(), set()
            mask_buffer = BytesIO()
            Image.new("L", (320, 320), 255).save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()
            with ThreadPoolExecutor(max_workers=protocol["max_workers"]) as executor:
                for kind in ("known", "geometry"):
                    def eligible(candidate):
                        if _near(candidate["center_epsg25832"], centers+refs["identity_points"], protocol["min_candidate_distance_m"]):
                            rejections["near_accepted_or_existing_training_pool"] += 1
                            audit.append({"candidate": candidate, "accepted": False, "rejection": "near_accepted_or_existing_training_pool"})
                            return False
                        return True
                    results = _ordered_acquisitions(plan[kind], executor,
                        lambda candidate: _acquire(candidate, ledger, old_rgb, refs["hashes"]),
                        lambda: protocol["target_"+kind]-accepted[kind], eligible)
                    for result in results:
                        candidate = result["candidate"]
                        payload = result.pop("payload", None)
                        if result["accepted"] and (result["decoded_rgb_sha256"] in rgb_hashes or _digest(payload) in image_hashes):
                            result.update(accepted=False, rejection="duplicate_new_image")
                        if result["accepted"] and _near(candidate["center_epsg25832"], centers, protocol["min_candidate_distance_m"]):
                            result.update(accepted=False, rejection="near_accepted_candidate")
                        if result["accepted"]:
                            identifier, bounds = candidate["id"], candidate["bounds"]
                            image_name, mask_name = "images/"+identifier+".png", "masks/"+identifier+".png"
                            (stage / image_name).write_bytes(payload)
                            (stage / mask_name).write_bytes(mask_bytes)
                            records.append({"id": identifier, "image": image_name, "mask": mask_name,
                                "image_sha256": _digest(payload), "mask_sha256": _digest(mask_bytes), "width": 320, "height": 320,
                                "gsd_m": .1, "capture_date": result["metadata"]["capture_date"], "source_id": "nrw-dop",
                                "spatial_group": "nrw-bulk-"+_digest(protocol_bytes)[:12]+"-"+str(math.floor(bounds[0]/1000))+"-"+str(math.floor(bounds[1]/1000)),
                                "building_ids": candidate["building_ids"], "split": "unassigned", "annotation_status": "unreviewed",
                                "georeference": {"crs": "EPSG:25832", "transform": [.1, 0, bounds[0], 0, -.1, bounds[3]]}})
                            rgb_hashes.add(result["decoded_rgb_sha256"])
                            image_hashes.add(_digest(payload))
                            centers.append(candidate["center_epsg25832"])
                            accepted[kind] += 1
                            result.update(id=identifier, center_crs="EPSG:4326", approximate_center=candidate["approximate_center"],
                                nearest_protected_reference_distance_m=min(math.dist(candidate["center_epsg25832"], p) for p in refs["protected_points"]))
                        else:
                            rejections[result["rejection"]] += 1
                        audit.append(result)
                        emit("acquiring-"+kind)
            complete = all(accepted[k] == protocol["target_"+k] for k in ("known", "geometry"))
            if not records or (not complete and not allow_partial):
                _atomic_json(run / "last-incomplete-report.json", {"complete": False, "accepted": dict(accepted),
                    "targets": {k: protocol["target_"+k] for k in ("known", "geometry")}, "rejections": dict(rejections),
                    "decisions": audit, "network_requests": ledger.state["network_requests"], "plan_sha256": plan_hash})
                emit("incomplete-not-published")
                raise BulkTrainingError(f"Frozen quotas incomplete: known {accepted['known']}/{protocol['target_known']}, geometry {accepted['geometry']}/{protocol['target_geometry']}; durable cache and plan retained at {run}. Explicit allow_partial is required to publish fewer images.")
            provenance = stage / "provenance"
            (provenance / "protocol.json").write_bytes(protocol_bytes)
            (provenance / "selection-plan.json").write_bytes(plan_bytes)
            _atomic_json(provenance / "reference-binding.json", binding)
            _atomic_json(provenance / "acquisition-decisions.json", audit)
            notices = _copy_notices(training, provenance)
            evidence = {}
            for region in plan["regions"]:
                if "response" in region:
                    evidence[region["response"]["url_sha256"]] = region["response"]
            for decision in audit:
                for name in ("metadata_response", "image_response"):
                    if name in decision:
                        evidence[decision[name]["url_sha256"]] = decision[name]
            response_dir = provenance / "responses"
            response_dir.mkdir()
            for key, item in evidence.items():
                # Copy exact cache bytes; immutable entries must never gain hardlinks.
                payload, _, verified = cached_fetch(item["url"], item["maximum_bytes_at_fetch"], cache_dir=cache,
                                                     fetcher=lambda *_: (_ for _ in ()).throw(BulkTrainingError("Missing retained response.")))
                if verified["response_sha256"] != item["response_sha256"]:
                    raise BulkTrainingError("Retained response hash differs from acquisition evidence.")
                (response_dir / (key+".bin")).write_bytes(payload)
            _atomic_json(provenance / "response-evidence.json", list(evidence.values()))
            assembled_at = _now()
            (provenance / "SOURCES.md").write_text(
                "# Spacement PVExtent NRW training-review images\n\nBuildings: "+nrw.COLLECTION_URL+"\nImagery: "+nrw.WMS_URL+
                "\nGeobasis NRW/openNRW; Datenlizenz Deutschland Zero 2.0: "+nrw.LICENSE_URL+
                "\nSource terms: https://www.bezreg-koeln.nrw.de/system/files/media/document/file/lizenzbedingungen_geobasis_nrw.pdf\n"
                "Original PV4GER TRAIN coordinates are used solely as recapture seeds. Their upstream coordinate and MIT notices are retained.\n"
                "Known-location recaptures can share a TRAIN parent identity; they are not new independent roofs or guaranteed PV-positive images.\n"
                "No original masks are read or transferred. No pretrained model or predictions are used.\n"
                "All new masks are 255 ignore placeholders, unreviewed and unassigned; these are images for review, not labelled training examples.\n"
                "Only original TRAIN images are decoded to exclude identical RGB; val/test/pilot images and every reference mask remain unopened.\n"
                "Exact mutable-service response bytes, URLs, hashes and original fetched_at_utc are retained; assembly time is separate.\n"
                "The native 0.1 m grid is verified against a dated metadata tile covering the whole crop. No resizing occurs.\n"
                "WMS renderings need not equal original orthophoto file pixels. Selection is a convenience sample, not an evaluation benchmark.\n",
                encoding="utf-8")
            (stage / "dataset.toml").write_text('schema_version = "1.0"\ndataset_id = "nrw-training-bulk"\ndataset_version = '+json.dumps(protocol["dataset_version"])+
                '\n\n[sources.nrw-dop]\nname = "Geobasis NRW current digital orthophotos"\nurl = '+json.dumps(nrw.WMS_URL)+
                '\nversion = '+json.dumps("assembled-"+assembled_at[:10])+'\nimage_license = "dl-de-zero-2.0"\nannotation_license = "not-annotated-ignore-placeholder"\n'
                'attribution = "Geobasis NRW/openNRW. Original TRAIN coordinate notices retained. All masks are unannotated ignore placeholders."\n', encoding="utf-8")
            (stage / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in records), encoding="utf-8")
            (provenance / "samples.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in audit if r["accepted"]), encoding="utf-8")
            final_refs = _references(training, protected, identity, project)
            if final_refs["info"] != refs["info"]:
                raise BulkTrainingError("Reference metadata or coordinates changed during acquisition.")
            validation = validate_dataset(stage)  # New unassigned pixels only.
            if not validation.ok:
                raise BulkTrainingError("New dataset validation failed: "+validation.errors[0].message)
            report = {"ok": True, "complete": complete, "records": len(records), "dataset_id": "nrw-training-bulk",
                "dataset_fingerprint": validation.stats["dataset_fingerprint"], "protocol_sha256": _digest(protocol_bytes),
                "selection_plan_sha256": plan_hash, "quotas": {k: {"target": protocol["target_"+k], "accepted": accepted[k], "candidate_slots": len(plan[k])} for k in ("known", "geometry")},
                "references": refs["info"], "distance_reference_records": len(refs["protected_points"]),
                "identity_reference_records": len(refs["rows"]), "identity_only_center_count": len(refs["identity_points"]),
                "network_request_count": ledger.state["network_requests"], "max_network_requests": protocol["max_network_requests"],
                "cache_hits_this_assembly": ledger.cache_hits, "cache_lookups_this_assembly": ledger.cache_lookups,
                "rejections": dict(rejections), "regions": plan["regions"], "assembled_at_utc": assembled_at,
                "response_count": len(evidence), "source_notices": notices, "run_dir": str(run),
                "original_training_images_decoded": training_rasters, "protected_rasters_opened": False,
                "reference_masks_opened": False, "original_masks_transferred": False, "model_predictions_used": False,
                "training_use_allowed": False, "intended_use": "training-review", "labelled_examples": 0,
                "ignored_pixels": len(records)*320*320, "pv_pixels": 0, "background_pixels": 0,
                "minimum_reference_distance_required_m": protocol["min_reference_distance_m"],
                "minimum_reference_distance_observed_m": min(r["nearest_protected_reference_distance_m"] for r in audit if r["accepted"]),
                "minimum_candidate_center_spacing_m": protocol["min_candidate_distance_m"],
                "minimum_known_seed_spacing_m": protocol["min_known_distance_m"],
                "validation": {"ok": True, "warning_count": validation.warning_count}}
            if (run / "requests.jsonl").exists():
                shutil.copyfile(run / "requests.jsonl", provenance / "request-attempts.jsonl")
            _atomic_json(provenance / "request-budget-state.json", ledger.state)
            _atomic_json(stage / "validation-report.json", validation.to_dict())
            _atomic_json(stage / "candidate-report.json", report)
            _publish(stage, output)
            stage = None
            emit("published" if complete else "partial-published")
            return report
    except BulkTrainingError:
        raise
    except ImportError as error:
        raise BulkTrainingError("Install train and geo extras for bulk acquisition.") from error
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, RecursionError, OverflowError) as error:
        raise BulkTrainingError(f"Bulk acquisition failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
