# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Bounded, prediction-blind acquisition for a geographically separate review pool.

Exclusion snapshots supply metadata and coordinates only. Their raster paths,
including the existing test split, are never resolved, opened or decoded.
All newly acquired masks remain unassigned, unreviewed ignore placeholders.
"""

from __future__ import annotations

from collections import Counter
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
from urllib.parse import urlencode

from PIL import Image

from . import nrw
from .pv4ger import DatasetImportError, _publish, _write_json
from .snapshots import load_centers
from .validation import _invalid_constant, _unique_object, validate_dataset

MIN_REFERENCE_DISTANCE_M = 5000.0 + 32.0 * math.sqrt(2)
REGION_ID = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
PROTOCOL_FIELDS = {"schema_version", "dataset_version", "regions", "min_reference_distance_m", "min_candidate_distance_m"}


class NegativeHoldoutError(DatasetImportError):
    """A separate review pool cannot be acquired under its frozen protocol."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _protocol(path: Path, *, distance_reference_splits: tuple[str, ...] | None = None) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise NegativeHoldoutError("Expected a regular protocol JSON file of at most 64 KiB.")
    with path.open("rb") as stream:
        payload = stream.read(64 * 1024 + 1)
    if len(payload) > 64 * 1024:
        raise NegativeHoldoutError("Protocol exceeds its byte bound.")
    value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    fields = PROTOCOL_FIELDS if distance_reference_splits is None else PROTOCOL_FIELDS | {"distance_reference_splits"}
    if (not isinstance(value, dict) or set(value) != fields or value["schema_version"] != "1.0"
            or not isinstance(value["dataset_version"], str) or not value["dataset_version"].strip()
            or len(value["dataset_version"]) > 128):
        raise NegativeHoldoutError("Invalid protocol fields, schema or dataset version.")
    if distance_reference_splits is not None and value["distance_reference_splits"] != list(distance_reference_splits):
        raise NegativeHoldoutError("Protocol distance-reference splits differ from the fixed training selection rule.")
    for key, minimum in (("min_reference_distance_m", MIN_REFERENCE_DISTANCE_M), ("min_candidate_distance_m", 200.0)):
        if not _finite(value[key]) or not minimum <= value[key] <= 100000:
            raise NegativeHoldoutError(f"{key} must be finite and at least {minimum} metres.")
    regions = value["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= 3:
        raise NegativeHoldoutError("Protocol requires one to three regions.")
    seen, boxes = set(), set()
    for region in regions:
        if (not isinstance(region, dict) or set(region) != {"id", "bbox_crs84", "limit"}
                or not isinstance(region["id"], str) or not REGION_ID.fullmatch(region["id"])
                or region["id"] in seen or type(region["limit"]) is not int or not 1 <= region["limit"] <= 8):
            raise NegativeHoldoutError("Region IDs must be unique and each limit must be an integer from 1 to 8.")
        box = region["bbox_crs84"]
        if (not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box)
                or not (5.8 <= box[0] < box[2] <= 9.5 and 50.2 <= box[1] < box[3] <= 52.6)
                or box[2] - box[0] > 0.020000001 or box[3] - box[1] > 0.020000001
                or tuple(box) in boxes):
            raise NegativeHoldoutError("Regions require distinct small CRS84 boxes around NRW, at most 0.02 degrees per axis.")
        seen.add(region["id"])
        boxes.add(tuple(box))
    return value, payload


def fetch_negative_holdout_candidates(exclude_dirs: list[Path | str], output_dir: Path | str,
                                      *, protocol_path: Path | str) -> dict:
    """Fetch at most 24 native crops, without inspecting exclusion rasters.

    Geometry and stable source identifiers determine the frozen candidate slots.
    Metadata/identity exclusions never trigger replacement slots or hidden retries.
    Acquisition does not establish negative labels or an independent benchmark.
    """
    return _fetch_review_candidates(
        exclude_dirs, output_dir, protocol_path=protocol_path,
        dataset_id="nrw-negative-holdout-candidates", spatial_group_prefix="nrw-negative-holdout-",
        purpose=None, distance_reference_splits=None,
    )


def _fetch_review_candidates(exclude_dirs: list[Path | str], output_dir: Path | str, *,
                             protocol_path: Path | str, dataset_id: str,
                             spatial_group_prefix: str, purpose: str | None,
                             distance_reference_splits: tuple[str, ...] | None = None) -> dict:
    """Shared acquisition; wrappers fix the pool namespace and intended use.

    Identity exclusion and native-source evidence concern every reference row.
    Training wrappers may fix a narrower geographic-distance reference split set.
    No wrapper creates supervision or train rows.
    ``purpose=None`` retains the existing negative-pool report contract.
    """
    stage = None
    try:
        from pyproj import Transformer
        from .training import _read_manifest

        if not isinstance(exclude_dirs, (list, tuple)) or not 1 <= len(exclude_dirs) <= 8:
            raise NegativeHoldoutError("Provide one to eight complete exclusion snapshots.")
        sources = [Path(path).resolve(strict=True) for path in exclude_dirs]
        if len(set(sources)) != len(sources):
            raise NegativeHoldoutError("Exclusion snapshot paths must be distinct.")
        output = Path(output_dir).absolute()
        if os.path.lexists(output):
            raise NegativeHoldoutError(f"Output already exists: {output}")
        output = output.resolve()
        protocol_at = Path(protocol_path).absolute()
        if (any(output.is_relative_to(source) or source.is_relative_to(output) for source in sources)
                or protocol_at.resolve().is_relative_to(output)):
            raise NegativeHoldoutError("Output, protocol and exclusion snapshots must be separate.")
        protocol, protocol_bytes = _protocol(protocol_at, distance_reference_splits=distance_reference_splits)
        project = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        inverse = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        reference_points, reference_info = [], []
        identity_reference_records = 0
        excluded_ids, excluded_images, excluded_buildings, excluded_groups = set(), set(), set(), set()
        for source in sources:
            _, records, fingerprint = _read_manifest(source)
            centers, coordinate_hash = load_centers(source, records)
            identity_reference_records += len(records)
            for row in records:
                point = project.transform(*centers[row["id"]], errcheck=True)
                if not all(math.isfinite(v) for v in point):
                    raise NegativeHoldoutError("Invalid exclusion centre projection.")
                if distance_reference_splits is None or row["split"] in distance_reference_splits:
                    reference_points.append(point)
                excluded_ids.add(row["id"])
                excluded_images.add(row["image_sha256"])
                excluded_groups.add(row["spatial_group"])
                excluded_buildings.update(row["building_ids"])
            reference_info.append({"dataset_fingerprint": fingerprint, "coordinate_sha256": coordinate_hash,
                                   "records": len(records), "split_counts": dict(Counter(r["split"] for r in records))})
        if not reference_points:
            raise NegativeHoldoutError("At least one geographic-distance reference row is required.")
        groups = {region["id"]: spatial_group_prefix + region["id"] for region in protocol["regions"]}
        if set(groups.values()) & excluded_groups:
            raise NegativeHoldoutError("A requested region group already belongs to an exclusion snapshot.")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-fetch-", dir=output.parent))
        for name in ("images", "masks", "provenance"):
            (stage / name).mkdir()
        provenance = stage / "provenance"
        (provenance / "protocol.json").write_bytes(protocol_bytes)
        at = datetime.now(timezone.utc)
        today = at.date()
        notes = ("# Geographically separated NRW review candidates\n\n"
                 "Buildings: " + nrw.COLLECTION_URL + "\nImagery: " + nrw.WMS_URL + "\n"
                 "Geobasis NRW/openNRW; Datenlizenz Deutschland - Zero 2.0: " + nrw.LICENSE_URL + "\n"
                 "Source terms: https://www.bezreg-koeln.nrw.de/system/files/media/document/file/lizenzbedingungen_geobasis_nrw.pdf\n"
                 "Stable-ID geometry selection under the retained protocol precedes imagery review and all model predictions.\n"
                 "Exclusion snapshots are checked through metadata and coordinates only, across every declared split.\n"
                 "Native GSD and capture date require one official metadata tile covering the entire 32 m crop.\n"
                 "Exact RGB PNG response bytes are retained at the requested 320x320 grid, without resizing.\n"
                 "WMS rendering is not guaranteed to be a pixel-identical extraction from source orthophoto files.\n"
                 "Services are mutable; retrieval time is not acquisition time.\n"
                 "All masks are 255 ignore placeholders, unreviewed and unassigned. No negative label is inferred.\n"
                 "This geographically separated convenience sample is not representative or an independent labelled benchmark.\n")
        if purpose is not None:
            notes += ("Intended use: review for possible future training; this pool is not an evaluation holdout.\n"
                      "No PV or background supervision exists until a separate review and snapshot step.\n"
                      "ID, image hash, building and group exclusions include every reference row.\n")
            if distance_reference_splits is not None:
                notes += "Geographic distance-reference splits: " + ", ".join(distance_reference_splits) + ".\n"
        (provenance / "SOURCES.md").write_text(notes, encoding="utf-8")
        (stage / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = ' + json.dumps(dataset_id) + '\ndataset_version = '
            + json.dumps(protocol["dataset_version"]) + '\n\n[sources.nrw-dop]\n'
            + 'name = "Geobasis NRW current digital orthophotos"\nurl = ' + json.dumps(nrw.WMS_URL)
            + '\nversion = ' + json.dumps("snapshot-" + today.isoformat())
            + '\nimage_license = "dl-de-zero-2.0"\nannotation_license = "not-annotated-ignore-placeholder"\n'
            + 'attribution = "Geobasis NRW/openNRW. All-255 masks are unannotated placeholders, not negative labels."\n',
            encoding="utf-8")
        records, audit, region_reports, selected_centers = [], [], [], []
        selected_ids, accepted_hashes = set(), set()
        counts = Counter(building_requests=0, metadata_requests=0, image_requests=0)
        for region in protocol["regions"]:
            exclusions = Counter()
            box = region["bbox_crs84"]
            query = nrw.BUILDING_URL + "?" + urlencode({"f": "json", "bbox": ",".join(map(str, box)),
                "limit": str(nrw.MAX_FEATURES), "gebnutzbez": "Gebäude", "crs": nrw.CRS_URI})
            counts["building_requests"] += 1
            buildings, headers = nrw._fetch(query, nrw.MAX_JSON_BYTES)
            if headers.get("content-crs", "").strip("<>") != nrw.CRS_URI:
                raise NegativeHoldoutError("Building response does not confirm EPSG:25832 coordinates.")
            document = json.loads(buildings, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            features = document.get("features") if isinstance(document, dict) else None
            if (not isinstance(document, dict) or document.get("type") != "FeatureCollection"
                    or not isinstance(features, list) or len(features) > nrw.MAX_FEATURES
                    or type(document.get("numberReturned")) is not int or document["numberReturned"] != len(features)
                    or any(not isinstance(f, dict) or not isinstance(f.get("id"), str) for f in features)
                    or len({f["id"] for f in features}) != len(features)):
                raise NegativeHoldoutError("Unexpected, duplicate or oversized building response.")
            (provenance / f"buildings-{region['id']}.geojson").write_bytes(buildings)
            selected = []
            for feature in sorted(features, key=lambda f: f["id"]):
                geometry = nrw._roof(feature)
                if geometry is None:
                    exclusions["unsuitable_geometry_or_function"] += 1
                    continue
                center, area = geometry
                lonlat = inverse.transform(*center, errcheck=True)
                identifier, building_id = "nrw-" + feature["id"], "geobasis-nrw:alkis:" + feature["id"]
                if not (box[0] <= lonlat[0] <= box[2] and box[1] <= lonlat[1] <= box[3]):
                    exclusions["center_outside_query"] += 1
                elif identifier in excluded_ids or building_id in excluded_buildings or identifier in selected_ids:
                    exclusions["excluded_identity"] += 1
                else:
                    distance = min(math.dist(center, point) for point in reference_points)
                    if distance < protocol["min_reference_distance_m"]:
                        exclusions["near_exclusion_snapshot"] += 1
                    elif any(math.dist(center, point) < protocol["min_candidate_distance_m"] for point in selected_centers):
                        exclusions["near_selected_candidate"] += 1
                    else:
                        selected.append((feature, center, area, lonlat, distance))
                        selected_centers.append(center)
                        selected_ids.add(identifier)
                if len(selected) == region["limit"]:
                    break
            accepted = 0
            for slot, (feature, center, area, lonlat, distance) in enumerate(selected, 1):
                bounds = nrw._bounds(center)
                metadata_url = nrw._map_url(bounds, metadata=True)
                counts["metadata_requests"] += 1
                metadata_error = None
                try:
                    payload, _ = nrw._fetch(metadata_url, nrw.MAX_METADATA_BYTES)
                except (OSError, ValueError) as error:
                    payload, metadata_error = b"", str(error)
                metadata_file = f"metadata-{region['id']}-{slot}.gml"
                (provenance / metadata_file).write_bytes(payload)
                metadata = nrw._metadata(payload, bounds, today)
                decision = {"id": "nrw-" + feature["id"], "region": region["id"], "selection_slot": slot,
                            "metadata_url": metadata_url, "metadata_file": metadata_file,
                            "metadata_response_sha256": _digest(payload), "metadata_error": metadata_error, **metadata}
                if metadata["native_gsd_m"] != 0.1 or metadata["capture_date"] is None:
                    reason = "native_gsd_unverified" if metadata["native_gsd_m"] != 0.1 else "capture_date_unverified"
                    exclusions[reason] += 1
                    audit.append({**decision, "accepted": False, "exclusion": reason})
                    continue
                image_url = nrw._map_url(bounds)
                counts["image_requests"] += 1
                image_payload, _ = nrw._fetch(image_url, nrw.MAX_IMAGE_BYTES)
                image_hash = _digest(image_payload)
                with Image.open(BytesIO(image_payload)) as image:
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (320, 320):
                        raise NegativeHoldoutError("Expected an unmodified RGB 320x320 PNG response.")
                    image.verify()
                if image_hash in excluded_images or image_hash in accepted_hashes:
                    exclusions["excluded_image_hash"] += 1
                    audit.append({**decision, "accepted": False, "exclusion": "excluded_image_hash",
                                  "image_response_sha256": image_hash, "image_url": image_url})
                    continue
                accepted_hashes.add(image_hash)
                identifier = decision["id"]
                image_name, mask_name = f"images/{identifier}.png", f"masks/{identifier}.png"
                (stage / image_name).write_bytes(image_payload)
                Image.new("L", (320, 320), 255).save(stage / mask_name)
                records.append({"id": identifier, "image": image_name, "mask": mask_name,
                    "image_sha256": image_hash, "mask_sha256": _digest((stage / mask_name).read_bytes()),
                    "width": 320, "height": 320, "gsd_m": 0.1, "source_id": "nrw-dop",
                    "capture_date": metadata["capture_date"], "split": "unassigned", "annotation_status": "unreviewed",
                    "building_ids": ["geobasis-nrw:alkis:" + feature["id"]], "spatial_group": groups[region["id"]],
                    "georeference": {"crs": "EPSG:25832", "transform": [0.1, 0, bounds[0], 0, -0.1, bounds[3]]}})
                audit.append({**decision, "accepted": True, "official_building_id": feature["id"],
                    "building_area_m2": area, "building_aktualit": feature["properties"].get("aktualit"),
                    "center_crs": "EPSG:4326", "approximate_center": list(lonlat), "center_epsg25832": list(center),
                    "nearest_reference_distance_m": distance, "requested_bbox_epsg25832": list(bounds),
                    "image_url": image_url, "image_response_sha256": image_hash,
                    "image_response_bytes": len(image_payload), "retrieved_at": at.isoformat()})
                accepted += 1
            region_reports.append({"id": region["id"], "bbox_crs84": box, "requested_records": region["limit"],
                "candidate_slots": len(selected), "accepted_records": accepted, "selection_exhausted": accepted < region["limit"],
                "building_query_url": query, "building_response_sha256": _digest(buildings),
                "building_features_returned": len(features), "selection_exclusions": dict(exclusions)})
        if not records:
            raise NegativeHoldoutError("No candidates satisfy the frozen identity, separation and native metadata rules.")
        (stage / "manifest.jsonl").write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in records), encoding="utf-8")
        (provenance / "samples.jsonl").write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in audit if r["accepted"]), encoding="utf-8")
        _write_json(provenance / "acquisition-decisions.json", audit)
        for source, expected in zip(sources, reference_info):
            _, rows, fingerprint = _read_manifest(source)
            if fingerprint != expected["dataset_fingerprint"] or load_centers(source, rows)[1] != expected["coordinate_sha256"]:
                raise NegativeHoldoutError("Exclusion metadata or coordinates changed during acquisition.")
        validation = validate_dataset(stage)  # Only newly downloaded, unassigned candidate pixels.
        if not validation.ok:
            raise NegativeHoldoutError(f"New candidate validation failed: {validation.errors[0].message}")
        _write_json(stage / "validation-report.json", validation.to_dict())
        accepted_audit = [r for r in audit if r["accepted"]]
        report = {"ok": True, "records": len(records), "requested_records": sum(r["limit"] for r in protocol["regions"]),
            "dataset_fingerprint": validation.stats["dataset_fingerprint"], "protocol_sha256": _digest(protocol_bytes),
            "references": reference_info, "reference_snapshot_count": len(sources), "reference_center_count": identity_reference_records,
            "regions": region_reports, "network_requests": dict(counts), "network_request_count": sum(counts.values()),
            "minimum_reference_distance_required_m": protocol["min_reference_distance_m"],
            "minimum_reference_distance_observed_m": min(r["nearest_reference_distance_m"] for r in accepted_audit),
            "minimum_candidate_center_spacing_m": protocol["min_candidate_distance_m"],
            "excluded_reference_ids": len(excluded_ids), "excluded_reference_image_hashes": len(excluded_images),
            "excluded_reference_building_ids": len(excluded_buildings), "exclusion_rasters_opened": False,
            "snapshot_retrieved_at": at.isoformat(), "pv_pixels": 0, "background_pixels": 0,
            "ignored_pixels": len(records) * 320 * 320, "dated_records": len(records),
            "model_predictions_used": False, "validation": {"ok": True, "warning_count": validation.warning_count},
            "limitations": ["All masks are unreviewed ignore placeholders; no negative labels or benchmark scores exist.",
                "Geographic separation concerns supplied snapshots, not unknown pretraining imagery or acquisition campaigns.",
                "Stable-ID geometry sampling is a small convenience sample, not representative of roof prevalence.",
                "Mutable WMS responses preserve exact rendered bytes; requested grid and native metadata are separate evidence."]}
        if purpose is not None:
            report.update(purpose=purpose, intended_use="training-review", training_use_allowed=False,
                          identity_reference_records=identity_reference_records,
                          distance_reference_records=len(reference_points),
                          distance_reference_splits=list(distance_reference_splits) if distance_reference_splits is not None else None)
        _write_json(stage / "candidate-report.json", report)
        _publish(stage, output)
        stage = None
        return report
    except NegativeHoldoutError:
        raise
    except ImportError as error:
        raise NegativeHoldoutError("Install the train and geo extras for metadata-only exclusion and projection.") from error
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, RecursionError, OverflowError) as error:
        raise NegativeHoldoutError(f"Negative holdout acquisition failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
