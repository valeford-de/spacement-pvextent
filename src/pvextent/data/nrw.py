# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Bounded, unlabelled NRW roof candidates for manual annotation review.

The generated 255 masks contain no supervised pixels. Neither building function
nor imagery metadata is used to infer PV presence or absence.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET

from PIL import Image

from .pv4ger import DatasetImportError, SOURCE_FILES, _partition, _publish, _verify
from .validation import validate_dataset

BUILDING_URL = "https://ogc-api.nrw.de/lika/v1/collections/gebaeude_bauwerk/items"
COLLECTION_URL = "https://ogc-api.nrw.de/lika/v1/collections/gebaeude_bauwerk"
WMS_URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
LICENSE_URL = "https://www.govdata.de/dl-de/zero-2-0"
CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/25832"
DEFAULT_BBOX = (7.435, 51.545, 7.445, 51.555)
MAX_FEATURES = 500
MAX_CANDIDATES = 24
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
PATCH_M = 32.0
PATCH_PIXELS = 320
MIN_CENTER_DISTANCE_M = 80.0
MIN_SOURCE_DISTANCE_M = 1000.0 + PATCH_M * math.sqrt(2)
BUILDING_ID = re.compile(r"DENW[A-Za-z0-9]{8,40}\Z")
ACCESSORY = re.compile(r"garage|carport|schuppen|gartenhaus|gewächshaus|transformator|toilette", re.I)

NOTES = """# NRW candidate provenance

Buildings: official simplified ALKIS, Geobasis NRW:
https://ogc-api.nrw.de/lika/v1/collections/gebaeude_bauwerk
The collection declares Datenlizenz Deutschland - Zero 2.0:
https://www.govdata.de/dl-de/zero-2-0
Coordinates were requested in official EPSG:25832. Feature IDs are the source
object identifier with its publisher-provided BL suffix, not roof segment IDs.
The property aktualit describes object currency, not image acquisition.

Imagery: Geobasis NRW current DOP, layer nw_dop_rgb:
https://www.wms.nrw.de/geobasis/wms_nw_dop
Terms (March 2020):
https://www.bezreg-koeln.nrw.de/system/files/media/document/file/lizenzbedingungen_geobasis_nrw.pdf
Imagery is under Datenlizenz Deutschland - Zero 2.0. Requested WMS images are
32 m x 32 m in EPSG:25832, sampled at 0.1 m/pixel on a snapped 0.1 m grid.
This requested/rendered sampling does not prove native 10 cm source detail or
pixel-identical extraction from original orthophotos. Native GSD is recorded
only when returned by the metadata layer. No sharpening or resizing is used.
Acquisition date is populated only when one dated metadata tile contains the
entire patch. Missing or ambiguous metadata yields null. Mosaic seams and
roof alignment still require review. WMS and building responses are mutable;
stored response bytes, hashes, request URLs and retrieval time identify this
snapshot. Retrieval time is never substituted for acquisition time.

Selection is a small geographic convenience sample in Dortmund-Nord, filtered
by building geometry/function, stable ID and center spacing, not by PV labels.
Each selected center is separated from ALL pinned PV4GER-SEG v0.9.1 centers by
at least 1000 m plus a 32 m patch diagonal. This does not establish independence
from unknown training data, imagery acquisition campaigns, or published models.
Building footprints do not establish roof boundaries or complete PV coverage.

All masks are 255 ignore placeholders, all records unreviewed and unassigned.
They contain no annotated positive or negative pixels. Human review must add
complete masks and separately establish positive/negative/hard-negative status.
"""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirects are not allowed", headers, fp)


def _fetch(url: str, maximum: int) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "Spacement-PVExtent/0.1 source-review"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=30) as response:
            payload = response.read(maximum + 1)
            if len(payload) > maximum:
                raise DatasetImportError(f"Official response exceeds {maximum} bytes: {url}")
            return payload, {key.lower(): value for key, value in response.headers.items()}
    except OSError as error:
        # Fixed public source URLs provide useful stage diagnostics without
        # exposing credentials. This path never retries or increases the budget.
        raise DatasetImportError(f"Official source request failed: {url}: {error}") from error


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _ring_area_center(ring: list) -> tuple[float, tuple[float, float]]:
    # Shift the large UTM coordinates before summation to avoid cancellation.
    ox, oy = ring[0]
    shifted = [(x - ox, y - oy) for x, y in ring]
    cross = [a[0] * b[1] - b[0] * a[1] for a, b in zip(shifted, shifted[1:])]
    twice = sum(cross)
    if abs(twice) < 1e-8:
        raise ValueError("Degenerate polygon")
    x = sum((a[0] + b[0]) * value for a, b, value in zip(shifted, shifted[1:], cross)) / (3 * twice)
    y = sum((a[1] + b[1]) * value for a, b, value in zip(shifted, shifted[1:], cross)) / (3 * twice)
    return abs(twice) / 2, (ox + x, oy + y)


def _in_ring(point: tuple[float, float], ring: list) -> bool:
    x, y = point
    inside = False
    for a, b in zip(ring, ring[1:]):
        if (a[1] > y) != (b[1] > y):
            intersection = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < intersection:
                inside = not inside
    return inside


def _inside(point: tuple[float, float], polygon: list) -> bool:
    return _in_ring(point, polygon[0]) and not any(_in_ring(point, hole) for hole in polygon[1:])


def _roof(feature: dict) -> tuple[tuple[float, float], float] | None:
    """Select a small complete building footprint with an interior center."""
    if not isinstance(feature, dict) or not BUILDING_ID.fullmatch(str(feature.get("id", ""))):
        return None
    properties = feature.get("properties", {})
    if (not isinstance(properties, dict) or properties.get("gebnutzbez") != "Gebäude"
            or ACCESSORY.search(str(properties.get("funktion", "")))
            or properties.get("rellage") == "Unter der Erdoberfläche"):
        return None
    geometry = feature.get("geometry", {})
    if not isinstance(geometry, dict) or geometry.get("type") != "MultiPolygon":
        return None
    polygons = geometry.get("coordinates")
    if not isinstance(polygons, list) or not 1 <= len(polygons) <= 20:
        return None
    try:
        points = []
        candidates = []
        total_area = 0.0
        for polygon in polygons:
            if not isinstance(polygon, list) or not 1 <= len(polygon) <= 20:
                return None
            areas = []
            for ring in polygon:
                if (not isinstance(ring, list) or not 4 <= len(ring) <= 1000
                        or ring[0] != ring[-1]):
                    return None
                for point in ring:
                    if (not isinstance(point, list) or len(point) != 2
                            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not math.isfinite(v) for v in point)
                            or not (250000 < point[0] < 600000 and 5500000 < point[1] < 5900000)):
                        return None
                points.extend(ring)
                areas.append(_ring_area_center(ring))
            area = areas[0][0] - sum(a[0] for a in areas[1:])
            if area <= 0:
                return None
            total_area += area
            center = tuple((areas[0][0] * areas[0][1][i]
                            - sum(a[0] * a[1][i] for a in areas[1:])) / area for i in (0, 1))
            candidates.append((area, center, polygon))
        if not 40 <= total_area <= 650:
            return None
        minx, maxx = min(p[0] for p in points), max(p[0] for p in points)
        miny, maxy = min(p[1] for p in points), max(p[1] for p in points)
        if maxx - minx > 28 or maxy - miny > 28:
            return None
        for _, centroid, polygon in sorted(candidates, key=lambda item: -item[0]):
            # A concave footprint's centroid may be outside: a bounded interior
            # grid gives a representative point, and inclusion is always checked.
            choices = [centroid, ((minx + maxx) / 2, (miny + maxy) / 2)]
            choices += [(minx + (maxx - minx) * i / 10, miny + (maxy - miny) * j / 10)
                        for i in range(1, 10) for j in range(1, 10)]
            for center in choices:
                if (_inside(center, polygon)
                        and max(abs(x - center[0]) for x, _ in points) <= 15
                        and max(abs(y - center[1]) for _, y in points) <= 15):
                    return center, total_area
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return None
    return None


def _bounds(center: tuple[float, float]) -> tuple[float, float, float, float]:
    # Integer decimeters make the requested pixel grid reproducible.
    left, bottom = (math.floor((value - PATCH_M / 2) * 10) for value in center)
    return left / 10, bottom / 10, (left + 320) / 10, (bottom + 320) / 10


def _map_url(bounds: tuple, *, metadata: bool = False) -> str:
    query = {"SERVICE": "WMS", "VERSION": "1.3.0", "CRS": "EPSG:25832",
             "BBOX": ",".join(f"{value:.1f}" for value in bounds), "WIDTH": "320", "HEIGHT": "320",
             "FORMAT": "image/png", "STYLES": ""}
    if metadata:
        query.update(REQUEST="GetFeatureInfo", LAYERS="nw_dop_utm_info", QUERY_LAYERS="nw_dop_utm_info",
                     I="160", J="160", INFO_FORMAT="application/vnd.ogc.gml", FEATURE_COUNT="3")
    else:
        query.update(REQUEST="GetMap", LAYERS="nw_dop_rgb", TRANSPARENT="FALSE")
    return WMS_URL + "?" + urlencode(query)


def _metadata(payload: bytes, bounds: tuple, today: date) -> dict:
    result = {"capture_date": None, "native_gsd_m": None, "tile_id": None,
              "date_basis": "unknown", "coverage": "unverified"}
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        return result
    try:
        root = ET.fromstring(payload)
        rows = root.findall(".//nw_dop_utm_info_feature")
        if len(rows) != 1:
            return result
        row = rows[0]
        box = row.find(".//{http://www.opengis.net/gml}Box")
        coordinates = box.findtext("{http://www.opengis.net/gml}coordinates") if box is not None else None
        if box is None or box.get("srsName") != "EPSG:25832" or not coordinates:
            return result
        values = [float(v) for v in coordinates.replace(",", " ").split()]
        if (len(values) != 4 or not all(math.isfinite(v) for v in values)
                or not (values[0] <= bounds[0] < bounds[2] <= values[2]
                        and values[1] <= bounds[1] < bounds[3] <= values[3])):
            return result
        result["coverage"] = "metadata_tile_contains_patch"
        result["tile_id"] = row.findtext("Kachelname")
        gsd = float(row.findtext("Bodenauflösung", "nan"))
        if math.isfinite(gsd) and 0 < gsd <= 1:
            result["native_gsd_m"] = gsd
        raw = row.findtext("Bildflugdatum", "")
        parsed = datetime.strptime(raw, "%d.%m.%Y").date()
        if date(1900, 1, 1) <= parsed <= today:
            result["capture_date"] = parsed.isoformat()
            result["date_basis"] = "single_metadata_tile_contains_patch"
    except (ET.ParseError, ValueError, TypeError, OverflowError):
        pass
    return result


def fetch_nrw_roof_candidates(
    source_dataset_dir: Path | str, output_dir: Path | str, *, limit: int = 24,
) -> dict[str, Any]:
    """Fetch at most 24 roof images; their masks remain entirely ignored.

    Network access is explicit in this command. One building request, at most
    24 imagery requests and 24 cached tile metadata requests are made, without
    retries; individual requests time out after 30 seconds.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CANDIDATES:
        raise DatasetImportError(f"--limit must be an integer between 1 and {MAX_CANDIDATES}.")
    try:
        from pyproj import Transformer
    except ImportError as error:
        raise DatasetImportError("Candidate dependencies missing. Run: uv sync --locked --extra geo") from error
    source, output = Path(source_dataset_dir).resolve(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise DatasetImportError(f"Output already exists: {output}")
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise DatasetImportError("Source and candidate output must be separate directories.")
    staging = None
    try:
        validation = validate_dataset(source)
        if not validation.ok or validation.stats["records"] != 4027:
            raise DatasetImportError("Candidates require the complete validated 4,027-record PV4GER dataset.")
        partition = source / "provenance" / "default_partition.json"
        expected = next(item for item in SOURCE_FILES if item.name == "default_partition.json")
        _verify(partition, expected)
        _, source_coordinates = _partition(partition)
        project = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        inverse = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        reference = {key: project.transform(*coordinate) for key, coordinate in source_coordinates.items()}
        query = BUILDING_URL + "?" + urlencode({
            "f": "json", "bbox": ",".join(map(str, DEFAULT_BBOX)), "limit": str(MAX_FEATURES),
            "gebnutzbez": "Gebäude", "crs": CRS_URI,
        })
        buildings, headers = _fetch(query, MAX_JSON_BYTES)
        if headers.get("content-crs", "").strip("<>") != CRS_URI:
            raise DatasetImportError("Building response does not confirm EPSG:25832 coordinates.")
        document = json.loads(buildings)
        if not isinstance(document, dict):
            raise DatasetImportError("Expected an official building FeatureCollection object.")
        features = document.get("features")
        if (document.get("type") != "FeatureCollection" or not isinstance(features, list)
                or len(features) > MAX_FEATURES or document.get("numberReturned") != len(features)
                or any(not isinstance(feature, dict) for feature in features)):
            raise DatasetImportError("Unexpected or oversized official building response.")
        identifiers = [feature.get("id") for feature in features if isinstance(feature, dict)]
        if len(set(identifiers)) != len(identifiers):
            raise DatasetImportError("Duplicate official building identifiers.")
        selected, exclusions = [], Counter()
        for feature in sorted(features, key=lambda value: str(value.get("id", ""))):
            geometry = _roof(feature)
            if geometry is None:
                exclusions["unsuitable_geometry_or_function"] += 1
                continue
            center, area = geometry
            lon, lat = inverse.transform(*center)
            if not (DEFAULT_BBOX[0] <= lon <= DEFAULT_BBOX[2] and DEFAULT_BBOX[1] <= lat <= DEFAULT_BBOX[3]):
                exclusions["center_outside_query"] += 1
                continue
            distance, nearest = min((math.dist(center, point), key) for key, point in reference.items())
            if distance < MIN_SOURCE_DISTANCE_M:
                exclusions["near_source_dataset"] += 1
                continue
            if any(math.dist(center, item[1]) < MIN_CENTER_DISTANCE_M for item in selected):
                exclusions["near_selected_candidate"] += 1
                continue
            selected.append((feature, center, area, (lon, lat), distance, nearest))
            if len(selected) == limit:
                break
        if not selected:
            raise DatasetImportError("No roof candidates satisfy the geometry and separation rules.")
        at = datetime.now(timezone.utc)
        today = at.date()
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-fetch-", dir=output.parent))
        for name in ("images", "masks", "provenance"):
            (staging / name).mkdir()
        provenance = staging / "provenance"
        (provenance / "buildings.geojson").write_bytes(buildings)
        (provenance / "SOURCES.md").write_text(NOTES, encoding="utf-8")
        (staging / "dataset.toml").write_text(f'''schema_version = "1.0"
dataset_id = "nrw-roof-candidates"
dataset_version = "snapshot-{today.isoformat()}-v1"

[sources.nrw-dop]
name = "Geobasis NRW current digital orthophotos"
url = "{WMS_URL}"
version = "snapshot-{today.isoformat()}"
image_license = "dl-de-zero-2.0"
annotation_license = "not-annotated-ignore-placeholder"
attribution = "Geobasis NRW/openNRW. All-255 masks are unannotated placeholders; no PV presence or absence is declared."
''', encoding="utf-8")
        records, audit, accepted = [], [], []
        metadata_cache = {}
        counts = Counter(building_requests=1)
        for feature, center, area, lonlat, distance, nearest in selected:
            bounds = _bounds(center)
            identifier = "nrw-" + feature["id"]
            key = tuple(math.floor(v / 1000) for v in center)
            if key not in metadata_cache:
                url = _map_url(bounds, metadata=True)
                try:
                    payload, _ = _fetch(url, MAX_METADATA_BYTES)
                except (OSError, ValueError) as error:
                    payload = b""
                    metadata_error = str(error)
                else:
                    metadata_error = None
                counts["metadata_requests"] += 1
                name = f"metadata-{key[0]}-{key[1]}.gml"
                (provenance / name).write_bytes(payload)
                metadata_cache[key] = (payload, url, name, metadata_error)
            payload, metadata_url, metadata_file, metadata_error = metadata_cache[key]
            metadata = _metadata(payload, bounds, today)
            if metadata["native_gsd_m"] != 0.1:
                exclusions["native_gsd_unknown_or_not_10cm"] += 1
                continue
            image_url = _map_url(bounds)
            image_payload, _ = _fetch(image_url, MAX_IMAGE_BYTES)
            counts["image_requests"] += 1
            with Image.open(BytesIO(image_payload)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (320, 320):
                    raise DatasetImportError(f"{identifier}: expected an RGB 320x320 PNG response.")
                image.verify()
            image_path, mask_path = f"images/{identifier}.png", f"masks/{identifier}.png"
            (staging / image_path).write_bytes(image_payload)
            Image.new("L", (320, 320), color=255).save(staging / mask_path)
            records.append({
                "id": identifier, "image": image_path, "mask": mask_path,
                "image_sha256": _digest(image_payload), "mask_sha256": _digest((staging / mask_path).read_bytes()),
                "width": 320, "height": 320, "gsd_m": 0.1, "source_id": "nrw-dop",
                "capture_date": metadata["capture_date"], "split": "unassigned", "annotation_status": "unreviewed",
                "building_ids": ["geobasis-nrw:alkis:" + feature["id"]],
                "spatial_group": "nrw-dortmund-nord-review-block-v1",
                "georeference": {"crs": "EPSG:25832", "transform": [0.1, 0, bounds[0], 0, -0.1, bounds[3]]},
            })
            accepted.append((feature, center, area, lonlat, distance, nearest))
            audit.append({
                "id": identifier, "official_building_id": feature["id"], "building_area_m2": area,
                "building_geometry_crs": "EPSG:25832", "building_aktualit": feature["properties"].get("aktualit"),
                "center_crs": "EPSG:4326", "approximate_center": list(lonlat), "center_epsg25832": list(center),
                "nearest_source_center_id": nearest, "nearest_source_distance_m": distance,
                "image_url": image_url, "image_response_sha256": _digest(image_payload),
                "image_response_bytes": len(image_payload), "requested_bbox_epsg25832": list(bounds),
                "requested_rendered_gsd_m": 0.1, "metadata_url": metadata_url, "metadata_file": metadata_file,
                "metadata_response_sha256": _digest(payload), "metadata_error": metadata_error,
                "retrieved_at": at.isoformat(), **metadata,
                "sampling_warning": "Rendered 0.1 m/pixel; native detail only as separately reported by source metadata.",
            })
        if not records:
            raise DatasetImportError("No candidate patches have verified native 10 cm source metadata.")
        (staging / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records), encoding="utf-8")
        (provenance / "samples.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in audit), encoding="utf-8")
        result = validate_dataset(staging)
        if not result.ok:
            issue = result.errors[0]
            raise DatasetImportError(f"Candidate dataset failed validation: {issue.code}: {issue.message}")
        _json(staging / "validation-report.json", result.to_dict())
        report = {
            "ok": True, "records": len(records), "requested_records": limit,
            "selection_exhausted": len(records) < limit, "snapshot_retrieved_at": at.isoformat(),
            "dataset_fingerprint": result.stats["dataset_fingerprint"],
            "source_dataset_fingerprint": validation.stats["dataset_fingerprint"],
            "source_coordinates_sha256": expected.sha256, "source_center_count": len(reference),
            "building_query_url": query, "building_response_sha256": _digest(buildings),
            "building_features_returned": len(features), "building_features_matched": document.get("numberMatched"),
            "building_response_truncated": bool(document.get("numberMatched", len(features)) > len(features)),
            "building_license_url": LICENSE_URL, "query_bbox_crs84": list(DEFAULT_BBOX),
            "selection_exclusions": dict(exclusions), "minimum_source_distance_required_m": MIN_SOURCE_DISTANCE_M,
            "minimum_source_distance_observed_m": min(item[4] for item in accepted),
            "minimum_candidate_center_spacing_m": MIN_CENTER_DISTANCE_M,
            "network_requests": dict(counts), "network_request_count": sum(counts.values()),
            "ignored_pixels": len(records) * 320 * 320, "pv_pixels": 0, "background_pixels": 0,
            "dated_records": sum(item["capture_date"] is not None for item in audit),
            "validation": {"ok": result.ok, "warning_count": result.warning_count},
            "limitations": ["Unreviewed ignore masks: no positive, negative or hard-negative labels exist.",
                            "Convenience sample from one geographic block; not an accuracy benchmark.",
                            "Separation from PV4GER centers does not prove independence from unknown model training.",
                            "Requested 0.1 m sampling is distinct from source native GSD.",
                            "Building currency and retrieval dates are not imagery acquisition dates."],
        }
        _json(staging / "candidate-report.json", report)
        _publish(staging, output)
        staging = None
        return report
    except DatasetImportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise DatasetImportError(f"NRW candidate fetch failed: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
