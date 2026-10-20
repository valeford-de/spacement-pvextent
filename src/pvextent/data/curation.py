# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Explicitly authorized two-pass AI curation of conservative background masks.

Review decisions never alter a source snapshot or assign a training split.
Only the intersection of both reviewers' native-pixel background polygons can
become supervised background. Everything else stays ignored, with no PV label.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from .pv4ger import DatasetImportError, _publish, _write_json
from .review import SOURCE_NOTICES
from .validation import (
    IDENTIFIER, SHA256, MAX_FILE_BYTES, MAX_METADATA_BYTES, MAX_MANIFEST_BYTES,
    ValidationReport, _invalid_constant, _path, _raster, _read, _unique_object,
    validate_dataset,
)

MAX_REVIEW_BYTES = 2 * 1024 * 1024
MAX_SAMPLES = 64
MAX_POLYGONS = 64
MAX_VERTICES = 128
MAX_REVIEW_VERTICES = 8192
MIN_BACKGROUND_PIXELS = 256
REVIEW_FIELDS = frozenset((
    "review_schema_version", "reviewer", "reviewer_type", "model_predictions_used", "samples",
))
SAMPLE_FIELDS = frozenset((
    "id", "image_sha256", "full_image_inspected", "decision", "background_polygons", "notes",
))


class NegativeCurationError(DatasetImportError):
    """A negative collection cannot be published with the required evidence."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bounded_bytes(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise NegativeCurationError(f"Expected a bounded regular file: {path}")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise NegativeCurationError(f"File exceeds its size bound: {path}")
    return payload


def _load_review(path: Path) -> tuple[dict, bytes]:
    payload = _bounded_bytes(path, MAX_REVIEW_BYTES)
    review = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                        parse_constant=_invalid_constant)
    if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
        raise NegativeCurationError("Expected exactly the review schema 1.0 header fields.")
    reviewer = review["reviewer"]
    if (review["review_schema_version"] != "1.0" or review["reviewer_type"] != "ai"
            or not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 256
            or review["model_predictions_used"] is not False):
        raise NegativeCurationError("Reviews require an identified AI reviewer and model_predictions_used=false.")
    samples = review["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise NegativeCurationError("Each review must select between 1 and 64 samples.")
    by_id = {}
    for row in samples:
        if not isinstance(row, dict) or set(row) != SAMPLE_FIELDS:
            raise NegativeCurationError("Expected exactly the sample review fields.")
        identifier, digest = row["id"], row["image_sha256"]
        if (not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier) or identifier in by_id
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)
                or row["full_image_inspected"] is not True
                or row["decision"] not in ("approve-negative", "defer")
                or not isinstance(row["notes"], str) or len(row["notes"]) > 8192):
            raise NegativeCurationError("Invalid/duplicate sample review, hash, inspection flag or decision.")
        by_id[identifier] = row
    return {**review, "samples": by_id}, payload


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    return (_cross(a, b, c) == 0 and min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def _intersects(a, b, c, d) -> bool:
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ((ab_c * ab_d < 0 and cd_a * cd_b < 0)
            or _on_segment(a, b, c) or _on_segment(a, b, d)
            or _on_segment(c, d, a) or _on_segment(c, d, b))


def _polygons(value: Any, width: int, height: int) -> list[list[tuple[int, int]]]:
    if not isinstance(value, list) or len(value) > MAX_POLYGONS:
        raise NegativeCurationError("Expected a bounded list of background polygons.")
    polygons = []
    for polygon in value:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= MAX_VERTICES + 1:
            raise NegativeCurationError("A background polygon needs 3..128 native-pixel vertices.")
        points = []
        for point in polygon:
            if (not isinstance(point, list) or len(point) != 2 or any(type(n) is not int for n in point)
                    or not 0 <= point[0] < width or not 0 <= point[1] < height):
                raise NegativeCurationError("Polygon coordinates must be integer pixel positions inside the native grid.")
            points.append(tuple(point))
        if points[-1] == points[0]:
            points.pop()
        if not 3 <= len(points) <= MAX_VERTICES or len(set(points)) != len(points):
            raise NegativeCurationError("Polygons require distinct vertices; only the closing endpoint may repeat.")
        area2 = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1]))
        if area2 == 0:
            raise NegativeCurationError("Degenerate background polygon.")
        edges = list(zip(points, points[1:] + points[:1]))
        for index, (a, b) in enumerate(edges):
            c = points[(index + 2) % len(points)]
            if _cross(a, b, c) == 0 and (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) < 0:
                raise NegativeCurationError("Polygon edges must not backtrack.")
            for other in range(index + 1, len(edges)):
                if other == index + 1 or (index == 0 and other == len(edges) - 1):
                    continue
                if _intersects(a, b, *edges[other]):
                    raise NegativeCurationError("Self-intersecting background polygons are not supported.")
        polygons.append(points)
    return polygons


def _metadata_bytes(source: dict, source_id: str, version: str) -> bytes:
    source["version"] = source["version"] + "; AI-negative annotation " + version
    source["annotation_license"] = "Apache-2.0"
    source["attribution"] += (
        " New background/ignore masks: copyright 2026 Valeford, Apache-2.0. "
        "Two independent AI visual review passes, explicitly authorized for experimental use; "
        "no model predictions used and no human review claimed. Annotation version: " + version + "."
    )
    return (f"\n[sources.{source_id}]\n" + "".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)}\n" for key, value in source.items()
    )).encode("utf-8")


def _image_payload(source: Path, record: dict) -> bytes:
    report = ValidationReport()
    _raster(source, record, "image", report, record["id"])
    if not report.ok:
        raise NegativeCurationError(f"Selected image is invalid: {report.errors[0].message}")
    path = _path(source, record["image"], report, record["id"])
    if path is None:
        raise NegativeCurationError("Selected image cannot be resolved inside its source dataset.")
    payload = _read(path, MAX_FILE_BYTES, report, record["id"])
    if payload is None or _digest(payload) != record["image_sha256"]:
        raise NegativeCurationError("Selected source image bytes differ from the reviewed SHA-256.")
    return payload


def _copy_source_provenance(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True)
    hashes = {}
    provenance = source / "provenance"
    if provenance.is_symlink():
        raise NegativeCurationError("Source provenance must not be a symbolic link.")
    notices = [(provenance / name, name) for name in SOURCE_NOTICES]
    notices += [(source / name, "source-root-" + name) for name in ("LICENSE", "NOTICE")]
    for path, target in notices:
        if os.path.lexists(path):
            payload = _bounded_bytes(path, 1024 * 1024)
            (destination / target).write_bytes(payload)
            hashes[target] = _digest(payload)
    return hashes


def curate_negative_collection(
    source_dirs: list[Path | str], review_paths: list[Path | str], output_dir: Path | str, *,
    dataset_version: str, allow_ai_review: bool = False,
) -> dict[str, Any]:
    """Publish an unassigned 0/255 collection from two authorized AI reviews.

    ``reviewed`` records explicitly retain AI origin and never imply human
    adjudication. Geographic assignment and any training merge are separate.
    """
    if allow_ai_review is not True:
        raise NegativeCurationError("AI curation requires explicit allow_ai_review=True authorization.")
    if (not isinstance(source_dirs, (list, tuple)) or not source_dirs
            or not isinstance(review_paths, (list, tuple)) or len(review_paths) != 2):
        raise NegativeCurationError("Provide source datasets and exactly two independent AI review files.")
    if not isinstance(dataset_version, str) or not dataset_version.strip() or len(dataset_version) > 256:
        raise NegativeCurationError("A nonempty dataset_version of at most 256 characters is required.")
    stage = None
    try:
        from .training import _read_manifest

        sources = [Path(path).resolve(strict=True) for path in source_dirs]
        if len(set(sources)) != len(sources):
            raise NegativeCurationError("Source dataset directories must be distinct.")
        reviews_at = [Path(path).absolute() for path in review_paths]
        output = Path(output_dir).absolute()
        if os.path.lexists(output):
            raise NegativeCurationError(f"Output already exists: {output}")
        output = output.resolve()
        if (any(output.is_relative_to(source) or source.is_relative_to(output) for source in sources)
                or any(path.resolve().is_relative_to(output) for path in reviews_at)):
            raise NegativeCurationError("Output must be separate from source datasets and cannot contain its review files.")
        reviews, review_payloads = [], []
        for path in reviews_at:
            review, payload = _load_review(path)
            reviews.append(review)
            review_payloads.append(payload)
        if reviews[0]["reviewer"].strip().casefold() == reviews[1]["reviewer"].strip().casefold():
            raise NegativeCurationError("Exactly two distinct AI reviewer identities are required.")
        selected = set(reviews[0]["samples"])
        if selected != set(reviews[1]["samples"]):
            raise NegativeCurationError("Both independent reviews must cover exactly the same sample IDs.")

        locations, source_info = {}, []
        for source in sources:
            metadata, records, fingerprint = _read_manifest(source)
            metadata_payload = _bounded_bytes(source / "dataset.toml", MAX_METADATA_BYTES)
            manifest_payload = _bounded_bytes(source / "manifest.jsonl", MAX_MANIFEST_BYTES)
            if _digest(metadata_payload + b"\0" + manifest_payload) != fingerprint:
                raise NegativeCurationError("Source metadata changed while reading the collection.")
            info = {"path": source, "metadata": metadata, "fingerprint": fingerprint,
                    "metadata_bytes": metadata_payload, "manifest_sha256": _digest(manifest_payload)}
            source_info.append(info)
            for record in records:
                if record["id"] in locations:
                    raise NegativeCurationError("Source sample IDs must be unique across all input datasets.")
                locations[record["id"]] = (info, record)
        if not selected <= locations.keys():
            raise NegativeCurationError("A reviewed sample ID is absent from the source datasets.")

        normalized = [{}, {}]
        for index, review in enumerate(reviews):
            total_vertices = 0
            for identifier in sorted(selected):
                _, record = locations[identifier]
                row = review["samples"][identifier]
                if record["split"] != "unassigned":
                    raise NegativeCurationError("Only unassigned source samples may be curated; frozen train/val/test are protected.")
                if row["image_sha256"] != record["image_sha256"]:
                    raise NegativeCurationError("Review image SHA-256 differs from the source manifest.")
                polygons = _polygons(row["background_polygons"], record["width"], record["height"])
                total_vertices += sum(map(len, polygons))
                if total_vertices > MAX_REVIEW_VERTICES:
                    raise NegativeCurationError("Review polygon vertices exceed the configured bound.")
                normalized[index][identifier] = polygons

        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-curation-", dir=output.parent))
        for folder in ("images", "masks", "provenance"):
            (stage / folder).mkdir()
        provenance = stage / "provenance"
        reviewer_report = []
        for index, (review, payload) in enumerate(zip(reviews, review_payloads)):
            name = f"review-{'a' if index == 0 else 'b'}.json"
            (provenance / name).write_bytes(payload)
            reviewer_report.append({"reviewer": review["reviewer"], "reviewer_type": "ai",
                                    "model_predictions_used": False, "file": f"provenance/{name}",
                                    "sha256": _digest(payload)})
        copied_sources, coordinate_rows = [], {}
        for info in source_info:
            original = info["metadata"]
            key = info["fingerprint"][:24]
            destination = provenance / "sources" / key
            hashes = _copy_source_provenance(info["path"], destination)
            (destination / "dataset.toml").write_bytes(info["metadata_bytes"])
            copied_sources.append({"dataset_id": original["dataset_id"], "dataset_version": original["dataset_version"],
                                   "dataset_fingerprint": info["fingerprint"], "manifest_sha256": info["manifest_sha256"],
                                   "metadata_file": f"provenance/sources/{key}/dataset.toml",
                                   "metadata_sha256": _digest(info["metadata_bytes"]), "notice_hashes": hashes})
            coordinates = info["path"] / "provenance" / "samples.jsonl"
            if os.path.lexists(coordinates):
                payload = _bounded_bytes(coordinates, 16 * 1024 * 1024)
                seen = set()
                for line in payload.decode("utf-8").splitlines():
                    row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in seen:
                        raise NegativeCurationError("Source coordinate provenance has invalid or duplicate IDs.")
                    seen.add(row["id"])
                    if row["id"] in selected and locations[row["id"]][0] is info:
                        coordinate_rows[row["id"]] = row
        if coordinate_rows:
            (provenance / "samples.jsonl").write_text(
                "".join(json.dumps(coordinate_rows[key], sort_keys=True, allow_nan=False) + "\n" for key in sorted(coordinate_rows)),
                encoding="utf-8",
            )

        metadata_bytes = ("schema_version = \"1.0\"\ndataset_id = \"pvextent-negative-curation\"\n"
                          f"dataset_version = {json.dumps(dataset_version, ensure_ascii=False)}\n").encode("utf-8")
        declared_sources, rows, original_rows, decisions = set(), [], [], []
        background_pixels, ignored_pixels = 0, 0
        for identifier in sorted(selected):
            info, original = locations[identifier]
            source = info["metadata"]["sources"][original["source_id"]]
            source_key = _digest(json.dumps({"source_id": original["source_id"], "source": source,
                                            "annotation_version": dataset_version}, sort_keys=True).encode())[:24]
            source_id = "ai-negative-" + source_key
            if source_id not in declared_sources:
                metadata_bytes += _metadata_bytes(dict(source), source_id, dataset_version)
                declared_sources.add(source_id)
            reviewed = all(review["samples"][identifier]["decision"] == "approve-negative" for review in reviews)
            consensus = Image.new("L", (original["width"], original["height"]), 0)
            if reviewed:
                masks = []
                for review_polygons in normalized:
                    plane = Image.new("L", consensus.size, 0)
                    draw = ImageDraw.Draw(plane)
                    for polygon in review_polygons[identifier]:
                        draw.polygon(polygon, fill=255)
                    masks.append(plane)
                consensus = ImageChops.darker(*masks)
            intersection_pixels = consensus.histogram()[255]
            accepted = reviewed and intersection_pixels >= MIN_BACKGROUND_PIXELS
            mask = Image.new("L", consensus.size, 255)
            if accepted:
                mask.paste(0, mask=consensus)
            count = intersection_pixels if accepted else 0
            background_pixels += count
            ignored_pixels += original["width"] * original["height"] - count
            key = _digest(identifier.encode())[:24]
            suffix = Path(original["image"]).suffix.lower()
            image_name, mask_name = f"images/{key}{suffix}", f"masks/{key}.png"
            payload = _image_payload(info["path"], original)
            (stage / image_name).write_bytes(payload)
            mask.save(stage / mask_name)
            rows.append({**original, "image": image_name, "mask": mask_name,
                         "image_sha256": _digest(payload), "mask_sha256": _digest((stage / mask_name).read_bytes()),
                         "source_id": source_id, "split": "unassigned",
                         "annotation_status": "reviewed" if accepted else "unreviewed"})
            original_rows.append({"id": identifier, "source_dataset_fingerprint": info["fingerprint"],
                                  "source_dataset_id": info["metadata"]["dataset_id"], "record": original})
            decisions.append({"id": identifier, "annotation_status": rows[-1]["annotation_status"],
                              "intersection_pixels": intersection_pixels, "background_pixels": count,
                              "ignored_pixels": original["width"] * original["height"] - count,
                              "reason": "two_ai_reviews" if accepted else "insufficient_consensus" if reviewed else "reviewer_deferred"})
        (stage / "dataset.toml").write_bytes(metadata_bytes)
        for path, content in ((stage / "manifest.jsonl", rows), (provenance / "original-source-records.jsonl", original_rows)):
            path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in content), encoding="utf-8")
        (provenance / "AI-ANNOTATION-PROVENANCE.md").write_text(
            "# Experimental AI-reviewed background masks\n\n"
            "Copyright 2026 Valeford. New masks: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0 .\n"
            "Two distinct AI visual reviewers inspected full native images without model predictions.\n"
            "This collection has explicit AI-review authorization; it does not claim human review.\n"
            "Polygon vertices address pixel centres; polygon boundaries are included. Only the intersection of both\n"
            "approved background unions is zero. Every other pixel is ignore (255); no PV (1) is generated.\n"
            "Both approvals and at least 256 intersection pixels are required for reviewed status.\n"
            "Every sample remains unassigned. Geographic merging and experimental training are separate actions.\n"
            "Original imagery rights and upstream notices remain with their sources.\n", encoding="utf-8",
        )
        validation = validate_dataset(stage)
        if not validation.ok:
            raise NegativeCurationError(f"Curated output failed validation: {validation.errors[0].message}")
        _write_json(stage / "validation-report.json", validation.to_dict())
        report = {"ok": True, "dataset_id": "pvextent-negative-curation", "dataset_version": dataset_version,
                  "records": len(rows), "reviewed_samples": sum(row["annotation_status"] == "reviewed" for row in rows),
                  "unreviewed_samples": sum(row["annotation_status"] == "unreviewed" for row in rows),
                  "reviewed_ids": [row["id"] for row in rows if row["annotation_status"] == "reviewed"],
                  "unreviewed_ids": [row["id"] for row in rows if row["annotation_status"] == "unreviewed"],
                  "background_pixels": background_pixels, "ignored_pixels": ignored_pixels, "pv_pixels": 0,
                  "minimum_consensus_pixels": MIN_BACKGROUND_PIXELS, "ai_review_authorized": True,
                  "human_review_claimed": False, "model_predictions_used": False, "split": "unassigned",
                  "reviewers": reviewer_report, "sources": copied_sources, "decisions": decisions,
                  "dataset_fingerprint": validation.stats["dataset_fingerprint"],
                  "manifest_sha256": validation.stats["manifest_sha256"], "metadata_sha256": validation.stats["metadata_sha256"],
                  "coordinate_records": len(coordinate_rows),
                  "validation": {"ok": True, "error_count": 0, "warning_count": validation.warning_count}}
        _write_json(stage / "curation-report.json", report)
        _publish(stage, output)
        stage = None
        return report
    except NegativeCurationError:
        raise
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError, RecursionError, OverflowError) as error:
        raise NegativeCurationError(f"Negative curation failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
