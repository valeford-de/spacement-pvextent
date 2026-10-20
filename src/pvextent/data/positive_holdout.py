# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Freeze conservative positive consensus without opening previous holdout pixels."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image, ImageChops, ImageDraw

from .curation import _bounded_bytes, _image_payload, _polygons, REVIEW_FIELDS
from .pv4ger import DatasetImportError, _publish, _write_json
from .snapshots import load_centers
from .validation import IDENTIFIER, SHA256, _invalid_constant, _unique_object, validate_dataset

PURPOSE = "experimental-ai-positive-holdout"
MINIMUM_POSITIVE_PIXELS = 100
SAMPLE_FIELDS = {"id", "image_sha256", "full_image_inspected", "decision",
                 "positive_polygons", "background_polygons", "notes"}
IDENTITY = ("image_sha256", "width", "height", "gsd_m", "capture_date", "georeference", "spatial_group", "building_ids")


class PositiveHoldoutError(DatasetImportError):
    """The positive holdout lacks frozen, independent consensus evidence."""


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _object(payload):
    value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise PositiveHoldoutError("Expected a bounded JSON object.")
    return value


def _review(payload, originals):
    review = _object(payload)
    if (set(review) != REVIEW_FIELDS or review["review_schema_version"] != "1.0"
            or review["reviewer_type"] != "ai" or review["model_predictions_used"] is not False
            or not isinstance(review["reviewer"], str) or not review["reviewer"].strip()
            or len(review["reviewer"]) > 256 or not isinstance(review["samples"], list)):
        raise PositiveHoldoutError("Expected strict schema 1.0 and an identified AI reviewer without predictions.")
    samples, vertices = {}, 0
    if len(review["samples"]) != len(originals):
        raise PositiveHoldoutError("Each review must cover exactly the selected candidate IDs.")
    for sample in review["samples"]:
        if not isinstance(sample, dict) or set(sample) != SAMPLE_FIELDS:
            raise PositiveHoldoutError("Expected exactly the positive sample review fields.")
        identifier = sample["id"]
        if (not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier) or identifier in samples
                or identifier not in originals or sample["image_sha256"] != originals[identifier]["image_sha256"]
                or sample["full_image_inspected"] is not True or sample["decision"] not in ("approve-positive", "defer")
                or not isinstance(sample["notes"], str) or len(sample["notes"]) > 8192):
            raise PositiveHoldoutError("Invalid review ID, imagery hash, full-image inspection or decision.")
        planes = []
        for field in ("positive_polygons", "background_polygons"):
            polygons = _polygons(sample[field], 320, 320)
            vertices += sum(len(polygon) for polygon in polygons)
            if vertices > 8192:
                raise PositiveHoldoutError("A review may contain at most 8192 vertices.")
            plane = Image.new("L", (320, 320), 0)
            for polygon in polygons:
                ImageDraw.Draw(plane).polygon(polygon, fill=255)
            planes.append(plane)
        if ImageChops.darker(*planes).getbbox() is not None:
            raise PositiveHoldoutError("Positive and background polygons overlap within a review.")
        samples[identifier] = {**sample, "planes": planes}
    if set(samples) != set(originals):
        raise PositiveHoldoutError("Each review must cover exactly the selected candidate IDs.")
    return {**review, "samples": samples}


def freeze_positive_holdout(candidate_dir: Path | str, negative_holdout_dir: Path | str,
                            exclude_dirs: list[Path | str], output_dir: Path | str, *,
                            review_paths: list[Path | str], protocol_path: Path | str) -> dict:
    """Label only the untouched complement of a frozen negative pilot.

    Both reviewers must agree on every supervised pixel. Uncertain records stay
    unassigned; source images and previous test masks are never modified.
    """
    stage = None
    try:
        from pyproj import Transformer
        from .training import _read_manifest

        candidates, negative = Path(candidate_dir).resolve(), Path(negative_holdout_dir).resolve()
        references = [Path(path).resolve() for path in exclude_dirs]
        output = Path(output_dir).absolute()
        inputs = [candidates, negative, *references]
        if (not references or len(set(inputs)) != len(inputs) or os.path.lexists(output)
                or not isinstance(review_paths, (list, tuple)) or len(review_paths) != 2):
            raise PositiveHoldoutError("Provide distinct input snapshots, two reviews and a new output directory.")
        output = output.resolve()
        if any(output.is_relative_to(path) or path.is_relative_to(output) for path in inputs):
            raise PositiveHoldoutError("Output must be separate from every input snapshot.")
        watched = {}
        def read(path, maximum=2 * 1024 * 1024):
            path = Path(path)
            payload = _bounded_bytes(path, maximum)
            digest = _digest(payload)
            if path in watched and watched[path][1] != digest:
                raise PositiveHoldoutError("Read-only evidence changed during holdout freezing.")
            watched[path] = maximum, digest
            return payload
        protocol_bytes = read(protocol_path)
        protocol = _object(protocol_bytes)
        if (protocol.get("schema_version") != "1.0" or protocol.get("purpose") != PURPOSE
                or protocol.get("threshold") != .5 or type(protocol.get("minimum_positive_pixels")) is not int
                or protocol["minimum_positive_pixels"] != MINIMUM_POSITIVE_PIXELS
                or protocol.get("frozen_before_predictions") is not True
                or any(protocol.get(key) is not False for key in
                       ("model_predictions_used_for_labels", "human_review_claimed", "training_use_allowed"))
                or not isinstance(protocol.get("dataset_version"), str)
                or not 1 <= len(protocol["dataset_version"].strip()) <= 256):
            raise PositiveHoldoutError("Expected the frozen experimental positive-holdout protocol.")
        checkpoints = protocol.get("checkpoint_sha256")
        if (not isinstance(checkpoints, dict) or set(checkpoints) != {"reference", "candidate"}
                or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in checkpoints.values())):
            raise PositiveHoldoutError("Freeze both checkpoint SHA-256 identities before evaluation.")
        meta, rows, candidate_fp = _read_manifest(candidates)
        negative_meta, negative_rows, negative_fp = _read_manifest(negative)
        originals, prior = {row["id"]: row for row in rows}, {row["id"]: row for row in negative_rows}
        negative_report = _object(read(negative / "provenance/negative-holdout-freeze.json"))
        if (not 1 <= len(rows) <= 24 or set(originals) != set(prior)
                or any(row["split"] != "unassigned" for row in rows)
                or any(row["split"] not in ("test", "unassigned") for row in negative_rows)
                or any(any(row[key] != originals[row["id"]][key] for key in IDENTITY) for row in negative_rows)
                or negative_meta["dataset_id"] != "pvextent-negative-holdout"
                or negative_report.get("ok") is not True
                or negative_report.get("purpose") != "experimental-ai-negative-holdout"
                or negative_report.get("dataset_fingerprint") != negative_fp
                or negative_report.get("candidate_dataset_fingerprint") != candidate_fp
                or negative_report.get("checkpoint_sha256") != checkpoints
                or negative_report.get("model_predictions_used_for_labels") is not False
                or negative_report.get("training_use_allowed") is not False
                or sorted(negative_report.get("reviewed_evaluation_ids", [])) != sorted(r["id"] for r in negative_rows if r["split"] == "test")):
            raise PositiveHoldoutError("Negative holdout identity or frozen acquisition lineage differs.")
        selected = {key: originals[key] for key in originals if prior[key]["split"] == "unassigned"}
        selected_ids = sorted(selected)
        if (len(selected) != 13 or protocol.get("candidate_dataset_fingerprint") != candidate_fp
                or protocol.get("negative_holdout_dataset_fingerprint") != negative_fp
                or not isinstance(protocol.get("selected_ids"), list)
                or protocol["selected_ids"] != selected_ids
                or protocol.get("selected_image_sha256") != {key: row["image_sha256"] for key, row in selected.items()}):
            raise PositiveHoldoutError("Protocol must select exactly the untouched negative-holdout complement.")
        acquisition_bytes = read(candidates / "provenance/protocol.json")
        acquisition = _object(acquisition_bytes)
        acquisition_report = _object(read(candidates / "candidate-report.json"))
        minimum = acquisition.get("min_reference_distance_m")
        if (acquisition_report.get("ok") is not True or acquisition_report.get("dataset_fingerprint") != candidate_fp
                or acquisition_report.get("protocol_sha256") != _digest(acquisition_bytes)
                or isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                or not math.isfinite(minimum) or minimum < 5000 + 32 * math.sqrt(2)
                or ("acquisition_protocol_sha256" in protocol and protocol["acquisition_protocol_sha256"] != _digest(acquisition_bytes))):
            raise PositiveHoldoutError("Acquisition evidence or fixed reference buffer differs.")
        reviews, review_evidence = [], []
        for index, path in enumerate(review_paths):
            payload = read(path)
            review = _review(payload, selected)
            reviews.append(review)
            review_evidence.append({"reviewer": review["reviewer"], "reviewer_type": "ai", "model_predictions_used": False,
                                    "file": f"provenance/review-{'a' if index == 0 else 'b'}.json", "sha256": _digest(payload)})
        if len({review["reviewer"].strip().casefold() for review in reviews}) != 2:
            raise PositiveHoldoutError("Exactly two distinct AI reviewers are required.")
        reference_rows, reference_points, reference_fps, coordinate_hashes = [], [], [], []
        project = Transformer.from_crs(4326, 25832, always_xy=True)
        for reference in references:
            _, excluded, fingerprint = _read_manifest(reference)
            centers, coordinate_hash = load_centers(reference, excluded)
            reference_rows.extend(excluded)
            reference_points.extend(project.transform(*point, errcheck=True) for point in centers.values())
            reference_fps.append(fingerprint)
            coordinate_hashes.append(coordinate_hash)
        acquired = acquisition_report.get("references")
        if (sorted(protocol.get("reference_dataset_fingerprints", [])) != sorted(reference_fps)
                or not isinstance(acquired, list) or any(not isinstance(item, dict) for item in acquired)
                or sorted((item.get("dataset_fingerprint", ""), item.get("coordinate_sha256", "")) for item in acquired)
                != sorted(zip(reference_fps, coordinate_hashes))):
            raise PositiveHoldoutError("Reference fingerprints or coordinates differ from the acquisition exclusion basis.")
        ids, hashes = {r["id"] for r in reference_rows}, {r["image_sha256"] for r in reference_rows}
        groups = {r["spatial_group"] for r in reference_rows}
        buildings = {key for r in reference_rows for key in r["building_ids"]}
        centers, candidate_coordinates = load_centers(candidates, rows)
        distances = {}
        for identifier, row in selected.items():
            if identifier in ids or row["image_sha256"] in hashes or row["spatial_group"] in groups or buildings.intersection(row["building_ids"]):
                raise PositiveHoldoutError("Holdout identity overlaps previous data; no source pixels were opened.")
            geo = row["georeference"]
            if row["width"] != 320 or row["height"] != 320 or row["gsd_m"] != .1 or geo is None or geo["crs"] != "EPSG:25832":
                raise PositiveHoldoutError("Expected native EPSG:25832 320x320 crops at 0.1 m/pixel.")
            a, b, c, d, e, f = geo["transform"]
            if any(abs(value - expected) > 1e-9 for value, expected in zip((a, b, d, e), (.1, 0, 0, -.1))):
                raise PositiveHoldoutError("Expected the native 0.1 m affine crop grid.")
            point = (a * 160 + b * 160 + c, d * 160 + e * 160 + f)
            if math.dist(point, project.transform(*centers[identifier], errcheck=True)) > .2:
                raise PositiveHoldoutError("Coordinate provenance disagrees with the affine crop centre.")
            distances[identifier] = min(math.dist(point, other) for other in reference_points)
            if distances[identifier] < minimum:
                raise PositiveHoldoutError("Holdout centre violates the fixed reference buffer.")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-positive-", dir=output.parent))
        lines = ['schema_version = "1.0"', 'dataset_id = "pvextent-positive-holdout"', 'dataset_version = ' + json.dumps(protocol["dataset_version"])]
        for key, fields in sorted(meta["sources"].items()):
            fields = {**fields, "annotation_license": "Apache-2.0", "version": fields["version"] + "; AI-positive annotation " + protocol["dataset_version"],
                      "attribution": fields["attribution"] + " New partial masks: copyright 2026 Valeford, Apache-2.0. Two independent AI reviews; no model predictions or human review claimed."}
            lines.append('\n[sources.' + json.dumps(key) + ']')
            lines.extend(name + ' = ' + json.dumps(value) for name, value in fields.items())
        (stage / "dataset.toml").write_text('\n'.join(lines) + '\n', encoding="utf-8")
        frozen_rows, decisions = [], []
        for identifier in selected_ids:
            row = selected[identifier]
            samples = [review["samples"][identifier] for review in reviews]
            positive, background = [ImageChops.darker(*(sample["planes"][index] for sample in samples)) for index in (0, 1)]
            eligible = all(sample["decision"] == "approve-positive" for sample in samples) and positive.histogram()[255] >= MINIMUM_POSITIVE_PIXELS
            mask = Image.new("L", (320, 320), 255)
            if eligible:
                mask.paste(1, mask=positive)
                mask.paste(0, mask=background)
            image_bytes = _image_payload(candidates, row)  # Only newly selected original images, never source masks.
            image_path, mask_path = f"images/{identifier}.png", f"masks/{identifier}.png"
            for relative in (image_path, mask_path):
                (stage / relative).parent.mkdir(exist_ok=True)
            (stage / image_path).write_bytes(image_bytes)
            if read(candidates / row["image"], 16 * 1024 * 1024) != image_bytes:
                raise PositiveHoldoutError("Selected imagery changed during holdout freezing.")
            buffer = io.BytesIO()
            mask.save(buffer, format="PNG")
            mask_bytes = buffer.getvalue()
            (stage / mask_path).write_bytes(mask_bytes)
            frozen_rows.append({**row, "image": image_path, "mask": mask_path, "mask_sha256": _digest(mask_bytes),
                                "split": "test" if eligible else "unassigned", "annotation_status": "reviewed" if eligible else "unreviewed"})
            histogram = mask.histogram()
            decisions.append({"id": identifier, "eligible": eligible, "positive_pixels": histogram[1], "background_pixels": histogram[0],
                              "ignored_pixels": histogram[255], "nearest_reference_center_m": distances[identifier]})
        eligible_ids = [row["id"] for row in frozen_rows if row["split"] == "test"]
        if not eligible_ids:
            raise PositiveHoldoutError("No masks pass two-review approval and the 100-positive-pixel consensus gate.")
        (stage / "manifest.jsonl").write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in frozen_rows), encoding="utf-8")
        provenance = stage / "provenance"
        provenance.mkdir()
        for source, folder, report_file in ((candidates, "acquisition", "candidate-report.json"),
                                             (negative, "negative-parent", "provenance/negative-holdout-freeze.json")):
            destination = provenance / folder
            destination.mkdir()
            for name in ("dataset.toml", "manifest.jsonl", report_file):
                (destination / Path(name).name).write_bytes(read(source / name))
        # Retain bounded acquisition evidence and notices, never provenance rasters.
        total = 0
        for path in sorted((candidates / "provenance").rglob("*")):
            if path.is_symlink():
                raise PositiveHoldoutError("Acquisition provenance links are not copied.")
            if path.is_dir():
                continue
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
                raise PositiveHoldoutError("Acquisition provenance must not contain rasters.")
            payload = read(path, 16 * 1024 * 1024)
            total += len(payload)
            if total > 64 * 1024 * 1024:
                raise PositiveHoldoutError("Acquisition provenance bounds exceeded.")
            target = provenance / "acquisition" / path.relative_to(candidates / "provenance")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        for name in ("LICENSE", "NOTICE"):
            if os.path.lexists(candidates / name):
                (provenance / "acquisition" / ("source-root-" + name)).write_bytes(read(candidates / name))
        (provenance / "protocol.json").write_bytes(protocol_bytes)
        (provenance / "acquisition-protocol.json").write_bytes(acquisition_bytes)
        for index, path in enumerate(review_paths):
            (stage / review_evidence[index]["file"]).write_bytes(read(path))
        coordinate_lines = read(candidates / "provenance/samples.jsonl", 16 * 1024 * 1024).decode("utf-8").splitlines()
        (provenance / "samples.jsonl").write_text(''.join(line + '\n' for line in coordinate_lines if _object(line)["id"] in selected), encoding="utf-8")
        final = validate_dataset(stage)  # Only the newly generated 13-image collection.
        if not final.ok:
            raise PositiveHoldoutError("Frozen positive holdout failed canonical validation.")
        for source, fingerprint in ((candidates, candidate_fp), (negative, negative_fp), *zip(references, reference_fps)):
            if _read_manifest(source)[2] != fingerprint:
                raise PositiveHoldoutError("Source declarations changed during holdout freezing.")
        if (load_centers(candidates, rows)[1] != candidate_coordinates
                or any(load_centers(ref, _read_manifest(ref)[1])[1] != value for ref, value in zip(references, coordinate_hashes))):
            raise PositiveHoldoutError("Coordinate provenance changed during holdout freezing.")
        for path, (maximum, digest) in watched.items():
            if _digest(_bounded_bytes(path, maximum)) != digest:
                raise PositiveHoldoutError("Read-only evidence changed during holdout freezing.")
        report = {"ok": True, "purpose": PURPOSE, "dataset_fingerprint": final.stats["dataset_fingerprint"], "protocol_sha256": _digest(protocol_bytes),
                  "acquisition_protocol_sha256": _digest(acquisition_bytes), "candidate_dataset_fingerprint": candidate_fp,
                  "negative_holdout_dataset_fingerprint": negative_fp, "reference_dataset_fingerprints": reference_fps,
                  "reference_coordinate_sha256": coordinate_hashes, "reference_records": len(reference_rows),
                  "selected_ids": selected_ids, "reviewed_evaluation_ids": eligible_ids, "records": len(frozen_rows),
                  "evaluation_samples": len(eligible_ids), "checkpoint_sha256": checkpoints, "minimum_positive_pixels": MINIMUM_POSITIVE_PIXELS,
                  "frozen_before_predictions": True, "model_predictions_used_for_labels": False, "human_review_claimed": False,
                  "training_use_allowed": False, "old_reference_pixels_opened": False, "previous_negative_test_pixels_opened": False,
                  "minimum_reference_distance_required_m": minimum, "minimum_reference_distance_observed_m": min(distances.values()),
                  "reviewers": review_evidence, "decisions": decisions,
                  "limitations": ["AI-reviewed partial consensus labels; no human-confirmed benchmark truth or prevalence estimate."]}
        _write_json(provenance / "positive-holdout-freeze.json", report)
        _write_json(stage / "validation-report.json", final.to_dict())
        _publish(stage, output)
        stage = None
        return report
    except PositiveHoldoutError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, OverflowError) as error:
        raise PositiveHoldoutError(f"Positive holdout freeze failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
