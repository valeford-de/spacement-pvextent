# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Freeze separately acquired AI-reviewed negatives before any predictions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image, ImageChops, ImageDraw

from .augmentation import _provenance
from .curation import _bounded_bytes, _load_review, _polygons, MIN_BACKGROUND_PIXELS
from .pv4ger import DatasetImportError, _publish, _write_json
from .snapshots import load_centers
from .validation import _invalid_constant, _unique_object, validate_dataset


class NegativeFreezeError(DatasetImportError):
    """The proposed negative holdout lacks frozen, spatially separate evidence."""


def _json(path: Path) -> tuple[dict, bytes]:
    payload = _bounded_bytes(path, 2 * 1024 * 1024)
    value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise NegativeFreezeError("Expected a bounded JSON object.")
    return value, payload


def freeze_negative_holdout(curated_dir: Path | str, candidate_dir: Path | str,
                            exclude_dirs: list[Path | str], output_dir: Path | str, *,
                            protocol_path: Path | str, acquisition_protocol_path: Path | str) -> dict:
    """Assign eligible new negatives to test; never open old reference pixels.

    A partial background label supports pixel diagnostics, not a claim that a
    whole roof is PV-free. All source collections remain unchanged.
    """
    stage = None
    try:
        from pyproj import Transformer
        from .training import _read_manifest, _raster

        curated, candidates = Path(curated_dir).resolve(), Path(candidate_dir).resolve()
        references = [Path(p).resolve() for p in exclude_dirs]
        output = Path(output_dir).absolute()
        if not references or len(set(references)) != len(references) or os.path.lexists(output):
            raise NegativeFreezeError("Provide distinct reference snapshots and a new output directory.")
        output = output.resolve()
        inputs = [curated, candidates, *references]
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
            raise NegativeFreezeError("Holdout output must be separate from all inputs.")
        if curated == candidates or any(p in references for p in (curated, candidates)):
            raise NegativeFreezeError("Candidates and curation must be distinct from excluded snapshots.")
        protocol, protocol_bytes = _json(Path(protocol_path))
        acquisition, acquisition_bytes = _json(Path(acquisition_protocol_path))
        digest = lambda data: hashlib.sha256(data).hexdigest()
        if (protocol.get("schema_version") != "1.0"
                or protocol.get("purpose") != "experimental-ai-negative-holdout"
                or protocol.get("acquisition_protocol_sha256") != digest(acquisition_bytes)
                or type(protocol.get("minimum_background_fraction")) is not float
                or protocol["minimum_background_fraction"] not in (0.5, 0.05)
                or protocol.get("threshold") != 0.5
                or protocol.get("model_predictions_used_for_labels") is not False
                or protocol.get("frozen_before_predictions") is not True
                or protocol.get("human_review_claimed") is not False
                or not isinstance(protocol.get("dataset_version"), str) or not protocol["dataset_version"].strip()):
            raise NegativeFreezeError("Expected the fixed experimental negative-holdout protocol.")
        minimum_background_fraction = protocol["minimum_background_fraction"]
        checkpoints = protocol.get("checkpoint_sha256")
        if (not isinstance(checkpoints, dict) or set(checkpoints) != {"reference", "candidate"}
                or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                       for v in checkpoints.values())):
            raise NegativeFreezeError("Freeze both checkpoint SHA-256 identities before evaluation.")
        minimum = acquisition.get("min_reference_distance_m")
        if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
                or not math.isfinite(minimum) or minimum < 5000 + 32 * math.sqrt(2)):
            raise NegativeFreezeError("A reference buffer of at least 5 km plus crop diagonal is required.")
        meta, rows, curated_fp = _read_manifest(curated)
        _, original, candidate_fp = _read_manifest(candidates)
        originals = {r["id"]: r for r in original}
        if (not 1 <= len(rows) <= 24 or {r["id"] for r in rows} != set(originals)
                or any(r["split"] != "unassigned" for r in rows + original)):
            raise NegativeFreezeError("Curate all of the at most 24 unassigned acquired candidates.")
        acquisition_report, _ = _json(candidates / "candidate-report.json")
        if (acquisition_report.get("protocol_sha256") != digest(acquisition_bytes)
                or acquisition_report.get("dataset_fingerprint") != candidate_fp
                or acquisition_report.get("ok") is not True):
            raise NegativeFreezeError("Candidate collection differs from the frozen acquisition protocol.")
        curation_report, _ = _json(curated / "curation-report.json")
        if (curation_report.get("dataset_fingerprint") != curated_fp
                or curation_report.get("model_predictions_used") is not False
                or curation_report.get("human_review_claimed") is not False
                or curation_report.get("ai_review_authorized") is not True):
            raise NegativeFreezeError("Expected explicit AI-review provenance without model predictions.")
        reviewers = curation_report.get("reviewers", [])
        if (not isinstance(reviewers, list) or len(reviewers) != 2
                or any(not isinstance(r, dict) or not isinstance(r.get("reviewer"), str)
                       or not r["reviewer"].strip() for r in reviewers)
                or len({r["reviewer"].strip().casefold() for r in reviewers}) != 2
                or any(r.get("reviewer_type") != "ai" or r.get("model_predictions_used") is not False for r in reviewers)):
            raise NegativeFreezeError("Exactly two distinct AI review passes are required.")
        reviews = []
        for index, evidence in enumerate(reviewers):
            relative = f"provenance/review-{'a' if index == 0 else 'b'}.json"
            review, payload = _load_review(curated / relative)
            if (evidence.get("file") != relative or evidence.get("sha256") != digest(payload)
                    or review["reviewer"] != evidence["reviewer"] or set(review["samples"]) != set(originals)):
                raise NegativeFreezeError("Retained review evidence differs from its recorded identities or hashes.")
            if any(review["samples"][r["id"]]["image_sha256"] != r["image_sha256"] for r in rows):
                raise NegativeFreezeError("Review evidence refers to different imagery.")
            reviews.append(review)
        reference_rows, reference_points, reference_fps, coordinate_hashes = [], [], [], []
        project = Transformer.from_crs(4326, 25832, always_xy=True)
        for reference in references:
            _, excluded, fingerprint = _read_manifest(reference)
            centers, coordinate_hash = load_centers(reference, excluded)
            reference_rows.extend(excluded)
            reference_points.extend(project.transform(*point, errcheck=True) for point in centers.values())
            reference_fps.append(fingerprint)
            coordinate_hashes.append(coordinate_hash)
        if sorted(reference_fps) != sorted(protocol.get("reference_dataset_fingerprints", [])):
            raise NegativeFreezeError("Reference snapshots differ from the frozen protocol.")
        acquired_references = acquisition_report.get("references")
        if (not isinstance(acquired_references, list) or any(not isinstance(r, dict) for r in acquired_references)
                or sorted((r.get("dataset_fingerprint", ""), r.get("coordinate_sha256", "")) for r in acquired_references)
                != sorted(zip(reference_fps, coordinate_hashes))):
            raise NegativeFreezeError("Reference coordinates differ from the acquisition's frozen exclusion basis.")
        ids = {r["id"] for r in reference_rows}
        images = {r["image_sha256"] for r in reference_rows}
        buildings = {key for r in reference_rows for key in r["building_ids"]}
        groups = {r["spatial_group"] for r in reference_rows}
        centers, candidate_coordinate_hash = load_centers(candidates, original)
        distances = {}
        for row in rows:
            old = originals[row["id"]]
            identity = ("image_sha256", "width", "height", "gsd_m", "capture_date", "georeference", "spatial_group", "building_ids")
            if any(row[key] != old[key] for key in identity):
                raise NegativeFreezeError("Reviewed imagery identity differs from acquisition.")
            if (row["id"] in ids or row["image_sha256"] in images or row["spatial_group"] in groups
                    or buildings.intersection(row["building_ids"])):
                raise NegativeFreezeError("Holdout identity overlaps previous data; no new pixels were opened.")
            declared = project.transform(*centers[row["id"]], errcheck=True)
            geo = row["georeference"]
            if geo is None or geo["crs"] != "EPSG:25832":
                raise NegativeFreezeError("Expected the acquired EPSG:25832 native crop grid.")
            a, b, c, d, e, f = geo["transform"]
            point = (a * row["width"] / 2 + b * row["height"] / 2 + c,
                     d * row["width"] / 2 + e * row["height"] / 2 + f)
            if math.dist(point, declared) > 0.2:
                raise NegativeFreezeError("Coordinate provenance disagrees with the actual crop centre.")
            distances[row["id"]] = min(math.dist(point, other) for other in reference_points)
            if distances[row["id"]] < minimum:
                raise NegativeFreezeError("Holdout centre violates the fixed reference buffer.")
        validation = validate_dataset(curated)  # Only new, entirely unassigned curation.
        if not validation.ok or validation.stats["dataset_fingerprint"] != curated_fp:
            raise NegativeFreezeError("Curated pixels or metadata are invalid or changed.")
        frozen_rows, decisions = [], []
        for row in rows:
            if row["width"] != 320 or row["height"] != 320 or row["gsd_m"] != 0.1:
                raise NegativeFreezeError("Holdout requires native 320x320 crops at 0.1 m/pixel.")
            mask = _raster(curated, row, "mask")
            histogram = mask.histogram()
            if sum(histogram[1:255]):
                raise NegativeFreezeError("Negative masks may contain only background 0 and ignore 255.")
            planes = []
            approved = all(r["samples"][row["id"]]["decision"] == "approve-negative" for r in reviews)
            for review in reviews:
                plane = Image.new("L", (320, 320), 0)
                for polygon in _polygons(review["samples"][row["id"]]["background_polygons"], 320, 320):
                    ImageDraw.Draw(plane).polygon(polygon, fill=255)
                planes.append(plane)
            consensus = ImageChops.darker(*planes)
            accepted = approved and consensus.histogram()[255] >= MIN_BACKGROUND_PIXELS
            expected = Image.new("L", (320, 320), 255)
            if accepted:
                expected.paste(0, mask=consensus)
            if (mask.tobytes() != expected.tobytes()
                    or row["annotation_status"] != ("reviewed" if accepted else "unreviewed")):
                raise NegativeFreezeError("Curated mask differs from the retained two-review consensus.")
            eligible = row["annotation_status"] == "reviewed" and histogram[0] >= 320 * 320 * minimum_background_fraction
            frozen_rows.append({**row, "split": "test" if eligible else "unassigned"})
            decisions.append({"id": row["id"], "eligible": eligible, "background_pixels": histogram[0],
                              "ignored_pixels": histogram[255], "nearest_reference_center_m": distances[row["id"]]})
        eligible_ids = sorted(r["id"] for r in frozen_rows if r["split"] == "test")
        if not eligible_ids:
            raise NegativeFreezeError("No masks pass the two-review and frozen background-coverage gate.")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-freeze-", dir=output.parent))
        lines = ['schema_version = "1.0"', 'dataset_id = "pvextent-negative-holdout"',
                 'dataset_version = ' + json.dumps(protocol["dataset_version"])]
        for key, fields in sorted(meta["sources"].items()):
            lines.append('\n[sources.' + json.dumps(key) + ']')
            lines.extend(name + ' = ' + json.dumps(value) for name, value in fields.items())
        (stage / "dataset.toml").write_text('\n'.join(lines) + '\n', encoding="utf-8")
        for row in frozen_rows:
            for role in ("image", "mask"):
                target = stage / row[role]
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = _bounded_bytes(curated / row[role], 16 * 1024 * 1024)
                if digest(payload) != row[role + "_sha256"]:
                    raise NegativeFreezeError("Curated pixels changed during holdout freezing.")
                with target.open("xb") as stream:
                    stream.write(payload)
        (stage / "manifest.jsonl").write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in frozen_rows), encoding="utf-8")
        provenance = stage / "provenance"
        _provenance(curated, provenance / "curation")
        _provenance(candidates, provenance / "acquisition")
        for source, folder, report_name in ((curated, "curation", "curation-report.json"),
                                             (candidates, "acquisition", "candidate-report.json")):
            for name in ("dataset.toml", "manifest.jsonl", report_name):
                payload = _bounded_bytes(source / name, 2 * 1024 * 1024)
                with (provenance / folder / name).open("xb") as stream:
                    stream.write(payload)
        (provenance / "protocol.json").write_bytes(protocol_bytes)
        (provenance / "acquisition-protocol.json").write_bytes(acquisition_bytes)
        shutil.copyfile(candidates / "provenance/samples.jsonl", provenance / "samples.jsonl")
        final = validate_dataset(stage)  # The explicitly new negative holdout only.
        if not final.ok:
            raise NegativeFreezeError("Frozen holdout failed canonical validation.")
        if _read_manifest(curated)[2] != curated_fp or _read_manifest(candidates)[2] != candidate_fp:
            raise NegativeFreezeError("Source declarations changed during holdout freezing.")
        if any(_read_manifest(ref)[2] != fp for ref, fp in zip(references, reference_fps)):
            raise NegativeFreezeError("Reference declarations changed during holdout freezing.")
        if (any(load_centers(ref, _read_manifest(ref)[1])[1] != h for ref, h in zip(references, coordinate_hashes))
                or load_centers(candidates, original)[1] != candidate_coordinate_hash):
            raise NegativeFreezeError("Coordinate provenance changed during holdout freezing.")
        report = {"ok": True, "purpose": "experimental-ai-negative-holdout", "dataset_fingerprint": final.stats["dataset_fingerprint"],
                  "protocol_sha256": digest(protocol_bytes), "acquisition_protocol_sha256": digest(acquisition_bytes),
                  "curated_dataset_fingerprint": curated_fp, "candidate_dataset_fingerprint": candidate_fp,
                  "reference_dataset_fingerprints": reference_fps, "reference_records": len(reference_rows),
                  "records": len(rows), "evaluation_samples": len(eligible_ids), "reviewed_evaluation_ids": eligible_ids,
                  "frozen_before_predictions": True, "model_predictions_used_for_labels": False, "human_review_claimed": False,
                  "minimum_background_fraction": minimum_background_fraction, "minimum_reference_distance_required_m": minimum,
                  "minimum_reference_distance_observed_m": min(distances.values()), "checkpoint_sha256": checkpoints,
                  "old_reference_pixels_opened": False, "training_use_allowed": False,
                  "decisions": decisions, "limitations": ["AI-reviewed partial background labels, not human-confirmed benchmark truth.",
                  "Only supervised background pixels may be scored; no roof-level false-positive rate or prevalence estimate."]}
        _write_json(provenance / "negative-holdout-freeze.json", report)
        _write_json(stage / "validation-report.json", final.to_dict())
        _publish(stage, output)
        stage = None
        return report
    except NegativeFreezeError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, OverflowError) as error:
        raise NegativeFreezeError(f"Negative holdout freeze failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
