# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Pair frozen validation measurements without opening imagery or checkpoints."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import median
import tempfile

from pvextent.data.pv4ger import _publish, _write_json

COUNTS = ("tp", "fp", "fn", "tn", "valid_pixels")
RATIOS = ("iou", "dice", "precision", "recall", "pixel_accuracy", "macro_iou", "median_iou")
HASH = re.compile(r"[0-9a-f]{64}\Z")
MAX_SAMPLES = 100000


class ValidationComparisonError(ValueError):
    """Frozen analysis evidence is invalid, changed or not comparable."""


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationComparisonError("Duplicate JSON object keys are not allowed.")
        result[key] = value
    return result


def _invalid(value):
    raise ValidationComparisonError(f"Non-finite JSON value: {value}.")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _invalid(value)
    return number


def _payload(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValidationComparisonError("Expected a bounded, regular local JSON file.")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValidationComparisonError("JSON byte bound exceeded.")
    return payload


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _number(value, low: float, high: float = math.inf) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low < value <= high


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(counts: dict) -> dict:
    tp, fp, fn, tn = (counts[key] for key in COUNTS[:4])
    return {**counts, "iou": _ratio(tp, tp + fp + fn),
            "dice": _ratio(2 * tp, 2 * tp + fp + fn),
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "pixel_accuracy": _ratio(tp + tn, counts["valid_pixels"])}


def _counts(row: dict, maximum: int) -> dict:
    if (not isinstance(row, dict) or any(type(row.get(key)) is not int or not 0 <= row[key] <= maximum for key in COUNTS)
            or row["valid_pixels"] == 0 or sum(row[key] for key in COUNTS[:4]) != row["valid_pixels"]):
        raise ValidationComparisonError("Pixel counts must be consistent nonnegative integers within the native grid.")
    return {key: row[key] for key in COUNTS}


def _read(directory: Path) -> tuple[dict, dict, dict, dict]:
    payloads = {"analysis-report.json": _payload(directory / "analysis-report.json", 8 * 1024 * 1024),
                "samples.json": _payload(directory / "samples.json", 32 * 1024 * 1024)}
    report, samples = (json.loads(payloads[key], object_pairs_hook=_unique, parse_constant=_invalid, parse_float=_float)
                       for key in ("analysis-report.json", "samples.json"))
    if (not isinstance(report, dict) or report.get("purpose") != "frozen-validation-error-analysis"
            or any(report.get(key) is not True for key in ("ok", "same_device_and_libraries", "matches_checkpoint_validation_counts"))
            or report.get("test_used") is not False):
        raise ValidationComparisonError("Expected successful, reproduced, validation-only analyses.")
    for key in ("checkpoint_sha256", "dataset_fingerprint", "val_selection_fingerprint", "samples_sha256"):
        if not isinstance(report.get(key), str) or not HASH.fullmatch(report[key]):
            raise ValidationComparisonError("Analysis identity hashes must be SHA-256 values.")
    dimensions = report.get("image_dimensions")
    if (not isinstance(dimensions, list) or len(dimensions) != 2
            or any(type(side) is not int or not 1 <= side <= 8192 for side in dimensions)
            or not _number(report.get("threshold"), 0, 1) or report["threshold"] == 1
            or not _number(report.get("gsd_m"), 0) or report.get("device") not in ("cpu", "mps", "cuda")
            or not isinstance(report.get("package_versions"), dict) or not report["package_versions"]
            or any(not isinstance(value, str) or not value for value in report["package_versions"].values())
            or type(report.get("checkpoint_epoch")) is not int or report["checkpoint_epoch"] < 1
            or type(report.get("samples")) is not int or not 1 <= report["samples"] <= MAX_SAMPLES):
        raise ValidationComparisonError("Invalid frozen analysis contract.")
    if (_digest(payloads["samples.json"]) != report["samples_sha256"]
            or not isinstance(samples, list) or len(samples) != report["samples"]):
        raise ValidationComparisonError("Analysis sample hash or declared count does not match.")
    rows, total = {}, {key: 0 for key in COUNTS}
    for row in samples:
        counts = _counts(row, dimensions[0] * dimensions[1])
        identifier = row.get("id")
        if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 512
                or identifier in rows or row.get("split") != "val"):
            raise ValidationComparisonError("Validation sample IDs must be nonempty, unique and in the val split.")
        for key, expected in (("target_pv_pixels", counts["tp"] + counts["fn"]),
                              ("predicted_pv_pixels", counts["tp"] + counts["fp"]),
                              ("error_pixels", counts["fp"] + counts["fn"])):
            if type(row.get(key)) is not int or row[key] != expected:
                raise ValidationComparisonError("Derived sample counts disagree with confusion counts.")
        rows[identifier] = counts
        for key in COUNTS:
            total[key] += counts[key]
    summary = report.get("summary")
    if (_counts(summary, dimensions[0] * dimensions[1] * len(rows)) != total
            or type(summary.get("samples")) is not int or summary["samples"] != len(rows)):
        raise ValidationComparisonError("Analysis aggregate counts do not reproduce its sample measurements.")
    ious = [value for row in rows.values() if (value := _metrics(row)["iou"]) is not None]
    metrics = {**_metrics(total), "samples": len(rows), "defined_iou_samples": len(ious),
               "undefined_iou_samples": len(rows) - len(ious),
               "macro_iou": math.fsum(ious) / len(ious) if ious else None,
               "median_iou": median(ious) if ious else None}
    return report, rows, metrics, {key: _digest(value) for key, value in payloads.items()}


def compare_validation_analyses(reference_dir: Path | str, candidate_dir: Path | str,
                                output_dir: Path | str) -> dict:
    """Publish paired crop diagnostics from matching frozen validation selections.

    Ratios are recomputed from integer counts. Undefined IoUs are excluded from
    improved/worse/tied counts and exposed separately. Training configurations
    and checkpoint authenticity are outside this report-only comparison's scope.
    """
    stage = None
    try:
        reference, candidate = Path(reference_dir).resolve(strict=True), Path(candidate_dir).resolve(strict=True)
        output = Path(output_dir).absolute()
        if os.path.lexists(output):
            raise ValidationComparisonError(f"Output already exists: {output}")
        output = output.resolve()
        if any(output.is_relative_to(source) or source.is_relative_to(output) for source in (reference, candidate)):
            raise ValidationComparisonError("Comparison output must be separate from both analyses.")
        a, rows_a, metrics_a, hashes_a = _read(reference)
        b, rows_b, metrics_b, hashes_b = _read(candidate)
        contract = ("val_selection_fingerprint", "threshold", "image_dimensions", "gsd_m", "device", "package_versions")
        if any(a[key] != b[key] for key in contract) or rows_a.keys() != rows_b.keys():
            raise ValidationComparisonError("Analyses differ in frozen validation selection, preprocessing or evaluation environment.")
        paired, classified = [], {key: 0 for key in ("improved", "worse", "tied", "undefined_in_either")}
        for identifier in sorted(rows_a):
            left, right = rows_a[identifier], rows_b[identifier]
            if (left["valid_pixels"] != right["valid_pixels"]
                    or left["tp"] + left["fn"] != right["tp"] + right["fn"]):
                raise ValidationComparisonError("Paired source PV or valid-pixel counts changed.")
            denominators = [row["tp"] + row["fp"] + row["fn"] for row in (left, right)]
            change, category = None, "undefined_in_either"
            if all(denominators):
                change = Fraction(right["tp"], denominators[1]) - Fraction(left["tp"], denominators[0])
                category = "improved" if change > 0 else "worse" if change < 0 else "tied"
            classified[category] += 1
            paired.append({"id": identifier, "split": "val", "reference": _metrics(left), "candidate": _metrics(right),
                           "iou_delta": float(change) if change is not None else None, "iou_comparison": category,
                           "fp_delta": right["fp"] - left["fp"], "fn_delta": right["fn"] - left["fn"]})
        ratio_delta = {key: metrics_b[key] - metrics_a[key] if metrics_a[key] is not None and metrics_b[key] is not None else None
                       for key in RATIOS}
        identities = lambda r, h, m: {**{key: r[key] for key in ("checkpoint_sha256", "checkpoint_epoch", "dataset_fingerprint")},
                                     "analysis_report_sha256": h["analysis-report.json"], "samples_sha256": h["samples.json"], "metrics": m}
        report = {"ok": True, "purpose": "paired-validation-comparison", "samples": len(paired), "test_used": False,
                  **{key: a[key] for key in contract}, "reference": identities(a, hashes_a, metrics_a),
                  "candidate": identities(b, hashes_b, metrics_b), "paired_iou": classified,
                  "delta": {"direction": "candidate minus reference", "counts": {key: metrics_b[key] - metrics_a[key] for key in COUNTS},
                            "ratios": ratio_delta, "percentage_points": {key: value * 100 if value is not None else None for key, value in ratio_delta.items()}},
                  "rankings": {"most_improved": [row["id"] for row in sorted(paired, key=lambda row: (-(row["iou_delta"] or 0), row["id"])) if row["iou_comparison"] == "improved"][:12],
                               "most_worsened": [row["id"] for row in sorted(paired, key=lambda row: (row["iou_delta"] or 0, row["id"])) if row["iou_comparison"] == "worse"][:12]},
                  "limitations": ["Development validation already selected both checkpoints; no independent test or significance result.",
                                  "Pixel errors against upstream labels are not roof-level false-positive/false-negative rates.",
                                  "Matching validation reports do not establish matching training protocols or a causal effect of training changes.",
                                  "No rasters or checkpoints were opened; comparison relies on the retained analysis evidence."]}
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-comparison-", dir=output.parent))
        _write_json(stage / "paired-samples.json", paired)
        report["paired_samples_sha256"] = _digest((stage / "paired-samples.json").read_bytes())
        _write_json(stage / "comparison-report.json", report)
        for source, hashes in ((reference, hashes_a), (candidate, hashes_b)):
            for name, digest in hashes.items():
                if _digest(_payload(source / name, 32 * 1024 * 1024)) != digest:
                    raise ValidationComparisonError("Input analysis evidence changed during comparison.")
        _publish(stage, output)
        stage = None
        return report
    except ValidationComparisonError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError) as error:
        raise ValidationComparisonError(f"Validation comparison failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
