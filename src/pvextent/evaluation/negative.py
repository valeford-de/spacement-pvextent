# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Frozen-checkpoint pixel diagnostics on a separately frozen negative set."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from PIL import Image

from pvextent.data.curation import _bounded_bytes
from pvextent.data.pv4ger import _publish, _sha256, _write_json
from pvextent.data.validation import SHA256, _invalid_constant, _unique_object

COUNT_FIELDS = ("false_positive_pixels", "true_negative_pixels", "supervised_background_pixels",
                "ignored_pixels", "predicted_pv_pixels_in_ignored_region_unscored")


class NegativeEvaluationError(ValueError):
    """The proposed evaluation differs from the frozen negative protocol."""


def _document(path: Path) -> tuple[dict, bytes]:
    payload = _bounded_bytes(path, 2 * 1024 * 1024)
    document = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(document, dict):
        raise NegativeEvaluationError("Expected a bounded JSON object.")
    return document, payload


def evaluate_negative_holdout(dataset_dir: Path | str, reference_checkpoint: Path | str,
                              candidate_checkpoint: Path | str, output_dir: Path | str, *,
                              device: str = "auto", progress: Callable[[str], None] | None = None) -> dict:
    """Score only approved background pixels; ignored predictions stay unscored.

    This does not compute a roof-level false-positive rate or a human-confirmed
    benchmark score. It never constructs a training optimizer or opens any
    raster outside the explicit new negative holdout's selected test records.
    """
    stage = None
    notify = progress or (lambda message: None)
    try:
        import torch
        from pvextent.data.training import _read_manifest, _raster
        from pvextent.models.resnet_unet import MODEL_NAME, WEIGHT_SPEC, build_resnet_unet, normalize_rgb
        from pvextent.training.baseline import MAX_CHECKPOINT_BYTES, _read_checkpoint
        from pvextent.training.smoke import _device, _runtime

        source = Path(dataset_dir).resolve(strict=True)
        checkpoints = {"reference": Path(reference_checkpoint).absolute(), "candidate": Path(candidate_checkpoint).absolute()}
        output = Path(output_dir).absolute()
        if os.path.lexists(output):
            raise NegativeEvaluationError(f"Output already exists: {output}")
        output = output.resolve()
        if any(output.is_relative_to(path) or path.is_relative_to(output)
               for path in (source, *(p.resolve().parent for p in checkpoints.values()))):
            raise NegativeEvaluationError("Evaluation output must be separate from dataset and checkpoint runs.")
        metadata, records, fingerprint = _read_manifest(source)
        if metadata["dataset_id"] != "pvextent-negative-holdout" or not 1 <= len(records) <= 24:
            raise NegativeEvaluationError("Only the explicit, separately frozen negative holdout may be evaluated.")
        evidence_names = ("negative-holdout-freeze.json", "protocol.json", "acquisition-protocol.json")
        evidence = {name: _document(source / "provenance" / name) for name in evidence_names}
        freeze, protocol, acquisition = (evidence[name][0] for name in evidence_names)
        digests = {name: hashlib.sha256(payload).hexdigest() for name, (_, payload) in evidence.items()}
        minimum_background = protocol.get("minimum_background_fraction")
        if (freeze.get("ok") is not True or freeze.get("purpose") != "experimental-ai-negative-holdout"
                or protocol.get("purpose") != "experimental-ai-negative-holdout" or protocol.get("schema_version") != "1.0"
                or freeze.get("dataset_fingerprint") != fingerprint
                or freeze.get("protocol_sha256") != digests["protocol.json"]
                or freeze.get("acquisition_protocol_sha256") != digests["acquisition-protocol.json"]
                or protocol.get("acquisition_protocol_sha256") != digests["acquisition-protocol.json"]
                or any(document.get("frozen_before_predictions") is not True for document in (freeze, protocol))
                or any(document.get("model_predictions_used_for_labels") is not False for document in (freeze, protocol))
                or any(document.get("human_review_claimed") is not False for document in (freeze, protocol))
                or type(minimum_background) is not float or minimum_background not in (0.5, 0.05)
                or freeze.get("minimum_background_fraction") != minimum_background
                or protocol.get("threshold") != 0.5 or freeze.get("training_use_allowed") is not False):
            raise NegativeEvaluationError("Negative holdout freeze or protocol evidence is invalid or changed.")
        selected_ids = freeze.get("reviewed_evaluation_ids")
        if (not isinstance(selected_ids, list) or not selected_ids
                or any(not isinstance(key, str) for key in selected_ids)
                or len(set(selected_ids)) != len(selected_ids)
                or any(row["split"] not in ("test", "unassigned") for row in records)):
            raise NegativeEvaluationError("Frozen selection must identify unique new test records only.")
        selected = sorted((row for row in records if row["split"] == "test"), key=lambda row: row["id"])
        if (set(selected_ids) != {row["id"] for row in selected}
                or any(row["annotation_status"] != "reviewed" or row["width"] != 320 or row["height"] != 320
                       or row["gsd_m"] != 0.1 for row in selected)):
            raise NegativeEvaluationError("Frozen IDs, review status or native grid differ from the holdout contract.")
        frozen_hashes = freeze.get("checkpoint_sha256")
        if (not isinstance(frozen_hashes, dict) or set(frozen_hashes) != set(checkpoints)
                or frozen_hashes != protocol.get("checkpoint_sha256")
                or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in frozen_hashes.values())):
            raise NegativeEvaluationError("Both checkpoint identities must be frozen in the evaluation protocol.")
        items = []
        for row in selected:
            mask = _raster(source, row, "mask")
            histogram = mask.histogram()
            if sum(histogram[1:255]) or histogram[0] < 320 * 320 * minimum_background:
                raise NegativeEvaluationError("Every negative mask must meet its frozen background-coverage gate and contain only labels 0/255.")
            labels = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(1, 320, 320)
            items.append({"record": row, "valid": labels.eq(0)})
        documents, hashes = {}, {}
        for role, path in checkpoints.items():
            if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES
                    or _sha256(path) != frozen_hashes[role]):
                raise NegativeEvaluationError("Checkpoint bytes differ from the frozen identity; weights were not loaded.")
        for role, path in checkpoints.items():
            document, digest = _read_checkpoint(path, torch, resume=False)
            config = document.get("config", {})
            if (digest != frozen_hashes[role] or config.get("model") != MODEL_NAME or config.get("encoder") != WEIGHT_SPEC
                    or config.get("preprocessing") != "native-float32-rgb-imagenet-mean-std-v1"
                    or config.get("threshold") != 0.5 or type(config.get("seed")) is not int
                    or not 0 <= config["seed"] <= 2**32 - 1 or not isinstance(config.get("package_versions"), dict)):
                raise NegativeEvaluationError("Checkpoint identity, model or preprocessing differs from the frozen protocol.")
            documents[role], hashes[role] = document, digest
        if documents["reference"]["config"]["seed"] != documents["candidate"]["config"]["seed"]:
            raise NegativeEvaluationError("The two checkpoint initialization seeds differ.")
        selected_device = _device(torch, device)
        libraries = {key: version(key) for key in ("torch", "torchvision", "segmentation-models-pytorch", "timm", "safetensors")}
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-negative-evaluation-", dir=output.parent))
        (stage / "predictions").mkdir()
        rows = [{"id": item["record"]["id"], "split": "test", "image_sha256": item["record"]["image_sha256"],
                 "mask_sha256": item["record"]["mask_sha256"]} for item in items]
        results = {}
        for role, document in documents.items():
            config = document["config"]
            with _runtime(torch, selected_device, config["seed"]), torch.no_grad():
                with torch.device("cpu"):
                    model = build_resnet_unet(None).to(selected_device).eval()
                model.load_state_dict(document["model_state_dict"], strict=True)
                for start in range(0, len(items), 2):
                    batch = items[start:start + 2]
                    images = []
                    for item in batch:
                        image = _raster(source, item["record"], "image")
                        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(320, 320, 3)
                        images.append(pixels.permute(2, 0, 1).to(torch.float32).div_(255))
                    logits = model(normalize_rgb(torch.stack(images).to(selected_device)))
                    if tuple(logits.shape) != (len(batch), 1, 320, 320) or not bool(torch.isfinite(logits).all()):
                        raise NegativeEvaluationError("Model produced invalid or non-finite native-grid logits.")
                    predictions = logits.sigmoid().ge(0.5).cpu()
                    for offset, (item, prediction) in enumerate(zip(batch, predictions)):
                        valid = item["valid"]
                        fp, total = int((prediction & valid).sum()), int(valid.sum())
                        key = hashlib.sha256(item["record"]["id"].encode()).hexdigest()[:24]
                        relative = f"predictions/{role}-{key}.png"
                        Image.frombytes("L", (320, 320), bytes(prediction[0].to(torch.uint8).flatten().tolist())).save(stage / relative)
                        rows[start + offset][role] = {"false_positive_pixels": fp, "true_negative_pixels": total - fp,
                            "supervised_background_pixels": total, "ignored_pixels": 320 * 320 - total,
                            "predicted_pv_pixels_in_ignored_region_unscored": int((prediction & ~valid).sum()),
                            "pixel_false_positive_rate": fp / total, "prediction": relative,
                            "prediction_sha256": _sha256(stage / relative)}
                    notify(f"Negative holdout {role}: {min(start + 2, len(items))}/{len(items)} crops")
                del model
            totals = {key: sum(row[role][key] for row in rows) for key in COUNT_FIELDS}
            totals["pixel_false_positive_rate"] = totals["false_positive_pixels"] / totals["supervised_background_pixels"]
            totals["crops_with_any_false_positive_pixels"] = sum(row[role]["false_positive_pixels"] > 0 for row in rows)
            results[role] = {"checkpoint_sha256": hashes[role], "checkpoint_epoch": document.get("epoch"),
                             "checkpoint_device": config.get("device"), "checkpoint_package_versions": config["package_versions"],
                             "same_device_and_libraries": config.get("device") == selected_device and config["package_versions"] == libraries,
                             "totals": totals}
        left, right = results["reference"]["totals"], results["candidate"]["totals"]
        rate_delta = right["pixel_false_positive_rate"] - left["pixel_false_positive_rate"]
        report = {"ok": True, "purpose": "experimental-ai-negative-holdout-evaluation", "samples": len(rows),
                  "dataset_fingerprint": fingerprint, "protocol_sha256": digests["protocol.json"],
                  "freeze_report_sha256": digests["negative-holdout-freeze.json"], "threshold": 0.5,
                  "device": selected_device, "package_versions": libraries, "batch_size": 2,
                  "image_dimensions": [320, 320], "gsd_m": 0.1, "source_metadata": metadata,
                  "minimum_background_fraction": minimum_background,
                  "results": results, "delta": {"direction": "candidate minus reference",
                      **{key: right[key] - left[key] for key in COUNT_FIELDS},
                      "pixel_false_positive_rate": rate_delta, "pixel_false_positive_rate_percentage_points": 100 * rate_delta},
                  "supervised_background_coverage": left["supervised_background_pixels"] / (len(rows) * 320 * 320),
                  "old_test_pixels_opened": False, "training_performed": False, "model_predictions_used_for_labels": False,
                  "human_review_claimed": False,
                  "limitations": ["Exploratory spatially separate AI-reviewed partial negatives, not human-confirmed benchmark truth.",
                      "Only approved background pixels are scored; predictions in ignored areas are unscored.",
                      "Samples with any false-positive pixels are crop diagnostics, not a roof-level false-positive rate.",
                      "This selected small collection does not estimate real-world prevalence or recall on PV-positive roofs."]}
        _write_json(stage / "samples.json", rows)
        report["samples_sha256"] = _sha256(stage / "samples.json")
        if _read_manifest(source)[2] != fingerprint:
            raise NegativeEvaluationError("Source declarations changed during evaluation.")
        for name, digest in digests.items():
            if hashlib.sha256(_document(source / "provenance" / name)[1]).hexdigest() != digest:
                raise NegativeEvaluationError("Frozen evaluation evidence changed during evaluation.")
        if any(_sha256(path) != hashes[role] for role, path in checkpoints.items()):
            raise NegativeEvaluationError("Checkpoint bytes changed during evaluation.")
        _write_json(stage / "negative-evaluation-report.json", report)
        _publish(stage, output)
        stage = None
        return report
    except NegativeEvaluationError:
        raise
    except ImportError as error:
        raise NegativeEvaluationError("Install the baseline extra for negative holdout evaluation.") from error
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, OverflowError, RecursionError) as error:
        raise NegativeEvaluationError(f"Negative holdout evaluation failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
