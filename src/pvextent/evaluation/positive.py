# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Frozen-checkpoint diagnostics on explicitly approved partial PV labels."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from PIL import Image

from pvextent.data.curation import _bounded_bytes
from pvextent.data.pv4ger import _publish, _sha256, _write_json
from pvextent.data.validation import SHA256
from pvextent.evaluation.negative import _document

COUNT_FIELDS = ("true_positive_pixels", "false_negative_pixels", "false_positive_pixels", "true_negative_pixels",
                "supervised_positive_pixels", "supervised_background_pixels", "ignored_pixels",
                "predicted_pv_pixels_in_ignored_region_unscored")


class PositiveEvaluationError(ValueError):
    """The proposed evaluation differs from the frozen positive protocol."""


def _rates(counts: dict) -> dict:
    positive, background = counts["supervised_positive_pixels"], counts["supervised_background_pixels"]
    return {"pixel_recall_on_consensus_pv": counts["true_positive_pixels"] / positive,
            "missed_consensus_pv_pixel_fraction": counts["false_negative_pixels"] / positive,
            "background_pixel_false_positive_rate": counts["false_positive_pixels"] / background if background else None}


def evaluate_positive_holdout(dataset_dir: Path | str, reference_checkpoint: Path | str,
                              candidate_checkpoint: Path | str, output_dir: Path | str, *,
                              device: str = "auto", progress: Callable[[str], None] | None = None) -> dict:
    """Score confirmed PV and background separately; label 255 remains unscored.

    Recall concerns only the frozen AI-consensus PV pixels. It is neither
    complete-array/roof recall nor human-confirmed benchmark performance.
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
            raise PositiveEvaluationError(f"Output already exists: {output}")
        output = output.resolve()
        if any(output.is_relative_to(path) or path.is_relative_to(output)
               for path in (source, *(p.resolve().parent for p in checkpoints.values()))):
            raise PositiveEvaluationError("Evaluation output must be separate from dataset and checkpoint runs.")
        metadata, records, fingerprint = _read_manifest(source)
        if metadata["dataset_id"] != "pvextent-positive-holdout" or len(records) != 13:
            raise PositiveEvaluationError("Only the explicit, separately frozen thirteen-crop positive cohort may be evaluated.")
        evidence_names = ("positive-holdout-freeze.json", "protocol.json")
        evidence = {name: _document(source / "provenance" / name) for name in evidence_names}
        freeze, protocol = (evidence[name][0] for name in evidence_names)
        digests = {name: hashlib.sha256(payload).hexdigest() for name, (_, payload) in evidence.items()}
        if (freeze.get("ok") is not True or freeze.get("purpose") != "experimental-ai-positive-holdout"
                or protocol.get("purpose") != "experimental-ai-positive-holdout" or protocol.get("schema_version") != "1.0"
                or freeze.get("dataset_fingerprint") != fingerprint or freeze.get("protocol_sha256") != digests["protocol.json"]
                or any(document.get("frozen_before_predictions") is not True for document in (freeze, protocol))
                or any(document.get("model_predictions_used_for_labels") is not False for document in (freeze, protocol))
                or any(document.get("human_review_claimed") is not False for document in (freeze, protocol))
                or any(document.get("training_use_allowed") is not False for document in (freeze, protocol))
                or any(type(document.get("minimum_positive_pixels")) is not int or document["minimum_positive_pixels"] != 100
                       for document in (freeze, protocol)) or protocol.get("threshold") != 0.5):
            raise PositiveEvaluationError("Positive holdout freeze or protocol evidence is invalid or changed.")
        cohort_ids, selected_ids = freeze.get("selected_ids"), freeze.get("reviewed_evaluation_ids")
        if (not isinstance(cohort_ids, list) or any(not isinstance(key, str) for key in cohort_ids)
                or len(cohort_ids) != len(set(cohort_ids)) or set(cohort_ids) != {row["id"] for row in records}
                or cohort_ids != protocol.get("selected_ids")
                or protocol.get("selected_image_sha256") != {row["id"]: row["image_sha256"] for row in records}
                or not isinstance(selected_ids, list) or not selected_ids or any(not isinstance(key, str) for key in selected_ids)
                or len(selected_ids) != len(set(selected_ids)) or any(row["split"] not in ("test", "unassigned") for row in records)):
            raise PositiveEvaluationError("Frozen selection must identify the complete cohort and unique new test records only.")
        selected = sorted((row for row in records if row["split"] == "test"), key=lambda row: row["id"])
        if (set(selected_ids) != {row["id"] for row in selected}
                or any(row["annotation_status"] != "reviewed" or row["width"] != 320 or row["height"] != 320
                       or row["gsd_m"] != 0.1 for row in selected)):
            raise PositiveEvaluationError("Frozen IDs, review status or native grid differ from the holdout contract.")
        frozen_hashes = freeze.get("checkpoint_sha256")
        if (not isinstance(frozen_hashes, dict) or set(frozen_hashes) != set(checkpoints)
                or frozen_hashes != protocol.get("checkpoint_sha256")
                or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in frozen_hashes.values())):
            raise PositiveEvaluationError("Both checkpoint identities must be frozen in the evaluation protocol.")
        items = []
        for row in selected:
            mask = _raster(source, row, "mask")
            histogram = mask.histogram()
            if histogram[1] < 100 or sum(histogram[2:255]):
                raise PositiveEvaluationError("Every positive mask requires at least 100 PV pixels and labels 0/1/255 only.")
            labels = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(1, 320, 320)
            items.append({"record": row, "positive": labels.eq(1), "background": labels.eq(0), "ignored": labels.eq(255)})
        documents, hashes, companions = {}, {}, {}
        for role, path in checkpoints.items():
            if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES
                    or _sha256(path) != frozen_hashes[role]):
                raise PositiveEvaluationError("Checkpoint bytes differ from the frozen identity; weights were not loaded.")
        for role, path in checkpoints.items():
            document, digest = _read_checkpoint(path, torch, resume=False)
            config = document.get("config", {})
            if (digest != frozen_hashes[role] or config.get("model") != MODEL_NAME or config.get("encoder") != WEIGHT_SPEC
                    or config.get("preprocessing") != "native-float32-rgb-imagenet-mean-std-v1"
                    or config.get("threshold") != 0.5 or type(config.get("seed")) is not int
                    or not 0 <= config["seed"] <= 2**32 - 1 or not isinstance(config.get("package_versions"), dict)):
                raise PositiveEvaluationError("Checkpoint identity, model or preprocessing differs from the frozen protocol.")
            documents[role], hashes[role] = document, digest
            companion = _bounded_bytes(path.with_suffix(path.suffix + ".sha256"), 128)
            if companion.decode("ascii").strip() != digest:
                raise PositiveEvaluationError("Checkpoint companion differs from the frozen identity.")
            companions[role] = hashlib.sha256(companion).hexdigest()
        if documents["reference"]["config"]["seed"] != documents["candidate"]["config"]["seed"]:
            raise PositiveEvaluationError("The two checkpoint initialization seeds differ.")
        selected_device = _device(torch, device)
        libraries = {key: version(key) for key in ("torch", "torchvision", "segmentation-models-pytorch", "timm", "safetensors")}
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-positive-evaluation-", dir=output.parent))
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
                    batch, images = items[start:start + 2], []
                    for item in batch:
                        image = _raster(source, item["record"], "image")
                        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(320, 320, 3)
                        images.append(pixels.permute(2, 0, 1).to(torch.float32).div_(255))
                    logits = model(normalize_rgb(torch.stack(images).to(selected_device)))
                    if tuple(logits.shape) != (len(batch), 1, 320, 320) or not bool(torch.isfinite(logits).all()):
                        raise PositiveEvaluationError("Model produced invalid or non-finite native-grid logits.")
                    predictions = logits.sigmoid().ge(0.5).cpu()
                    for offset, (item, prediction) in enumerate(zip(batch, predictions)):
                        tp, fp = int((prediction & item["positive"]).sum()), int((prediction & item["background"]).sum())
                        positive, background = int(item["positive"].sum()), int(item["background"].sum())
                        counts = dict(zip(COUNT_FIELDS, (tp, positive - tp, fp, background - fp, positive, background,
                            int(item["ignored"].sum()), int((prediction & item["ignored"]).sum()))))
                        key = hashlib.sha256(item["record"]["id"].encode()).hexdigest()[:24]
                        relative = f"predictions/{role}-{key}.png"
                        Image.frombytes("L", (320, 320), bytes(prediction[0].to(torch.uint8).flatten().tolist())).save(stage / relative)
                        rows[start + offset][role] = {**counts, **_rates(counts), "prediction": relative,
                                                     "prediction_sha256": _sha256(stage / relative)}
                    notify(f"Positive holdout {role}: {min(start + 2, len(items))}/{len(items)} crops")
                del model
            totals = {key: sum(row[role][key] for row in rows) for key in COUNT_FIELDS}
            totals.update(_rates(totals))
            totals["crops_with_zero_true_positive_pixels"] = sum(row[role]["true_positive_pixels"] == 0 for row in rows)
            results[role] = {"checkpoint_sha256": hashes[role], "checkpoint_epoch": document.get("epoch"),
                             "checkpoint_device": config.get("device"), "checkpoint_package_versions": config["package_versions"],
                             "same_device_and_libraries": config.get("device") == selected_device and config["package_versions"] == libraries,
                             "totals": totals}
        left, right = results["reference"]["totals"], results["candidate"]["totals"]
        delta = {"direction": "candidate minus reference", **{key: right[key] - left[key] for key in COUNT_FIELDS}}
        for key in _rates(left):
            delta[key] = right[key] - left[key] if left[key] is not None and right[key] is not None else None
            delta[f"{key}_percentage_points"] = 100 * delta[key] if delta[key] is not None else None
        delta["crops_with_zero_true_positive_pixels"] = right["crops_with_zero_true_positive_pixels"] - left["crops_with_zero_true_positive_pixels"]
        report = {"ok": True, "purpose": "experimental-ai-positive-holdout-evaluation", "samples": len(rows),
                  "dataset_fingerprint": fingerprint, "protocol_sha256": digests["protocol.json"],
                  "freeze_report_sha256": digests["positive-holdout-freeze.json"], "threshold": 0.5,
                  "device": selected_device, "package_versions": libraries, "batch_size": 2,
                  "image_dimensions": [320, 320], "gsd_m": 0.1, "source_metadata": metadata, "minimum_positive_pixels": 100,
                  "results": results, "delta": delta,
                  "supervised_positive_coverage": left["supervised_positive_pixels"] / (len(rows) * 320 * 320),
                  "supervised_background_coverage": left["supervised_background_pixels"] / (len(rows) * 320 * 320),
                  "ignored_coverage": left["ignored_pixels"] / (len(rows) * 320 * 320),
                  "old_test_pixels_opened": False, "negative_holdout_pixels_opened": False, "training_performed": False,
                  "model_predictions_used_for_labels": False, "human_review_claimed": False, "training_use_allowed": False,
                  "limitations": ["Exploratory AI-consensus partial positive labels, not human-confirmed benchmark truth.",
                      "Recall and missed fraction score confirmed PV pixels only; ignored predictions are unscored.",
                      "Background counts and rates concern only independently confirmed partial background.",
                      "Zero-TP counts are crop diagnostics, not object, array or roof recall.",
                      "This small selected cohort does not estimate representative real-world performance or prevalence."]}
        _write_json(stage / "samples.json", rows)
        report["samples_sha256"] = _sha256(stage / "samples.json")
        if _read_manifest(source)[2] != fingerprint:
            raise PositiveEvaluationError("Source declarations changed during evaluation.")
        for name, digest in digests.items():
            if hashlib.sha256(_document(source / "provenance" / name)[1]).hexdigest() != digest:
                raise PositiveEvaluationError("Frozen evaluation evidence changed during evaluation.")
        for row in selected:
            for kind in ("image", "mask"):
                _raster(source, row, kind)
        if any(_sha256(path) != hashes[role] for role, path in checkpoints.items()):
            raise PositiveEvaluationError("Checkpoint bytes changed during evaluation.")
        if any(hashlib.sha256(_bounded_bytes(path.with_suffix(path.suffix + ".sha256"), 128)).hexdigest() != companions[role]
               for role, path in checkpoints.items()):
            raise PositiveEvaluationError("Checkpoint companion bytes changed during evaluation.")
        _write_json(stage / "positive-evaluation-report.json", report)
        _publish(stage, output)
        stage = None
        return report
    except PositiveEvaluationError:
        raise
    except ImportError as error:
        raise PositiveEvaluationError("Install the baseline extra for positive holdout evaluation.") from error
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, OverflowError, RecursionError) as error:
        raise PositiveEvaluationError(f"Positive holdout evaluation failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
