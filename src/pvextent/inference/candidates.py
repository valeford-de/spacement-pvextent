# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Offline predictions for bounded, entirely unannotated candidate snapshots."""

from __future__ import annotations

import hashlib
from html import escape
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from PIL import Image

from pvextent.data.pv4ger import _publish, _sha256, _write_json
from pvextent.data.review import SOURCE_NOTICES
from pvextent.data.validation import MAX_FILE_BYTES, ValidationReport, _path, _read, validate_dataset
from pvextent.training.baseline import _read_checkpoint
from pvextent.training.smoke import _device, _runtime

MAX_CANDIDATES = 24
NOTICE = ("These are model predictions on unreviewed candidate imagery. Source masks remain "
          "entirely ignore (255), and every human review decision remains pending. Predictions "
          "do not establish PV presence, PV absence, or annotation correctness. No accuracy, "
          "false-positive, or false-negative result can be measured from these masks.")


class CandidatePredictionError(ValueError):
    """Candidate inference failed without publishing a partial output bundle."""


def _copy_pixels(source: Path, staging: Path, record: dict, role: str, target: str) -> None:
    report = ValidationReport()
    path = _path(source, record[role], report, record["id"])
    if not report.ok:
        raise CandidatePredictionError(f"Unsafe candidate {role} path: {record['id']}")
    payload = _read(path, MAX_FILE_BYTES, report, record["id"])
    if not report.ok or hashlib.sha256(payload).hexdigest() != record[f"{role}_sha256"]:
        raise CandidatePredictionError(f"Candidate {role} hash/read failure: {record['id']}")
    (staging / target).write_bytes(payload)


def _overlay(image_path: Path, prediction: Image.Image, output: Path) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGBA")
    foreground = prediction.point([0, 255] + [0] * 254)
    tint = Image.new("RGBA", prediction.size, (0, 0, 0, 0))
    tint.paste((0, 200, 150, 120), mask=foreground)
    Image.alpha_composite(rgb, tint).convert("RGB").save(output)


def _gallery(samples: list[dict], fingerprint: str, checkpoint_hash: str) -> str:
    cards = []
    for sample in samples:
        identifier = escape(sample["id"])
        cards.append(f'''<article><h2>{identifier}</h2>
<p>Human review: <strong>pending</strong>. Source annotation: unreviewed; all source pixels ignored.</p>
<figure><a href="{sample['image']}" target="_blank" rel="noopener"><img src="{sample['image']}" alt="{identifier}: original RGB image"></a>
<figcaption>Original RGB image. Click to inspect native pixels before opening predictions.</figcaption></figure>
<details><summary>Show model prediction</summary>
<p>Predicted PV pixels: {sample['predicted_pv_pixels']} ({100 * sample['predicted_pv_fraction']:.2f}% of this crop).
Green in the overlay means the model predicts PV; it is not a verified annotation.</p>
<figure><a href="{sample['overlay']}" target="_blank" rel="noopener"><img src="{sample['overlay']}" alt="{identifier}: model prediction overlay"></a></figure>
<p><a href="{sample['prediction']}">Prediction PNG (0/1)</a> ·
<a href="{sample['mask']}">Unchanged source mask (255 ignore)</a></p>
</details></article>''')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>PVExtent candidate predictions</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 20px;color:#13241d;background:#f5f7f6}
article{background:white;border:1px solid #d6dfda;padding:20px;margin:24px 0}h1{font-size:26px}h2{font-size:18px;overflow-wrap:anywhere}
figure{margin:12px 0}img{display:block;max-width:100%;height:auto;image-rendering:pixelated}summary{cursor:pointer;font-weight:600;padding:12px 0}
code{overflow-wrap:anywhere}a{color:#006e52}.notice{border-left:4px solid #006e52;padding:12px;background:#e6f1eb}</style></head>
<body><h1>Spacement PVExtent · candidate predictions</h1><p class="notice">''' + escape(NOTICE) + '''</p>
<p>Inspect the RGB image first. Prediction panels start closed. Record human decisions in the
<a href="https://github.com/valeford-de/spacement-pvextent/wiki/Label-review">local label-review workflow</a>;
this prediction gallery does not approve, edit or relabel any sample.</p>
<p>Source dataset fingerprint: <code>''' + escape(fingerprint) + '''</code><br>Checkpoint SHA-256:
<code>''' + escape(checkpoint_hash) + '''</code></p>''' + "\n".join(cards) + '''
<footer><a href="review-manifest.json">Pending review manifest</a> ·
<a href="prediction-report.json">Prediction report</a> · <a href="dataset.toml">Source dataset metadata</a></footer></body></html>
'''


def export_candidate_predictions(
    source_dir: Path | str, checkpoint_path: Path | str, output_dir: Path | str, *,
    device: str = "auto", progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Predict native candidate pixels offline; never evaluate placeholder masks."""
    staging = None
    try:
        import torch
        from pvextent.data.training import _read_manifest
        from pvextent.models.resnet_unet import MODEL_NAME, WEIGHT_SPEC, build_resnet_unet, normalize_rgb
        source, checkpoint = Path(source_dir).resolve(), Path(checkpoint_path)
        output = Path(output_dir).absolute()
        if os.path.lexists(output):
            raise CandidatePredictionError(f"Prediction output already exists: {output}")
        output = output.resolve()
        checkpoint_directory = checkpoint.resolve().parent
        if (source.is_relative_to(output) or output.is_relative_to(source)
                or checkpoint_directory.is_relative_to(output) or output.is_relative_to(checkpoint_directory)):
            raise CandidatePredictionError("Candidate source, checkpoint directory and prediction output must be disjoint.")
        metadata, records, fingerprint = _read_manifest(source)
        # This guard precedes any raster path resolution, reads or decoding.
        if (not 1 <= len(records) <= MAX_CANDIDATES
                or any(row["split"] != "unassigned" or row["annotation_status"] != "unreviewed" for row in records)):
            raise CandidatePredictionError("Expected at most 24 exclusively unassigned, unreviewed candidate records.")
        dimensions = {(row["width"], row["height"]) for row in records}
        resolutions = {float(row["gsd_m"]) for row in records}
        if len(dimensions) != 1 or len(resolutions) != 1:
            raise CandidatePredictionError("Candidates require common native dimensions and one GSD; no resizing is performed.")
        width, height = dimensions.pop()
        if any(side < 32 or side > 1024 or side % 32 for side in (width, height)):
            raise CandidatePredictionError("Candidate dimensions must be 32..1024 pixels and divisible by 32.")
        validation = validate_dataset(source)
        if not validation.ok or validation.stats["dataset_fingerprint"] != fingerprint:
            raise CandidatePredictionError("Candidate snapshot is invalid or its declarations changed during validation.")
        if validation.stats["pv_pixels"] or validation.stats["background_pixels"]:
            raise CandidatePredictionError("Every candidate mask must contain only 255 ignore pixels; supervised masks are not accepted.")
        document, checkpoint_hash = _read_checkpoint(checkpoint, torch, resume=False)
        config = document.get("config", {})
        threshold, seed = config.get("threshold"), config.get("seed")
        if (config.get("model") != MODEL_NAME or config.get("encoder") != WEIGHT_SPEC
                or config.get("preprocessing") != "native-float32-rgb-imagenet-mean-std-v1"
                or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 < threshold < 1
                or type(seed) is not int or not 0 <= seed <= 2**32 - 1):
            raise CandidatePredictionError("Checkpoint model, normalization, encoder, threshold or seed is unsupported.")
        selected_device = _device(torch, device)
        notify = progress or (lambda message: None)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-candidates-", dir=output.parent))
        for folder in ("images", "masks", "predictions", "overlays", "provenance"):
            (staging / folder).mkdir()
        metadata_bytes = (source / "dataset.toml").read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != validation.stats["metadata_sha256"]:
            raise CandidatePredictionError("Candidate source metadata changed during export.")
        (staging / "dataset.toml").write_bytes(metadata_bytes)
        samples = []
        for record in records:
            file_id = hashlib.sha256(record["id"].encode()).hexdigest()[:24]
            sample = {"id": record["id"], "source_record": record,
                      "image": f"images/{file_id}{Path(record['image']).suffix.lower()}",
                      "mask": f"masks/{file_id}.png", "prediction": f"predictions/{file_id}.png",
                      "overlay": f"overlays/{file_id}.png", "decision": "pending", "notes": "", "tags": []}
            for role in ("image", "mask"):
                _copy_pixels(source, staging, record, role, sample[role])
            samples.append(sample)
        notices = []
        for name in SOURCE_NOTICES:
            notice = source / "provenance" / name
            if notice.exists():
                if not notice.resolve().is_relative_to(source) or not notice.is_file() or notice.stat().st_size > 1024 * 1024:
                    raise CandidatePredictionError(f"Unsafe or oversized candidate source notice: {notice}")
                shutil.copyfile(notice, staging / "provenance" / name)
                notices.append(f"provenance/{name}")
        with _runtime(torch, selected_device, seed):
            with torch.device("cpu"):
                model = build_resnet_unet(None).to(device=selected_device, dtype=torch.float32)
            model.load_state_dict(document["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                for index, sample in enumerate(samples, 1):
                    with Image.open(staging / sample["image"]) as image:
                        raw = bytearray(image.tobytes())
                    pixels = torch.frombuffer(raw, dtype=torch.uint8).reshape(height, width, 3)
                    inputs = pixels.permute(2, 0, 1).to(torch.float32).div_(255).unsqueeze(0).to(selected_device)
                    logits = model(normalize_rgb(inputs))
                    if tuple(logits.shape) != (1, 1, height, width) or not bool(torch.isfinite(logits).all()):
                        raise CandidatePredictionError("Model returned unsupported or non-finite candidate logits.")
                    prediction = logits.sigmoid().ge(threshold).to(device="cpu", dtype=torch.uint8)[0, 0]
                    predicted = int(prediction.sum().item())
                    mask = Image.frombytes("L", (width, height), bytes(prediction.flatten().tolist()))
                    mask.save(staging / sample["prediction"])
                    _overlay(staging / sample["image"], mask, staging / sample["overlay"])
                    sample.update(predicted_pv_pixels=predicted, predicted_pv_fraction=predicted / (width * height),
                                  prediction_sha256=_sha256(staging / sample["prediction"]),
                                  overlay_sha256=_sha256(staging / sample["overlay"]))
                    notify(f"Predicted candidate {index}/{len(samples)}: {sample['id']}")
        review = {"candidate_prediction_schema_version": "1.0", "dataset_fingerprint": fingerprint,
                  "checkpoint_sha256": checkpoint_hash, "threshold": threshold, "notice": NOTICE,
                  "source_metadata": metadata, "source_notices": notices, "source_masks_unchanged": True,
                  "predictions_are_annotations": False, "samples": samples}
        _write_json(staging / "review-manifest.json", review)
        report = {
            "ok": True, "records": len(samples), "model": MODEL_NAME, "device": selected_device,
            "threshold": threshold, "dataset_fingerprint": fingerprint, "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": document.get("epoch"), "image_dimensions": [width, height],
            "source_gsd_m": resolutions.pop(), "training_gsd_m": None,
            "source_annotation_status": "unreviewed", "human_review_status": "pending",
            "evaluation_performed": False, "domain_transfer_evaluated": False,
            "outside_training_snapshot": fingerprint != config.get("dataset_fingerprint"),
            "domain_warning": "Outside the training snapshot; source-domain transfer and PV accuracy are unmeasured. Training GSD is not declared in this checkpoint.",
            "prediction_values": {"0": "model predicts background", "1": "model predicts PV"},
            "review_manifest_sha256": _sha256(staging / "review-manifest.json"),
            "package_versions": {key: version(key) for key in (
                "torch", "torchvision", "segmentation-models-pytorch", "timm", "safetensors")},
            "checkpoint_package_versions": config.get("package_versions"),
            "samples": [{key: sample[key] for key in ("id", "predicted_pv_pixels", "predicted_pv_fraction")}
                        for sample in samples],
            "limitations": [NOTICE, "Native source dimensions and GSD are preserved; no resampling is performed.",
                            "Candidate imagery is not an independent labelled benchmark."],
        }
        report["package_versions_match_checkpoint"] = (
            report["package_versions"] == report["checkpoint_package_versions"]
            if isinstance(report["checkpoint_package_versions"], dict) else None)
        _write_json(staging / "prediction-report.json", report)
        (staging / "index.html").write_text(_gallery(samples, fingerprint, checkpoint_hash), encoding="utf-8")
        _publish(staging, output)
        staging = None
        return report
    except CandidatePredictionError:
        raise
    except ImportError as error:
        raise CandidatePredictionError("Candidate inference dependencies missing. Run: uv sync --locked --extra baseline") from error
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        raise CandidatePredictionError(f"Candidate predictions failed: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
