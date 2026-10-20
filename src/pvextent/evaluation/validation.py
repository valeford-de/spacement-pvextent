# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Validation-only error analysis of a frozen segmentation checkpoint."""

from __future__ import annotations

import hashlib
from html import escape
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from PIL import Image

from pvextent.data.pv4ger import _publish, _sha256, _write_json


class ValidationAnalysisError(ValueError):
    """The requested analysis cannot be published as a complete result."""


def _gallery(rows: list[dict], report: dict) -> str:
    cards = []
    for row in rows:
        figures = "".join(
            f'<figure><a href="{row["views"][key]}" target="_blank" rel="noopener">'
            f'<img src="{row["views"][key]}" alt="{escape(label)}" loading="lazy"></a>'
            f'<figcaption>{escape(label)}</figcaption></figure>'
            for key, label in (("image", "RGB"), ("target", "Upstream PV labels"),
                               ("prediction", "Predicted PV"), ("errors", "Pixel errors")))
        score = "undefined" if row["iou"] is None else f'{row["iou"]:.4f}'
        cards.append(f'<article><h2>{escape(row["id"])}</h2>'
                     f'<p>IoU {score}; FP {row["fp"]:,}; FN {row["fn"]:,}; '
                     f'source PV area {row["target_pv_area_m2"]:.2f} m².<br>'
                     f'Selected for: {escape(", ".join(row["selected_for"]))}</p>'
                     f'<div class="views">{figures}</div></article>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>PVExtent validation errors</title>
<style>body{font:16px/1.5 system-ui;margin:24px;color:#17241f;background:#f4f5f6}
main{max-width:1440px;margin:auto}article{padding:16px;margin:24px 0;background:white}
.views{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
figure{margin:0}img{width:100%;height:auto;image-rendering:pixelated}h2{font-size:18px;overflow-wrap:anywhere}
@media(max-width:800px){.views{grid-template-columns:repeat(2,minmax(0,1fr))}}</style><main>
<h1>Spacement PVExtent · validation error review</h1>
<p>Selected difficult validation examples, not a representative performance sample.
The checkpoint and threshold are frozen. Source labels remain upstream annotations;
these differences are review candidates, not confirmed annotation errors.</p>
<p>Green = TP; red = FP; blue = FN; gray = TN; orange = ignore.
Click images for native-resolution inspection. No test imagery is included.</p>
<p>Checkpoint SHA-256: <code>''' + escape(report["checkpoint_sha256"]) + '''</code></p>
<p><a href="analysis-report.json">Aggregate report</a> ·
<a href="samples.json">All validation sample measurements</a></p>''' + "\n".join(cards) + '</main></html>\n'


def _views(stage: Path, identifier: str, item: dict, prediction, torch) -> dict:
    """Render diagnostic rasters; all real pixels stay in the local output."""
    key = hashlib.sha256(identifier.encode()).hexdigest()[:24]
    height, width = prediction.shape
    target, valid = item["target"][0].eq(1), item["valid"][0]
    rgb = item["image"].mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    errors = torch.full((height, width, 3), 45, dtype=torch.uint8)
    for selector, color in ((valid & prediction & target, (0, 190, 120)),
                            (valid & prediction & ~target, (220, 45, 45)),
                            (valid & ~prediction & target, (50, 110, 230)),
                            (~valid, (245, 170, 50))):
        errors[selector] = torch.tensor(color, dtype=torch.uint8)
    panels = {"image": ("RGB", rgb), "target": ("L", target.to(torch.uint8).mul(255)),
              "prediction": ("L", prediction.to(torch.uint8).mul(255)), "errors": ("RGB", errors)}
    paths = {}
    for name, (mode, pixels) in panels.items():
        relative = f"views/{key}-{name}.png"
        Image.frombytes(mode, (width, height), bytes(pixels.flatten().tolist())).save(stage / relative)
        paths[name] = relative
    return paths


def analyze_validation(dataset_dir: Path | str, checkpoint_path: Path | str, output_dir: Path | str,
                       *, device: str = "auto", top_k: int = 12,
                       progress: Callable[[str], None] | None = None) -> dict:
    """Measure every selected validation crop without changing labels or weights."""
    stage = None
    notify = progress or (lambda message: None)
    try:
        import torch
        from pvextent.data.segmentation import SegmentationDataset
        from pvextent.evaluation.error_metrics import sample_statistics, aggregate_statistics
        from pvextent.models.resnet_unet import MODEL_NAME, WEIGHT_SPEC, build_resnet_unet, normalize_rgb
        from pvextent.training.baseline import _read_checkpoint
        from pvextent.training.smoke import _device, _runtime

        if type(top_k) is not int or not 1 <= top_k <= 24:
            raise ValidationAnalysisError("top_k must be an integer from 1 to 24.")
        source, checkpoint, output = Path(dataset_dir).resolve(), Path(checkpoint_path).absolute(), Path(output_dir).absolute()
        if os.path.lexists(output):
            raise ValidationAnalysisError(f"Output already exists: {output}")
        output = output.resolve()
        if (source.is_relative_to(output) or output.is_relative_to(source)
                or output.is_relative_to(checkpoint.parent.resolve())):
            raise ValidationAnalysisError("Analysis output must be separate from the dataset and checkpoint run.")
        document, digest = _read_checkpoint(checkpoint, torch, resume=False)
        config = document["config"]
        if (config["model"] != MODEL_NAME or config["encoder"] != WEIGHT_SPEC
                or config["preprocessing"] != "native-float32-rgb-imagenet-mean-std-v1"
                or not 0 < config["threshold"] < 1):
            raise ValidationAnalysisError("Expected the fixed baseline model, encoder, normalization and frozen threshold.")
        dataset = SegmentationDataset(source, split="val", max_samples=config["max_val_samples"] or None,
                                      seed=config["seed"])
        if (dataset.dataset_fingerprint != config["dataset_fingerprint"]
                or dataset.selection_fingerprint != config["val_selection_fingerprint"]):
            raise ValidationAnalysisError("Validation snapshot differs from the checkpoint's frozen selection.")
        selected_device = _device(torch, device)
        libraries = {key: version(key) for key in config["package_versions"]}
        comparable = libraries == config["package_versions"] and selected_device == config["device"]
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-analysis-", dir=output.parent))
        (stage / "predictions").mkdir()
        (stage / "views").mkdir()
        (stage / "provenance").mkdir()
        (stage / "dataset.toml").write_bytes((source / "dataset.toml").read_bytes())
        from pvextent.data.review import SOURCE_NOTICES
        notices = []
        for name in SOURCE_NOTICES:
            notice = source / "provenance" / name
            if notice.exists():
                if not notice.resolve().is_relative_to(source) or not notice.is_file() or notice.stat().st_size > 1024 * 1024:
                    raise ValidationAnalysisError("Invalid or oversized source notice.")
                shutil.copyfile(notice, stage / "provenance" / name)
                notices.append(f"provenance/{name}")
        rows = []
        with _runtime(torch, selected_device, config["seed"]), torch.no_grad():
            with torch.device("cpu"):
                model = build_resnet_unet(None).to(selected_device).eval()
            model.load_state_dict(document["model_state_dict"], strict=True)
            for start in range(0, len(dataset), config["batch_size"]):
                items = [dataset[index] for index in range(start, min(start + config["batch_size"], len(dataset)))]
                images = torch.stack([item["image"] for item in items]).to(selected_device)
                logits = model(normalize_rgb(images))
                if logits.shape != (len(items), 1, images.shape[2], images.shape[3]) or not bool(torch.isfinite(logits).all()):
                    raise ValidationAnalysisError("Model produced invalid segmentation logits.")
                predictions = logits.sigmoid().ge(config["threshold"]).cpu()
                for item, prediction in zip(items, predictions):
                    identifier = item["id"]
                    statistics = sample_statistics(prediction[0], item["target"][0].eq(1), item["valid"][0], gsd_m=dataset.gsd_m)
                    relative = f'predictions/{hashlib.sha256(identifier.encode()).hexdigest()[:24]}.png'
                    height, width = prediction.shape[1:]
                    Image.frombytes("L", (width, height), bytes(prediction[0].to(torch.uint8).flatten().tolist())).save(stage / relative)
                    rows.append({"id": identifier, "split": "val", "source_id": dataset.records[len(rows)]["source_id"],
                                 "prediction": relative, **statistics})
                notify(f"Validation analysis: {min(start + len(items), len(dataset))}/{len(dataset)} crops") if start % 50 == 0 else None
        summary = aggregate_statistics(rows)
        actual_counts = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
        matches = all(actual_counts[key] == document["validation"][key] for key in actual_counts)
        if comparable and not matches:
            raise ValidationAnalysisError("Recorded checkpoint validation counts were not reproduced on the same device/libraries.")
        selections: dict[str, list[str]] = {}
        ranked = (("lowest-iou", sorted(rows, key=lambda row: (row["iou"] is None, row["iou"] or 0, row["id"]))),
                  ("most-fp-pixels", sorted(rows, key=lambda row: (-row["fp"], row["id"]))),
                  ("most-fn-pixels", sorted(rows, key=lambda row: (-row["fn"], row["id"]))))
        for reason, ordered in ranked:
            for row in ordered[:top_k]:
                selections.setdefault(row["id"], []).append(reason)
        by_id = {row["id"]: row for row in rows}
        indices = {record["id"]: index for index, record in enumerate(dataset.records)}
        selected = []
        for identifier, reasons in selections.items():
            row = by_id[identifier]
            with Image.open(stage / row["prediction"]) as raster:
                prediction = torch.frombuffer(bytearray(raster.tobytes()), dtype=torch.uint8).reshape(raster.height, raster.width).bool()
            row["selected_for"] = reasons
            row["views"] = _views(stage, identifier, dataset[indices[identifier]], prediction, torch)
            selected.append(row)
        report = {"ok": True, "purpose": "frozen-validation-error-analysis", "samples": len(rows), "test_used": False,
                  "checkpoint_sha256": digest, "checkpoint_epoch": document["epoch"], "threshold": config["threshold"],
                  "dataset_fingerprint": dataset.dataset_fingerprint, "val_selection_fingerprint": dataset.selection_fingerprint,
                  "device": selected_device, "package_versions": libraries, "same_device_and_libraries": comparable,
                  "matches_checkpoint_validation_counts": matches, "image_dimensions": list(dataset.dimensions),
                  "gsd_m": dataset.gsd_m, "summary": summary, "gallery_samples": len(selected), "top_k": top_k,
                  "source_metadata": dataset.metadata, "source_notices": notices,
                  "rankings": {reason: [row["id"] for row in ordered[:top_k]] for reason, ordered in ranked},
                  "limitations": ["Validation already selected this checkpoint; no independent test result.",
                                  "Differences from upstream labels are not automatically confirmed label errors.",
                                  "PV area bins describe total source-labelled PV per crop, not individual array sizes.",
                                  "Three-pixel boundary bands are diagnostics, not boundary-tolerant scores or new thresholds."]}
        _write_json(stage / "samples.json", rows)
        _write_json(stage / "analysis-report.json", report)
        (stage / "index.html").write_text(_gallery(selected, report), encoding="utf-8")
        report["samples_sha256"] = _sha256(stage / "samples.json")
        _write_json(stage / "analysis-report.json", report)
        _publish(stage, output)
        stage = None
        return report
    except ImportError as error:
        raise ValidationAnalysisError("Baseline dependencies missing. Run: uv sync --locked --extra baseline") from error
    except ValidationAnalysisError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ValidationAnalysisError(f"Validation analysis failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
