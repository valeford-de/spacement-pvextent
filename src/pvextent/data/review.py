# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Export a local, immutable annotation-review bundle without changing labels."""

from __future__ import annotations

import hashlib
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import tomllib
from typing import Any

from PIL import Image

from .pv4ger import DatasetImportError, _publish, _sha256, _write_json
from .validation import validate_dataset

POSITIVE_EXAMPLES = 12
SOURCE_NOTICES = ("upstream-LICENSE", "original-MIT-LICENSE", "upstream-README", "SOURCES.md")
NOTICE = ("Review decisions are evidence for future curation only. This bundle does not "
          "modify canonical masks, split assignments, or annotation status. A mask without "
          "PV labels does not establish that its image contains no visible PV.")


def _visualize(image_path: Path, mask_path: Path, label_path: Path, overlay_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        rgb = image.convert("RGBA")
        pv = mask.point([255 if value == 1 else 0 for value in range(256)])
        ignore = mask.point([255 if value == 255 else 0 for value in range(256)])
        labels = Image.new("RGB", mask.size, (0, 0, 0))
        labels.paste((0, 200, 150), mask=pv)
        labels.paste((255, 179, 71), mask=ignore)
        labels.save(label_path)
        tint = Image.new("RGBA", mask.size, (0, 0, 0, 0))
        tint.paste((0, 200, 150, 120), mask=pv)
        tint.paste((255, 179, 71, 140), mask=ignore)
        Image.alpha_composite(rgb, tint).convert("RGB").save(overlay_path)


def _html(document: dict[str, Any]) -> str:
    cards = []
    for index, sample in enumerate(document["samples"]):
        record = sample["source_record"]
        identifier = escape(record["id"])
        figures = "".join(
            f'<figure><a href="{sample[field]}" target="_blank" rel="noopener">'
            f'<img src="{sample[field]}" alt="{identifier}: {label}" loading="lazy"></a>'
            f'<figcaption>{label}</figcaption></figure>'
            for field, label in (("image", "RGB image"), ("labels", "Source labels"), ("overlay", "Overlay"))
        )
        negative = (
            '<button type="button" data-decision="no-visible-pv">'
            'No visible PV (requires full image review)</button>'
            if sample["no_pv_label_candidate"] else ""
        )
        cards.append(f'''<article data-index="{index}">
<h2>{identifier}</h2>
<p class="meta">Group: {escape(record['spatial_group'])}<br>
Source: {escape(record['source_id'])}; split: {escape(record['split'])};
annotation: {escape(record['annotation_status'])}; GSD: {record['gsd_m']} m/pixel.<br>
PV-labelled pixels: {sample['pv_pixels']}; ignore pixels: {sample['ignored_pixels']}.</p>
<div class="figures">{figures}</div>
<p>Click an image to inspect its native pixels. Green = PV (1); orange = ignore (255);
black in the label view = background (0). The original mask is
<a href="{sample['mask']}" target="_blank" rel="noopener">available unchanged</a>.</p>
<label><input type="checkbox" class="inspected"> I inspected the entire image at native resolution.</label>
<div class="decisions">
<button type="button" data-decision="needs-edit">Needs edit</button>
<button type="button" data-decision="exclude">Exclude</button>
<button type="button" data-decision="unsure">Unsure</button>
<button type="button" data-decision="accept-source-mask">Accept source mask</button>
{negative}<button type="button" data-decision="pending">Reset to pending</button>
</div><p>Decision: <strong class="status">pending</strong></p>
<label>Notes<textarea class="notes" rows="2"></textarea></label>
<label>Tags (comma separated)<input class="tags" type="text"></label>
</article>''')
    # The inline JSON is data, not markup: escape every HTML script terminator.
    embedded = json.dumps(document, sort_keys=True, allow_nan=False).replace("&", "\\u0026")
    embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e")
    embedded = embedded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Spacement PVExtent — local annotation review</title>
<style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f4f5f6;color:#15201e}
header,main,footer{max-width:1280px;margin:auto;padding:24px}header{padding-bottom:0}
h1{font-size:28px;margin:0 0 12px}h2{font-size:18px;overflow-wrap:anywhere}.meta{font-size:14px;overflow-wrap:anywhere}
article{background:white;border:1px solid #d8dedd;padding:20px;margin:24px 0;border-radius:6px}
.figures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
figure{margin:0}img{display:block;width:100%;height:auto;image-rendering:pixelated}figcaption{font-weight:600}
label{display:block;margin:12px 0}input[type=text],textarea{display:block;width:100%;padding:8px;font:inherit}
input[type=checkbox]{margin-right:8px}button{font:inherit;padding:8px 12px;margin:4px;border:1px solid #79978e;background:white;cursor:pointer}
button.active{background:#006e52;color:white}.decisions{margin-top:12px}a{color:#006e52}
.notice{border-left:4px solid #006e52;padding:12px;background:#e8f3ee}code{overflow-wrap:anywhere}
@media(max-width:720px){.figures{grid-template-columns:1fr}header,main,footer{padding:16px}}
</style></head><body><header>
<h1>Spacement PVExtent · annotation review</h1>
<p class="notice">''' + escape(NOTICE) + '''</p>
<p>Dataset fingerprint: <code>''' + escape(document["dataset_fingerprint"]) + '''</code><br>
All decisions start pending. No source files are edited. Decisions stay in this page's memory
until you download the JSON; refreshing the page resets them.</p>
<label>Reviewer name<input id="reviewer" type="text" autocomplete="name"></label>
<button id="export" type="button">Download review decisions (JSON)</button>
<p id="message" role="status"></p>
</header><main>''' + "\n".join(cards) + '''</main>
<footer><a href="review-manifest.json">Review manifest</a> ·
<a href="dataset.toml">Source dataset metadata</a></footer>
<script id="review-data" type="application/json">''' + embedded + '''</script>
<script>
"use strict";
const bundle = JSON.parse(document.getElementById("review-data").textContent);
const states = bundle.samples.map(sample => ({id:sample.id,decision:"pending",notes:"",tags:[],full_image_inspected:false}));
const cards = Array.from(document.querySelectorAll("article"));
cards.forEach((card,index) => {
  card.querySelectorAll("button[data-decision]").forEach(button => {
    button.addEventListener("click",() => {
      const decision = button.dataset.decision;
      if(decision === "no-visible-pv" && !card.querySelector(".inspected").checked){
        document.getElementById("message").textContent = "Inspect the entire image and check the review box before recording no visible PV.";
        return;
      }
      states[index].decision = decision;
      card.querySelector(".status").textContent = decision;
      card.querySelectorAll("button[data-decision]").forEach(candidate => candidate.classList.toggle("active",candidate === button && decision !== "pending"));
      document.getElementById("message").textContent = "Decisions are unsaved until downloaded. Canonical annotations are unchanged.";
    });
  });
  card.querySelector(".inspected").addEventListener("change",event => {
    if(!event.target.checked && states[index].decision === "no-visible-pv"){
      states[index].decision = "pending";
      card.querySelector(".status").textContent = "pending";
      card.querySelectorAll("button[data-decision]").forEach(button => button.classList.remove("active"));
    }
  });
});
document.getElementById("export").addEventListener("click",() => {
  const reviewer = document.getElementById("reviewer").value.trim();
  if(!reviewer){document.getElementById("message").textContent = "Enter a reviewer name to attribute the decisions.";return;}
  cards.forEach((card,index) => {
    states[index].notes = card.querySelector(".notes").value;
    states[index].tags = Array.from(new Set(card.querySelector(".tags").value.split(",").map(value => value.trim()).filter(Boolean)));
    states[index].full_image_inspected = card.querySelector(".inspected").checked;
  });
  const result = {review_schema_version:"1.0",dataset_fingerprint:bundle.dataset_fingerprint,
    review_manifest_sha256:bundle.review_manifest_sha256 || null,reviewer:reviewer,
    exported_at:new Date().toISOString(),notice:bundle.notice,decisions:states};
  const url = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+"\\n"],{type:"application/json"}));
  const link = document.createElement("a");link.href=url;link.download="pvextent-review-decisions.json";
  document.body.appendChild(link);link.click();link.remove();setTimeout(() => URL.revokeObjectURL(url),1000);
  document.getElementById("message").textContent = "Downloaded review evidence. Canonical annotations are unchanged.";
});
</script></body></html>
'''


def export_review_bundle(
    source_dir: Path | str, output_dir: Path | str, *, sample_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Copy review candidates into an offline gallery; never apply review decisions.

    Default selection includes all masks without label 1, prioritizing all-ignore
    masks, followed by twelve positive examples in sorted ID order.
    """
    source, output = Path(source_dir).resolve(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise DatasetImportError(f"Review output already exists: {output}")
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise DatasetImportError("Source dataset and review output must be separate directories.")
    staging = None
    try:
        validation = validate_dataset(source)
        if not validation.ok:
            first = validation.errors[0]
            raise DatasetImportError(f"Cannot review invalid dataset: {first.code}: {first.message}")
        manifest = source / "manifest.jsonl"
        metadata = source / "dataset.toml"
        manifest_bytes, metadata_bytes = manifest.read_bytes(), metadata.read_bytes()
        if (hashlib.sha256(manifest_bytes).hexdigest() != validation.stats["manifest_sha256"]
                or hashlib.sha256(metadata_bytes).hexdigest() != validation.stats["metadata_sha256"]):
            raise DatasetImportError("Source metadata changed during validation.")
        records = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
        by_id = {record["id"]: record for record in records}
        if sample_ids is not None:
            if (not isinstance(sample_ids, (list, tuple)) or not sample_ids
                    or any(not isinstance(value, str) for value in sample_ids)
                    or len(set(sample_ids)) != len(sample_ids)):
                raise DatasetImportError("sample_ids must be a nonempty list of unique sample identifiers.")
            unknown = set(sample_ids) - by_id.keys()
            if unknown:
                raise DatasetImportError(f"Unknown review sample identifiers: {', '.join(sorted(unknown))}")
        counts = {}
        for record in records:
            payload = (source / record["mask"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != record["mask_sha256"]:
                raise DatasetImportError(f"Source mask changed during review selection: {record['id']}")
            with Image.open(BytesIO(payload)) as mask:
                histogram = mask.histogram()
            counts[record["id"]] = {"pv_pixels": histogram[1], "ignored_pixels": histogram[255]}
        if sample_ids is None:
            candidates = [record for record in records if not counts[record["id"]]["pv_pixels"]]
            candidates.sort(key=lambda record: (
                counts[record["id"]]["ignored_pixels"] != record["width"] * record["height"], record["id"]))
            positives = sorted((record for record in records if counts[record["id"]]["pv_pixels"]),
                               key=lambda record: record["id"])[:POSITIVE_EXAMPLES]
            selected = candidates + positives
        else:
            selected = [by_id[identifier] for identifier in sample_ids]
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-review-", dir=output.parent))
        for folder in ("images", "masks", "labels", "overlays", "provenance"):
            (staging / folder).mkdir()
        (staging / "dataset.toml").write_bytes(metadata_bytes)
        notices = []
        for name in SOURCE_NOTICES:
            notice = source / "provenance" / name
            if notice.exists():
                if not notice.resolve().is_relative_to(source) or not notice.is_file() or notice.stat().st_size > 1024 * 1024:
                    raise DatasetImportError(f"Unsafe or oversized source notice: {notice}")
                shutil.copyfile(notice, staging / "provenance" / name)
                notices.append(f"provenance/{name}")
        samples = []
        for record in selected:
            identifier = record["id"]
            file_id = hashlib.sha256(identifier.encode()).hexdigest()[:24]
            image_path = f"images/{file_id}{Path(record['image']).suffix.lower()}"
            mask_path = f"masks/{file_id}.png"
            overlay_path, labels_path = f"overlays/{file_id}.png", f"labels/{file_id}.png"
            for field, relative in (("image", image_path), ("mask", mask_path)):
                shutil.copyfile(source / record[field], staging / relative)
                if _sha256(staging / relative) != record[f"{field}_sha256"]:
                    raise DatasetImportError(f"Source {field} changed during export: {identifier}")
            _visualize(staging / image_path, staging / mask_path, staging / labels_path, staging / overlay_path)
            samples.append({
                "id": identifier, "source_record": record, "image": image_path, "mask": mask_path,
                "labels": labels_path, "overlay": overlay_path, "overlay_sha256": _sha256(staging / overlay_path),
                "labels_sha256": _sha256(staging / labels_path), **counts[identifier],
                "no_pv_label_candidate": counts[identifier]["pv_pixels"] == 0,
                "decision": "pending", "notes": "", "tags": [],
            })
        document = {
            "review_schema_version": "1.0", "dataset_fingerprint": validation.stats["dataset_fingerprint"],
            "source_metadata": tomllib.loads((staging / "dataset.toml").read_text(encoding="utf-8")),
            "source_notices": notices, "notice": NOTICE,
            "selection": "all-no-pv-labels-plus-12-positive-examples" if sample_ids is None else "explicit-sample-ids",
            "samples": samples,
        }
        _write_json(staging / "review-manifest.json", document)
        manifest_hash = _sha256(staging / "review-manifest.json")
        (staging / "index.html").write_text(_html({**document, "review_manifest_sha256": manifest_hash}), encoding="utf-8")
        report = {
            "ok": True, "records": len(samples), "source_records": len(records),
            "no_pv_label_candidates": sum(sample["no_pv_label_candidate"] for sample in samples),
            "positive_examples": sum(not sample["no_pv_label_candidate"] for sample in samples),
            "dataset_fingerprint": validation.stats["dataset_fingerprint"],
            "review_manifest_sha256": manifest_hash, "selection": document["selection"], "output": str(output),
        }
        _publish(staging, output)
        staging = None
        return report
    except DatasetImportError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise DatasetImportError(f"Review export failed: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
