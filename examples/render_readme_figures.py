# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Render the README figures from real model output.

Every figure shows what the V5 (Vidda) checkpoint and the ``pvextent`` CLI actually
produce; nothing is drawn by hand. Requires the local, untracked inputs
``runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt``,
``data/pvextent-training-v4`` and ``data/positive-holdout-frozen-v1``, plus network
access to the public Geobasis NRW WMS. Run from the repository root:

    uv run --locked --no-sync python examples/render_readme_figures.py
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pvextent.cad.vectorize import _label_font, render_cad_overlay
from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import normalize_rgb

CHECKPOINT = Path("runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt")
TEST_SET = Path("data/pvextent-training-v4")
NRW_HOLDOUT = Path("data/positive-holdout-frozen-v1")
ASSETS = Path("docs/assets")
WMS_CURRENT = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
WMS_HISTORIC = "https://www.wms.nrw.de/geobasis/wms_nw_hist_dop"
CHANGE_SITE = "nrw-DENW07AL0000ytXZBL"  # Holdout building with PV added between 2015 and 2025
CHANGE_YEAR = 2015

PV = np.array([255, 190, 40]); FALSE_POS = np.array([230, 40, 40]); MISSED = np.array([40, 110, 255])
TRUTH = np.array([40, 200, 120])


def load_model(device: str) -> torch.nn.Module:
    model = build_deeplabv3plus_resnet50(None)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"])
    return model.to(device).eval()


def predict(model: torch.nn.Module, rgb: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        return torch.sigmoid(model(normalize_rgb(tensor)))[0, 0].cpu().numpy()


def tint(rgb: np.ndarray, selection: np.ndarray, color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[selection] = (1 - alpha) * out[selection] + alpha * color
    return out


def error_map(rgb: np.ndarray, truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    out = tint(rgb, truth & predicted, PV)
    out = tint(out, predicted & ~truth, FALSE_POS, 0.65)
    return tint(out, truth & ~predicted, MISSED, 0.65)


def grid_figure(rows: list[tuple[str, list[np.ndarray]]], headers: list[str], notes: list[str], path: Path) -> None:
    """Lay out 320 px panels at 2x nearest-neighbour scale with a header row and notes."""
    size, gap, head, label = 640, 16, 64, 30
    font, bold = _label_font(22), _label_font(26)
    width = len(headers) * size + (len(headers) + 1) * gap
    height = head + len(rows) * (size + label + gap) + 30 * len(notes) + 16
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, text in enumerate(headers):
        draw.text((gap + column * (size + gap), 22), text, fill=(20, 20, 20), font=bold)
    y = head
    for caption, panels in rows:
        draw.text((gap, y), caption, fill=(70, 70, 70), font=font)
        for column, panel in enumerate(panels):
            image = Image.fromarray(panel.clip(0, 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
            canvas.paste(image, (gap + column * (size + gap), y + label))
        y += size + label + gap
    for note in notes:
        draw.text((gap, y + 4), note, fill=(70, 70, 70), font=font)
        y += 30
    canvas.save(path, quality=90, optimize=True)


def percentile_test_crops(model: torch.nn.Module, device: str) -> list[tuple[str, dict, np.ndarray]]:
    """Score every sealed test crop and return the crops at the 90th, 50th and 10th IoU percentile."""
    records = [json.loads(line) for line in open(TEST_SET / "manifest.jsonl", encoding="utf-8")]
    scored = []
    for record in (r for r in records if r.get("split") == "test"):
        rgb = np.asarray(Image.open(TEST_SET / record["image"]).convert("RGB"))
        mask = np.asarray(Image.open(TEST_SET / record["mask"]))
        truth = mask == 1
        if truth.sum() < 500:
            continue
        predicted = (predict(model, rgb, device) >= 0.5) & (mask != 255)
        tp, fp, fn = (truth & predicted).sum(), (predicted & ~truth).sum(), (truth & ~predicted).sum()
        scored.append((tp / (tp + fp + fn), record, predicted))
    ious = np.array([item[0] for item in scored])
    picks = []
    for name, q in (("90th percentile", 90), ("Median", 50), ("10th percentile", 10)):
        iou, record, predicted = scored[int(np.argmin(np.abs(ious - np.percentile(ious, q))))]
        picks.append((f"{name}  ·  IoU {iou * 100:.1f}%  ·  {record['id']}", record, predicted))
    print(f"Scored {len(scored)} test crops; median per-crop IoU {np.median(ious):.3f}")
    return picks


def render_segmentation(picks: list) -> None:
    rows = []
    for caption, record, predicted in picks:
        rgb = np.asarray(Image.open(TEST_SET / record["image"]).convert("RGB"))
        truth = np.asarray(Image.open(TEST_SET / record["mask"])) == 1
        rows.append((caption, [rgb, tint(rgb, truth, TRUTH), error_map(rgb, truth, predicted)]))
    grid_figure(rows, ["Orthophoto", "Ground truth", "V5 (Vidda) prediction"],
                ["Prediction at threshold 0.5: yellow = correct PV, red = false positive, blue = missed PV.",
                 "Sealed PV4GER test crops at 10 cm/px, picked at fixed per-crop IoU percentiles, not by hand."],
                ASSETS / "segmentation-preview.png")


def render_nrw_holdout(model: torch.nn.Module, device: str) -> None:
    """Show every independently reviewed PV crop of the separate NRW holdout."""
    rows = []
    for line in open(NRW_HOLDOUT / "manifest.jsonl", encoding="utf-8"):
        record = json.loads(line)
        mask = np.asarray(Image.open(NRW_HOLDOUT / record["mask"]))
        truth = mask == 1
        if record["annotation_status"] != "reviewed" or not truth.any():
            continue
        rgb = np.asarray(Image.open(NRW_HOLDOUT / record["image"]).convert("RGB"))
        predicted = (predict(model, rgb, device) >= 0.5) & (mask != 255)
        recall = (truth & predicted).sum() / truth.sum()
        rows.append((f"{record['id']}  ·  recall on reviewed PV {recall * 100:.0f}%",
                     [rgb, tint(rgb, truth, TRUTH), error_map(rgb, truth, predicted)]))
    grid_figure(rows, ["NRW orthophoto", "Reviewed PV (partial)", "V5 (Vidda) prediction"],
                ["Prediction at threshold 0.5: yellow = correct PV, red = false positive, blue = missed PV.",
                 "All reviewed PV crops of the spatially separate Geobasis NRW holdout (2025/2026 flights)."],
                ASSETS / "nrw-holdout.jpg")


def render_vectorization(picks: list, work: Path) -> None:
    """Run ``pvextent vectorize --profile vidda`` and redraw its GeoJSON at 2x for legibility."""
    panels = []
    for _, record, _ in picks:
        out = work / record["id"]
        subprocess.run([sys.executable, "-m", "pvextent", "vectorize", str(TEST_SET / record["image"]), str(out),
                        "--profile", "vidda"], check=True, capture_output=True)
        geojson = json.loads((out / "vectorized.geojson").read_text(encoding="utf-8"))
        for feature in geojson["features"]:
            props = feature["properties"]
            props["pixel_polygon"] = [[2 * x, 2 * y] for x, y in props["pixel_polygon"]]
            props["fitted_modules_px"] = [[[2 * x, 2 * y] for x, y in m] for m in props.get("fitted_modules_px", [])]
            props["centroid_pixel"] = [2 * v for v in props["centroid_pixel"]]
        image = Image.open(TEST_SET / record["image"]).convert("RGB").resize((640, 640), Image.NEAREST)
        render_cad_overlay(image, geojson, out / "overlay_2x.png")
        panels.append(Image.open(out / "overlay_2x.png"))
    canvas = Image.new("RGB", (len(panels) * 656 + 16, 672), "white")
    for column, panel in enumerate(panels):
        canvas.paste(panel, (16 + column * 656, 16))
    canvas.save(ASSETS / "cad-overlay.png", optimize=True)


def fetch_wms(url: str, layer: str, bbox: tuple[float, ...]) -> Image.Image:
    query = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "CRS": "EPSG:25832", "LAYERS": layer,
             "STYLES": "", "BBOX": ",".join(f"{v:.1f}" for v in bbox), "WIDTH": 640, "HEIGHT": 640,
             "FORMAT": "image/png", "TRANSPARENT": "FALSE"}
    with urllib.request.urlopen(url + "?" + urlencode(query), timeout=60) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def render_change_and_spectral(work: Path) -> None:
    """Run ``pvextent temporal`` and ``pvextent spectral`` on a real 64 m NRW window."""
    record = next(json.loads(line) for line in open(NRW_HOLDOUT / "manifest.jsonl", encoding="utf-8")
                  if CHANGE_SITE in line)
    transform = record["georeference"]["transform"]
    cx, cy = transform[2] + 16.0, transform[5] - 16.0
    bbox = (cx - 32, cy - 32, cx + 32, cy + 32)
    paths = {}
    for name, url, layer in (("t1", WMS_HISTORIC, f"nw_hist_dop_{CHANGE_YEAR}"), ("rgb", WMS_CURRENT, "nw_dop_rgb"),
                             ("nir", WMS_CURRENT, "nw_dop_nir")):
        paths[name] = work / f"{name}.png"
        fetch_wms(url, layer, bbox).save(paths[name])

    subprocess.run([sys.executable, "-m", "pvextent", "temporal", str(paths["t1"]), str(paths["rgb"]),
                    str(work / "temporal"), "--date-t1", str(CHANGE_YEAR), "--date-t2", record["capture_date"][:4],
                    "--checkpoint", str(CHECKPOINT), "--threshold", "0.5"], check=True, capture_output=True)
    Image.open(work / "temporal" / "temporal_cad_overlay.png").convert("RGB").save(ASSETS / "temporal-change.jpg", quality=90)

    subprocess.run([sys.executable, "-m", "pvextent", "spectral", str(paths["rgb"]), str(work / "spectral"),
                    "--nir", str(paths["nir"])], check=True, capture_output=True)
    cir = Image.open(work / "spectral" / "cir_composite.png").convert("RGB")
    ndvi = Image.open(work / "spectral" / "ndvi.png").convert("RGB")
    canvas = Image.new("RGB", (2 * 640 + 3 * 16, 640 + 32), "white")
    canvas.paste(cir, (16, 16))
    canvas.paste(ndvi, (640 + 32, 16))
    canvas.save(ASSETS / "spectral.jpg", quality=90)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    picks = percentile_test_crops(model, device)
    render_segmentation(picks)
    render_nrw_holdout(model, device)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        render_vectorization(picks, work)
        render_change_and_spectral(work)
    print("Figures written to", ASSETS)


if __name__ == "__main__":
    main()
