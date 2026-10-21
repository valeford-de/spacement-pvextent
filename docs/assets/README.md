# README image sources

The README contains two derived documentation figures. They are illustrations,
not a dataset release or an additional model evaluation. Raw crops, masks,
prediction PNGs and checkpoints remain outside Git.

## Segmentation preview

`segmentation-preview.png` shows two real NRW orthophotos, their final partial
labels and actual output from the existing baseline. Each panel keeps the
native 320 × 320 grid at 0.1 m/pixel; there is no image resampling. Colour tints
are display overlays, not additional annotations.

The examples are the first two nonexcluded positive IDs, sorted
lexicographically in the [final V4 collection](../research/nrw-bulk-mask-curation-audit.md).
Selection was fixed before either prediction was computed. This is an
illustrative selection, not a representative benchmark. Known TRAIN-location
recaptures may be correlated with the earlier training data.

| Row | Sample ID | Capture date | Predicted PV pixels |
| --- | --- | --- | ---: |
| 1 | `nrw-bulk-00904a25c74b2a0659499a7e` | 2024-06-25 | 45,979 |
| 2 | `nrw-bulk-01620679b5b34ee78aed4744` | 2024-08-12 | 22,101 |

- Imagery: Geobasis NRW/openNRW, fetched from the [NRW orthophoto service](https://www.wms.nrw.de/geobasis/wms_nw_dop), under [Datenlizenz Deutschland Zero 2.0](https://www.govdata.de/dl-de/zero-2-0).
- Labels: experimental Valeford AI-reviewed partial annotations, Apache-2.0. Green means approved PV, blue means approved background; untinted pixels are unknown.
- Predictions: orange means model-predicted PV at threshold `0.5`. Prediction counts are not accuracy scores. No mask was expanded from predictions.
- Checkpoint: `baseline-resnet18-continued-001/best.pt`, epoch 5, documented in the [training report](../research/training-extension-audit.md).
- Checkpoint SHA-256: `2e04c0e009f3b76a1a0c233aa9efea064251f6aed572d8db9272c2455c83edc6`.
- Rendering: CPU inference, native float32 RGB `[0,1]`, ImageNet mean/std normalization, unchanged weights, no augmentation or training.

These predictions were made only after mask curation, for this documentation.
No labels, dataset assignments or existing model scores changed. Original base
and external pilot rasters were blocked by path and inode during rendering;
none was opened. The complete local rendering recipe and hash report are
retained in `artifacts/render-readme-previews-v1.py` and
`artifacts/readme-previews-v1/report.json`.

## Training history

`training-history.svg` plots actual recorded epoch losses and internal validation
IoU. Epochs 1–3 come from
[`baseline-negative-comparison-v1.json`](../research/baseline-negative-comparison-v1.json);
epochs 4–6 come from [`training-extension-v1.json`](../research/training-extension-v1.json).
They share the same 3,293 training and 278 validation records. Straight lines
connect measured epochs; no intermediate measurements are invented.

The [renderer](../../examples/render_training_history.py) reads only these
versioned scalar reports. Rebuild from the repository root in an isolated
environment:

```sh
uv run --no-project --with matplotlib==3.11.2 python examples/render_training_history.py
```

The graph is an original Valeford figure under Apache-2.0. It describes the
existing six-epoch run, which did not use the 482 new partial masks.

## Brand assets

`pvextent-banner.png` and `valeford-logo.png` are the supplied Spacement and
Valeford brand assets.
