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
| 1 | `nrw-bulk-00904a25c74b2a0659499a7e` | 2024-06-25 | 48,271 |
| 2 | `nrw-bulk-01620679b5b34ee78aed4744` | 2024-08-12 | 22,838 |

- Imagery: Geobasis NRW/openNRW, fetched from the [NRW orthophoto service](https://www.wms.nrw.de/geobasis/wms_nw_dop), under [Datenlizenz Deutschland Zero 2.0](https://www.govdata.de/dl-de/zero-2-0).
- Labels: experimental Valeford AI-reviewed partial annotations, Apache-2.0. Green means approved PV, blue means approved background; untinted pixels are unknown.
- Predictions: orange means model-predicted PV at threshold `0.5`. Prediction counts are not accuracy scores. No mask was expanded from predictions.
- Checkpoint: `baseline-resnet18-bulk-v4-001/best.pt`, epoch 3, documented in the [V4 training audit](../research/nrw-bulk-training-audit.md).
- Checkpoint SHA-256: `32ca5148f0e18879c14fcc7b0b0e3e0ba09e272f25cf2cf6a45b8a9fd793f8bd`.
- Rendering: CPU inference, native float32 RGB `[0,1]`, ImageNet mean/std normalization, unchanged weights, no augmentation or training.

These predictions demonstrate actual output from the V4-trained model.
No labels or dataset assignments changed.

## Training history

`training-history.svg` plots actual recorded epoch losses and internal validation
IoU comparing the baseline reference with the new V4 bulk training run.
V4 incorporates the 482 AI-reviewed NRW partial masks across 3,775 total training crops
and 278 validation crops. Straight lines connect measured epochs; no intermediate
measurements are invented.

The [renderer](../../examples/render_training_history.py) reads only these
versioned scalar reports. Rebuild from the repository root:

```sh
uv run --no-project --with matplotlib==3.11.2 python examples/render_training_history.py
```

The graph is an original Valeford figure under Apache-2.0. It documents
both the V4 target-source training trajectory and the earlier baseline control.

## Brand assets

`pvextent-banner.png` and `valeford-logo.png` are the supplied Spacement and
Valeford brand assets.
