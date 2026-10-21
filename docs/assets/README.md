# README image sources

The README contains two derived documentation figures. They are illustrations,
not a dataset release or an additional model evaluation. Raw crops, masks,
prediction PNGs and checkpoints remain outside Git.

## Segmentation preview

`segmentation-preview.png` shows two real aerial orthophotos from the geographically
held-out validation partition, their complete ground-truth annotations, and actual
predictions from the V4-trained model. Each panel retains the native 320 × 320 grid
at 0.1 m/pixel; there is no image resampling.

Both examples are 100% supervised (zero unlabelled or ignored pixels):

| Row | Sample ID | Description | Ground Truth PV | Predicted PV | Sample IoU | Sample Dice |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `pv4ger-96e0ab3402e42dd6c89225c4` | Residential Rooftop PV | 12,604 px | 12,527 px | **98.74%** | **99.37%** |
| 2 | `pv4ger-c55923b625fc5b0b224b3c3d` | Commercial Rooftop PV | 48,342 px | 48,562 px | **98.82%** | **99.41%** |

- Imagery: GeoBasis NRW/openNRW under [Datenlizenz Deutschland Zero 2.0](https://www.govdata.de/dl-de/zero-2-0).
- Labels: Complete reviewed ground truth. Green means confirmed PV array, blue means confirmed background roof/surrounding.
- Predictions: Orange means model-predicted PV at threshold `0.5`.
- Checkpoint: `baseline-resnet18-bulk-v4-001/best.pt`, epoch 3, documented in the [V4 training audit](../research/nrw-bulk-training-audit.md).
- Checkpoint SHA-256: `32ca5148f0e18879c14fcc7b0b0e3e0ba09e272f25cf2cf6a45b8a9fd793f8bd`.

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
