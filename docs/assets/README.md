# README image sources

All README figures show real output of the V5 (Vidda) checkpoint and the `pvextent`
CLI. Nothing is drawn or edited by hand. Raw crops, masks, predictions and
checkpoints remain outside Git.

Rebuild every figure from the repository root:

```sh
uv run --locked --no-sync python examples/render_readme_figures.py
```

The script needs the local checkpoint
`runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt`, the datasets
`data/pvextent-training-v4` and `data/positive-holdout-frozen-v1`, and network access
to the public Geobasis NRW WMS.

## Figures

| File | Content | Input |
| --- | --- | --- |
| `segmentation-preview.png` | Orthophoto, ground truth and V5 prediction at threshold 0.5 | Sealed PV4GER test crops at the 90th, 50th and 10th per-crop IoU percentile of 418 crops with at least 500 PV pixels |
| `nrw-holdout.jpg` | Same layout, recall on reviewed PV pixels | Every reviewed PV crop of the spatially separate NRW positive holdout |
| `cad-overlay.png` | `pvextent vectorize --profile vidda`, GeoJSON redrawn at 2x with the library renderer | The three percentile test crops |
| `temporal-change.jpg` | `pvextent temporal --checkpoint <V5> --threshold 0.5` | 64 m window around `nrw-DENW07AL0000ytXZBL`: `nw_hist_dop_2015` vs. `nw_dop_rgb` (2025 flight) |
| `spectral.jpg` | `pvextent spectral --nir`: CIR composite and NDVI | Same window, `nw_dop_rgb` and `nw_dop_nir` |

Selected crops on the sealed test set:

| Percentile | Sample ID | IoU |
| --- | --- | ---: |
| 90th | `pv4ger-c4cdf711ece690c241b09ca3` | 97.4% |
| 50th | `pv4ger-743ac1439eb370f97a8915b4` | 91.3% |
| 10th | `pv4ger-c9c1365a90396b67acb3afb3` | 64.6% |

The WMS layers are 8-bit display products, so the NDVI is relative, not calibrated
reflectance. WMS responses are mutable; a later rebuild may differ.

- Imagery: GeoBasis NRW/openNRW, including the PV4GER crops derived from it, under [Datenlizenz Deutschland Zero 2.0](https://www.govdata.de/dl-de/zero-2-0).
- Checkpoint: `deeplabv3plus-resnet50-v5-photometric-001/best.pt`, SHA-256 `9380f5b7ebf5af23aeb445832402714865e387db5ce496d070575aada6e351a6`.

## Other assets

`training-history.svg` plots recorded epoch losses and internal validation IoU of the
V4 run and the baseline control, rendered by
[`render_training_history.py`](../../examples/render_training_history.py).
`pvextent-banner.png`, `pvextent-vidda-banner.png`, `pvextent-tinden-banner.png` and
`valeford-logo.png` are Spacement and Valeford brand assets.
