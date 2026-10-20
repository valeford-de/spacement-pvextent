# Frozen validation error analysis

The validation analyzer inspects a saved baseline without training, changing
thresholds or reading test imagery. Use a new output directory:

```sh
uv run --locked --extra baseline pvextent evaluate validation \
  data/pv4ger-seg-geographic-v1 runs/baseline-resnet18-001/best.pt \
  artifacts/baseline-resnet18-errors-001 --device mps --top-k 12 --json
```

Choose the device used by the original run for the closest reproduction.
`--device auto` is also supported. The model is restored offline from `best.pt`
after verifying its SHA-256 companion. Dataset and validation-selection
fingerprints must match the checkpoint. The recorded batch size, native pixels,
ImageNet normalization and threshold are retained. No optimizer is constructed.
The analyzer only constructs a `val` dataset view; even invalid or unavailable
test raster files are outside its pixel-access path.

When device and package versions match the checkpoint, the analysis must
reproduce its validation TP/FP/FN/TN exactly. A mismatch fails without publishing
a partial gallery. Other devices can produce slightly different predictions;
the report records both the execution context and whether counts match.

## What is measured

Every selected validation crop receives masked integer confusion counts and
pixel IoU, Dice, precision and recall. Global scores sum all pixel counts first.
Macro IoU gives each crop equal weight; median and minimum expose the score
distribution. Undefined IoU, where neither source nor prediction has PV, is
excluded from macro statistics and counted explicitly.

Additional strata distinguish:

- Total source-labelled PV area per crop: below 10 m², 10 to below 50 m², and
  at least 50 m². These are **crop-level area bins**, not individual array sizes.
- Crops with source PV touching their image boundary versus those without.
- Errors near source-label boundaries versus errors further from them.

The boundary diagnostic uses a fixed three-pixel Chebyshev neighbourhood. A
pixel is near a boundary when its 7 × 7 neighbourhood contains valid source PV
and valid background. At 0.1 m/pixel this radius is 0.3 m along the image axes.
Any neighbourhood containing ignored pixels is excluded from boundary/interior
classification; its errors are reported as unclassified. Space outside the
crop supplies no artificial PV, background or ignore labels.

Boundary errors, interior errors and unclassified errors sum to all FP plus
FN pixels. This is an error-location diagnostic. It does not forgive boundary
errors, modify ground truth, optimize a tolerance or replace the original IoU.

## Inspect the output

Open `OUTPUT/index.html` directly in a browser. It contains the union of the
`--top-k` lowest-IoU, most-FP-pixel and most-FN-pixel crops. Selection is
deterministic, with sample-ID tie breaking; the same example can belong to more
than one ranking. This gallery deliberately emphasizes errors and is not a
representative sample for estimating performance.

Each example shows RGB, source labels, prediction and error map. Green means
TP, red FP, blue FN, gray TN and orange ignore. Click a panel to inspect native
pixels. Original dataset metadata and available source notices accompany the
export. Source masks and annotation status are unchanged.

| Artifact | Contents |
| --- | --- |
| `analysis-report.json` | Aggregate metrics, strata, execution context, rankings and input hashes |
| `samples.json` | Measurements for every selected validation crop, plus selected display paths |
| `predictions/` | Native 0/1 prediction PNGs for all analyzed crops |
| `views/` | Diagnostic display PNGs for ranked examples |
| `index.html` | Offline review gallery |
| `dataset.toml`, `provenance/` | Original source metadata and available notices |

Existing outputs are never overwritten. Real imagery, labels, predictions and
per-sample measurements belong in the ignored `artifacts/` directory or outside
Git. Research notes may retain curated aggregate findings and hashes.

## Interpretation

Disagreement with a source mask is not automatically a model error or an
annotation error. Inspect the image before proposing corrections. In particular,
PV4GER array polygons sometimes bridge visible gaps between modules, while the
[project annotation guide](annotation-guide.md) specifies visible module
surfaces. Relabelling changes the target and requires a separately versioned
dataset; do not silently rewrite masks after seeing validation predictions.

This validation split already selected the checkpoint. Repeated inspection
supports development, not independent final performance claims. The frozen
test split remains reserved. New all-ignore roof candidates require a separate
[candidate prediction review](candidate-predictions.md); they cannot supply
accuracy or false-positive measurements until their annotations are reviewed.
