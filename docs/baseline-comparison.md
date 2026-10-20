# Baseline training comparison

This experiment compares the original U-Net/ResNet18 baseline with a fresh run
using the augmented training snapshot. Architecture, initialization protocol,
hyperparameters and validation partition stay fixed. The change is a small
addition of explicitly authorized AI-reviewed background supervision, described
in the [negative curation guide](negative-curation.md).

This guide defines reproduction and interpretation. Completed measurements and
artifact hashes belong in the
[comparison audit](research/baseline-negative-comparison-audit.md); the new
dataset alone does not establish a model improvement.

## Freeze the inputs

| Input | Reference | Augmented run |
| --- | --- | --- |
| Dataset | `data/pv4ger-seg-geographic-v1` | `data/pvextent-training-v2` |
| Run directory | `runs/baseline-resnet18-001` | `runs/baseline-resnet18-negatives-001` |
| Training records | 3,282 | 3,293 |
| Validation records | 278 | 278, unchanged |
| Reserved test records | 421 | 421, unchanged and unopened |
| Unassigned records | 46 | 42 |
| Epochs | 3 | 3 |
| Optimizer updates | 4,923 | 4,941 |

Both datasets retain native **320 × 320 RGB crops at 0.1 m/pixel**. The
augmentation preserves all validation/test record declarations and inherited
pixels, with no holdout raster decoding during the merge. Labels `0` and `1`
supply background and PV supervision; `255` contributes neither loss nor pixel
confusion counts. Existing upstream annotation provenance remains unchanged.

The added eleven partial negative records are **0.334% of training records**.
They supply **217,276 background pixels**, or **0.06461% of the 336,294,076
supervised training pixels per augmented epoch**. Their remaining pixels are
ignored. This pixel share is not their exact contribution to optimization:
masked BCE plus soft Dice is computed and optimized per batch, and batch Dice
is nonlinear. See the [baseline loss definition](baseline-training.md#training-and-validation-protocol).

Record the exact dataset fingerprints, configuration, encoder artifact hashes,
package versions and resolved device from each run's `config.json` and
`baseline-report.json`. Retain the original snapshots and run artifacts.
Inherited image/mask files are hard links and must remain immutable.

The [frozen comparison protocol](research/baseline-negative-comparison-protocol.json)
was recorded before the augmented run's first epoch result. Its preflight
confirmed matching configurations apart from the dataset and training-selection
fingerprints, and an exactly matching complete initial-validation record. This
checks the starting measurement without promising bit-identical MPS training.

## Use the same training protocol

Both runs use the versioned
[baseline configuration](../configs/baseline-resnet18.toml):

- `unet-resnet18-v1`, with the same pinned ImageNet ResNet18 encoder and seeded
  random decoder initialization.
- Seed `20261020`, three epochs, batch size two and AdamW learning rate and
  weight decay both `0.0001`; gradient norm clipping at `1.0`.
- Native pixels, the same ImageNet channel normalization, and paired horizontal
  and vertical flips with probability `0.5` each; no resizing.
- Complete declared training and validation splits, with no sample limits.
- Validation threshold `0.5` and best completed epoch selected by global
  validation pixel IoU; ties retain the earlier best epoch.

Each run starts with a fresh model and optimizer. The augmented run does not
resume or fine-tune the already trained reference checkpoint. Both initialize
from the same cached encoder artifact; model construction and training stay
offline once the [verified cache](pretrained-weights.md) is present.

The training loop samples every training record **once per epoch**, in a seeded
shuffle without replacement. There is no negative oversampling, replacement
sampling, class reweighting or extra negative-only training stage in this
baseline comparison. The separate tiny negative-only smoke check verified the
pipeline and is not a pretrained baseline result.

After preparing the cache, the augmented run uses:

```sh
uv sync --locked --extra baseline
uv run --locked --extra baseline pvextent train baseline \
  data/pvextent-training-v2 runs/baseline-resnet18-negatives-001 \
  --config configs/baseline-resnet18.toml \
  --weights-dir data/pretrained/resnet18.a1_in1k \
  --device mps --json
```

The reference used the same configuration on
`data/pv4ger-seg-geographic-v1`. The paths above identify the local experiment;
if a successful run already exists, retain it and choose a fresh output path
for reproduction. Neither input snapshots nor completed run directories are
overwritten. Use the original MPS device and locked package versions for the
closest reproduction; MPS execution is not guaranteed to be bit-identical.

## Analyze each best checkpoint against its own snapshot

The validation analyzer checks both the dataset and validation-selection
fingerprints. Although the validation records are unchanged, the full dataset
fingerprints differ. Analyze each checkpoint against the snapshot it used:

```sh
uv run --locked --extra baseline pvextent evaluate validation \
  data/pv4ger-seg-geographic-v1 runs/baseline-resnet18-001/best.pt \
  artifacts/baseline-resnet18-errors-001 --device mps --top-k 12 --json

uv run --locked --extra baseline pvextent evaluate validation \
  data/pvextent-training-v2 runs/baseline-resnet18-negatives-001/best.pt \
  artifacts/baseline-resnet18-negatives-errors-001 \
  --device mps --top-k 12 --json
```

The original analysis already exists locally. Open its reports or gallery
directly instead of rerunning into the same path. New analyses require new
output directories and a successfully completed checkpoint with its SHA-256
companion. The analyzer restores each model offline, retains its threshold and
batch size, and opens only validation pixels. In the original execution context,
its integer TP/FP/FN/TN must reproduce the checkpoint's recorded counts.

Each analysis contains `analysis-report.json`, `samples.json`, prediction PNGs
and an offline error gallery. See the
[validation error guide](validation-errors.md) for strata, boundary diagnostics
and the distinction between global pixel scores and macro crop scores.

## Compare paired validation reports

After both analyses complete successfully, compare their saved reports:

```sh
uv run --locked pvextent evaluate compare-validations \
  artifacts/baseline-resnet18-errors-001 \
  artifacts/baseline-resnet18-negatives-errors-001 \
  artifacts/baseline-resnet18-negative-comparison-001 --json
```

The output directory must be new. The comparison consumes `analysis-report.json`
and `samples.json` from each analysis without reopening image, mask or
prediction rasters. It runs no model and does not touch test pixels.

Both analyses must report success and reproduce their checkpoint confusion
counts. The comparison requires matching validation-selection fingerprints,
thresholds, native dimensions, GSD, device and package versions. It verifies
each `samples.json` against its recorded SHA-256 and checks paired sample
identifiers and ground-truth pixel totals. The matching validation-selection
fingerprint identifies the same original records and declared image/mask hashes.
The full dataset and checkpoint hashes can differ: those differences identify
the two experiments. These checks do not prove matching training protocols;
compare the run configurations and frozen protocol separately.

| Output | Contents |
| --- | --- |
| `comparison-report.json` | Aggregate differences, execution context, input hashes and paired IoU summary |
| `paired-samples.json` | Per-crop reference/candidate measurements and changes, paired by identifier |

Deltas use **candidate minus reference**. Integer pixel-count differences,
ratio differences and percentage-point changes are reported separately; a
change in percentage points is not a relative percentage improvement. Paired
IoU summaries distinguish improved, worse, tied and undefined in either
analysis. Undefined cases do not silently become zero-IoU examples. Global
scores still aggregate pixel confusion counts; they are not averages of the
paired crop-score changes.

## Interpret the difference

The protocol holds the major settings constant, but adding eleven examples
also changes the shuffle, batch membership and sequence of random flips. With
an odd augmented training count, its last batch contains one crop. It performs
18 additional optimizer updates over three epochs. The same seed therefore
does not give each original crop the same optimization history, and MPS can
introduce further numerical variation. A score difference cannot be attributed
solely to the semantic value of the new negative annotations.

Compare global confusion counts and pixel metrics alongside paired crop
changes. Fewer false-positive pixels can accompany more missed PV pixels;
inspect precision and recall together. A selected error gallery prioritizes
investigation and is not a representative evaluation sample. Disagreement with
upstream labels still needs annotation adjudication.

The paired analysis uses each run's selected `best.pt`. Also retain the fixed
final-epoch measurements from both run reports and histories: comparing epoch
3 against epoch 3 answers a different question from comparing independently
selected best epochs. Neither comparison changes the fixed threshold.

These 278 validation crops already select checkpoints and support development.
Repeated comparison does not make them an independent benchmark. Their pixel
false positives do not establish a roof-level false-positive rate on unseen
negative roofs or real-world prevalence. The eleven AI-reviewed partial masks
are experimental training labels, with no claim of human-reviewed benchmark
truth. All 421 test examples remain reserved for a separately fixed evaluation
protocol; no new model release or independent performance claim follows from
this comparison.
