![Spacement PVExtent](docs/assets/pvextent-banner.png)

![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)

Photovoltaic array segmentation from orthophotos.

Developed and maintained by **Valeford** under the **Spacement** brand.

PVExtent is a standalone research project for training and evaluating models
that identify the visible extent of photovoltaic arrays in aerial imagery.
Its modules separate data preparation, model construction, training,
evaluation, and inference.

## Status

Dataset import, validation, geographic split snapshots, a local label-review
gallery, and a bounded NRW roof candidate fetcher are implemented. The complete
PV4GER-SEG release has been imported and split with a 1 km geographic buffer.
The training smoke workflow verifies gradients, optimizer updates, checkpoint
reload and local continuation. The configured U-Net/ResNet18 baseline now trains
on the full training split, uses validation for checkpoint selection, and records
global pixel metrics and resumable state. The first full three-epoch run is
complete: its best checkpoint reached pixel IoU **0.91572** and Dice
**0.95600** on the internal validation split. The original 421 PV4GER test images
remain unopened, and model weights have not been released. See the
[run audit](docs/research/baseline-resnet18-audit.md) for the recorded evidence.
Frozen error analysis has reproduced the original counts on all 278 validation
crops and exported an offline gallery of difficult examples. Label adjudication,
roof-level false-positive measurement, an independent benchmark and general
inference remain outstanding. Explicitly authorized two-pass AI curation has
created 11 partial background masks for experimental training, with four
ambiguous crops left ignored. A new training snapshot includes these negatives
while preserving the original validation/test records. A fresh three-epoch
candidate run using the same protocol is complete: its best checkpoint reached
validation IoU **0.91903** and Dice **0.95781**, with higher precision and lower
recall than the original baseline. The original artifacts and scores remain
preserved; this is a development comparison, with no isolated causal claim for
the eleven masks. See the
[comparison audit](docs/research/baseline-negative-comparison-audit.md) and
[research roadmap](docs/roadmap.md) for evidence and next steps.
A spatially separate exploratory negative pilot has also evaluated both fixed
models on seven AI-reviewed partial masks: 31 versus zero false-positive pixels
within 150,374 approved background pixels. Ignored areas remain unscored. Its
small sample and amended coverage protocol do not establish a roof-level rate,
recall or general model superiority. See the
[holdout audit](docs/research/negative-holdout-audit.md).
A subsequent blind positive pilot exposes poor transfer: on three newly
AI-reviewed crops with 3,577 confirmed PV pixels, the reference has **0%** recall
and the candidate only **4.05%**. A known-validation control exactly reproduces
both models' earlier predictions; no input-path defect was found. These partial
labels do not establish whole-array recall, but the observed misses prevent a
general operational detection claim. See the
[positive audit](docs/research/positive-holdout-audit.md). The next priority is
separate NRW training positives and investigation of source differences, with
the pilot retained outside training and tuning.

## Documentation

The [project wiki](https://github.com/valeford-de/spacement-pvextent/wiki)
provides an introduction and getting-started documentation. The versioned
[research roadmap](docs/roadmap.md) remains in this repository.

Start with the [dataset format](docs/data-format.md) and
[annotation guide](docs/annotation-guide.md) when preparing imagery and masks.
The [PV4GER-SEG import guide](docs/pv4ger-import.md) explains how to obtain and
convert the first public segmentation dataset.
Continue with [geographic splits](docs/geographic-splits.md),
[local label review](docs/label-review.md), and
[unreviewed NRW roof candidates](docs/nrw-candidates.md).
The [training smoke guide](docs/training-smoke.md) documents the initial pipeline
check and its [measured audit](docs/research/training-smoke-audit.md).
Continue with the [pretrained baseline guide](docs/baseline-training.md),
[encoder provenance and licenses](docs/pretrained-weights.md), and
[baseline run audit](docs/research/baseline-resnet18-audit.md).
The [validation error guide](docs/validation-errors.md) and
[analysis audit](docs/research/validation-error-audit.md) document the frozen
checkpoint review. [Candidate prediction guidance](docs/candidate-predictions.md)
covers visual review of unannotated roofs without ground-truth scores.
The [negative curation guide](docs/negative-curation.md) and
[curation audit](docs/research/negative-curation-audit.md) describe the separate
AI review format, partial masks and immutable training augmentation.
The [baseline comparison guide](docs/baseline-comparison.md) and
[comparison audit](docs/research/baseline-negative-comparison-audit.md) describe
the fresh run on the augmented snapshot and its validation trade-off.
The [negative holdout guide](docs/negative-holdout.md) and
[pilot audit](docs/research/negative-holdout-audit.md) document geographically
separate acquisition, blind AI review, the V1-to-V2 amendment and pixel diagnostics.
The [positive holdout guide](docs/positive-holdout.md) documents the separate
blind positive reviews, consensus masks and recall on confirmed PV pixels;
its [audit](docs/research/positive-holdout-audit.md) records the observed failure
and validation control.

For private vulnerability reports, see the [security policy](.github/SECURITY.md).

## Development setup

Requires Python 3.11 or newer; development uses Python 3.12. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run --locked pvextent --help
uv run --locked python -m unittest discover -s tests -v
```

Run the complete suite, including dataset import, geographic processing,
training and pretrained-baseline tests, with all optional dependencies:

```sh
uv run --locked --extra import --extra geo --extra train --extra baseline \
  python -m unittest discover -s tests -v
```

## Validate a dataset

```sh
uv run --locked pvextent data validate /path/to/dataset --json
```

The validator checks metadata, file hashes, image/mask alignment, mask classes,
and declared split boundaries. A passing report does not certify annotation
truth, usage rights, or geographic independence.

To try the format using generated synthetic pixels, without downloading imagery:

```sh
uv run --locked python examples/create_demo_dataset.py data/demo
uv run --locked pvextent data validate data/demo --json
```

The demo is a structural example with unreviewed labels and unknown capture
dates/georeferences, so its valid report includes warnings. It is not a training
dataset.

## Import PV4GER-SEG

```sh
uv sync --locked --extra import
uv run --locked --extra import pvextent data import-pv4ger-seg \
  data/raw/pv4ger-seg-v0.9.1 data/pv4ger-seg-v0.9.1 --download
```

This converts 4,027 native 320 × 320 RGB crops and masks at 10 cm/pixel from the
pinned GEO-Bench release. Source hashes, notices, and publisher assignments are
preserved. Canonical splits remain unassigned because nearby crops cross the
publisher's split boundaries. See the [import guide](docs/pv4ger-import.md) for
offline use, smoke runs, licenses, and the limits of proximity grouping.

## Prepare geographic splits and review

```sh
uv sync --locked --extra import --extra geo
uv run --locked --extra geo pvextent data split \
  data/pv4ger-seg-v0.9.1 data/pv4ger-seg-geographic-v1 --json
uv run --locked pvextent data review \
  data/pv4ger-seg-v0.9.1 data/review-pv4ger-seg-v0.9.1 --json
```

The [measured split audit](docs/research/geographic-split-audit.md) records
3,282 training, 278 validation and 421 test samples; 46 stay unassigned for
review or spatial buffering. This is an internal source holdout, not a new
independent benchmark. Open the review directory's `index.html` in a browser
to inspect copied images, masks and overlays. Downloaded review decisions are
evidence; they do not automatically change canonical annotations.

The [NRW sampler](docs/nrw-candidates.md) can also fetch up to 24 new roof
candidates. Its source snapshot remains unreviewed with all-ignore masks.
The candidate prediction exporter has run on all seven NRW crops, with
predictions kept separate from source labels.

Two independent AI inspections of 15 candidate crops, explicitly authorized
for experimental training, now produce a separate curated collection. Only
the intersection of their approved background polygons becomes label `0`;
everything else stays ignore `255`. Eleven crops have partial reviewed masks,
and four remain entirely ignored. This is AI review rather than human benchmark
ground truth. No model predictions were used in the curation.

The [augmentation workflow](docs/negative-curation.md) has prepared
`data/pvextent-training-v2`: 3,293 training, 278 validation, 421 test and
42 unassigned records. The original holdout declarations and pixel files are
preserved. A separate fresh baseline run has now used this snapshot; the
original baseline still describes its original 3,282 training examples.

## Run the training smoke test

```sh
uv sync --locked --extra train
uv run --locked --extra train pvextent train smoke \
  data/pv4ger-seg-geographic-v1 runs/smoke-001 \
  --device auto --steps 8 --samples 8 --batch-size 2 --json
```

This trains a small, randomly initialized U-Net using only selected training
pixels. It checks finite losses and gradients, changed parameters, and a
checkpoint reload. See the [training guide](docs/training-smoke.md) for
continuation with optimizer/RNG state and exact dataset fingerprints. Probe
losses describe the same training examples; they are not accuracy results.

## Train the pretrained baseline

```sh
uv sync --locked --extra baseline
uv run --locked --extra baseline pvextent train baseline \
  data/pv4ger-seg-geographic-v1 runs/baseline-001 \
  --config configs/baseline-resnet18.toml \
  --weights-dir data/pretrained/resnet18.a1_in1k \
  --download-weights --device auto --json
```

This fine-tunes an SMP U-Net with a hash-pinned ImageNet ResNet18 encoder from
`timm`. The versioned configuration fixes preprocessing, augmentation, seed,
optimizer and validation threshold. Full training and validation views preserve
native resolution; test pixels are never opened. `best.pt` is selected by global
validation pixel IoU, while `last.pt` retains model, optimizer and RNG state for
epoch-boundary continuation. See the [baseline guide](docs/baseline-training.md)
for configuration, checkpoints and failure recovery. Validation scores describe
the internal source holdout and do not establish roof-level false-positive rates
or independent test performance.

## Compare augmented training

The candidate trained from the same pretrained initialization and configuration
for three epochs on 3,293 examples, with 4,941 optimizer updates. The reference
used 3,282 examples and 4,923 updates. Both selected epoch 3 at threshold `0.5`
using the same 278 validation crops; all original 421 PV4GER test images remain
unopened.

| Internal validation measure | Original baseline | Augmented candidate |
| --- | ---: | ---: |
| Pixel IoU | 0.91572 | 0.91903 |
| Dice | 0.95600 | 0.95781 |
| Precision | 0.95477 | 0.96085 |
| Recall | 0.95724 | 0.95479 |
| Mean crop IoU | 0.87085 | 0.87638 |
| Median crop IoU | 0.92806 | 0.92733 |
| False-positive pixels | 276,590 | 237,303 |
| False-negative pixels | 260,814 | 275,765 |

The candidate has 39,287 fewer false-positive pixels and 14,951 more
false-negative pixels. Its modest IoU and precision gains come with worse
recall; paired crop IoU improves on 139 images, worsens on 138 and ties on one.
Added records also change shuffle order, flips, batches and update
count, so this does not isolate the effect of the eleven new masks or establish
a general quality improvement. These are development pixel scores against
upstream labels, not an independent benchmark or roof-level false-positive
rate. See the [comparison guide](docs/baseline-comparison.md) for reproduction
and the [dedicated audit](docs/research/baseline-negative-comparison-audit.md)
for evidence. No model weights have been released.

## Exploratory negative holdout

A separate geometry-selected NRW pool contains 20 new images. Blind AI reviews
produced seven eligible partial masks; thirteen crops remain unassigned. The
original 50% background-coverage rule admitted none. Before any model
predictions, a registered V2 lowered that gate to 5%, retaining identical
images, review files and checkpoints without expanding labels or filling the
quota. The new holdout stays outside training.

Both fixed models were then evaluated on 150,374 consensus background pixels
across seven new test crops. The reference produced 31 false-positive pixels
(0.0206153% of scored background); the candidate produced zero. Another
566,426 pixels were ignored, including 220 reference and 93 candidate PV
predictions that remain **unscored**. This small AI-labelled pilot does not
measure whole-roof false-positive rates, recall or representative performance.
The candidate remains experimental. See the [guide](docs/negative-holdout.md)
and [audit](docs/research/negative-holdout-audit.md) for protocol history and evidence.

## Package layout

| Module | Responsibility |
| --- | --- |
| `pvextent.data` | Dataset preparation, provenance, validation, and train/validation tensor views |
| `pvextent.models` | Tiny smoke U-Net and pretrained U-Net/ResNet18 adapter |
| `pvextent.training` | Smoke runs, configured baseline training, and resumable checkpoints |
| `pvextent.evaluation` | Pixel metrics, validation comparisons/galleries, and experimental positive/negative pilot diagnostics |
| `pvextent.inference` | Offline candidate review predictions; general inference planned |

Quantitative array matching, roof-level false-positive measurement, an external
benchmark, general inference and released PVExtent weights remain planned.

## License and attribution

Original project code is copyright Valeford and licensed under the
[Apache License 2.0](LICENSE). Preserve the applicable copyright, license,
and [NOTICE](NOTICE) information when redistributing it, as required by the
license.

Datasets and model checkpoints have their own provenance and licensing
requirements. No third-party imagery or model weights are included in this
repository. The importer records the [PV4GER-SEG sources and terms](docs/pv4ger-import.md#source-versions-and-licenses)
separately and preserves upstream notices with each imported snapshot. The
pinned `timm` encoder weights have an Apache-2.0 declaration in their publisher
model card; their [provenance](docs/pretrained-weights.md) is recorded separately
from dependency code and dataset licenses.
New Valeford PV/background/ignore masks from the experimental AI curation are
Apache-2.0 annotations; this does not replace the original imagery or upstream
annotation licenses. Their source notices and AI review evidence accompany the
local curated snapshot.

## Maintained by

<a href="https://valeford.de">
  <img src="docs/assets/valeford-logo.png" alt="Valeford" width="220">
</a>
