# Experimental positive holdout

This workflow measures recall on **doubly confirmed PV pixels** in a small,
spatially separate pilot. Two new blind AI reviews define partial labels before
inference. These labels are experimental evidence, not human-confirmed
benchmark truth. Separate AI passes can still share systematic mistakes; their
workflow separation does not establish statistically independent errors.
The [measured audit](research/positive-holdout-audit.md) records
the actual retained sample and results.

## Fixed selection and boundaries

The starting pool is the thirteen unassigned cases from the twenty-image
[external negative-pilot acquisition](negative-holdout.md). The seven negative
cases already evaluated are excluded by manifest identity before raster access.
The thirteen cases had no predictions in the retained pipeline artifacts when
this positive extension was registered. They belong to the same acquisition
cohort and regions as the negative pilot; this is not a new independent regional
sample or a representative prevalence survey.

The [versioned protocol](research/positive-holdout-protocol-v1.json) fixes all
thirteen IDs and image SHA-256 values, the source snapshot fingerprints, both
checkpoint hashes, probability threshold `0.5`, and a minimum of 100 consensus
PV pixels per eligible crop. At native 0.1 m ground sampling distance, those
100 pixels cover 1 m². The protocol was committed before the new reviews and
predictions. Neither replacement cases nor a later gate amendment are used.

Reference exclusion uses the 4,034-record training snapshot, including all
existing training, validation, test and unassigned centres. The acquisition
requires a centre buffer of at least `5000 + 32 * sqrt(2)` metres. The freeze
binds reference coordinate hashes to acquisition evidence, recomputes distances
from the native EPSG:25832 affine crop centres, and rejects ID, building,
image-hash or spatial-group overlap. Old reference rasters and the seven prior
negative-pilot rasters are never opened by this workflow.

## Independent partial annotation

Each identified AI reviewer inspects every selected original RGB crop at native
320 × 320 resolution, without earlier review notes, other reviewers' labels,
model predictions or checkpoint outputs. The
[annotation guide](annotation-guide.md) defines the visible module surface.
Resolved module gaps are excluded. Array bounding rectangles, completed hidden
modules and uncertain collectors must not become PV labels. Uncertain edges,
shadow and unresolved structures remain ignored.

Each bounded JSON review has these exact top-level fields:

```json
{
  "review_schema_version": "1.0",
  "reviewer": "independent-reviewer-name",
  "reviewer_type": "ai",
  "model_predictions_used": false,
  "samples": []
}
```

`samples` contains every protocol ID exactly once. Each sample has exactly:

- `id` and `image_sha256`, matching the frozen source;
- `full_image_inspected: true`;
- `decision`: `approve-positive` or `defer`;
- `positive_polygons` and `background_polygons`: lists of simple polygons,
  each containing integer `[x, y]` vertices in `0..319`;
- `notes`: the reviewer's uncertainty and interpretation.

A reviewer must not mark overlapping PV and background. Bounds are 64 polygons
per class per sample, 8,192 total vertices per review and 2 MiB per review file.
Reviewer names must remain distinct after trimming and case folding.

The frozen mask is:

| Label | Rule |
| ---: | --- |
| `1` | Intersection of both reviewers' PV unions |
| `0` | Intersection of both reviewers' background unions |
| `255` | Every other pixel, including disagreement |

Both reviewers must approve a crop and its PV intersection must contain at
least 100 pixels. Otherwise its entire mask stays `255`, its status remains
`unreviewed`, and its split stays `unassigned`. Eligible crops become
`reviewed/test`; no crop enters training. The freeze retains all thirteen
selected source images and keeps source response bytes unchanged. It writes
to a new directory atomically, preserving the acquisition evidence, both exact
reviews, source notices and protocol hashes. Imagery remains DL-DE-Zero-2.0;
new Valeford AI annotations use Apache-2.0.

```sh
uv run --locked --extra geo pvextent data freeze-positive-holdout \
  data/nrw-negative-holdout-candidates-v1 \
  data/negative-holdout-frozen-v2 \
  data/positive-holdout-frozen-v1 \
  --exclude data/pvextent-training-v2 \
  --review artifacts/positive-holdout-review-a.json \
  --review artifacts/positive-holdout-review-b.json \
  --protocol docs/research/positive-holdout-protocol-v1.json --json
```

The new command requires a new output path. The freeze fails without publication
if no crop passes the fixed gate. Raw reviews, masks and source imagery stay
outside Git. The training augmentation guard also rejects curated descendants
that declare this reserved holdout lineage.

## Frozen two-model evaluation

The evaluator accepts only the separately frozen thirteen-record positive
cohort. It loads eligible `test` rasters and checks frozen evidence and checkpoint
identities, native dimensions, preprocessing, fixed threshold and minimum PV
coverage. Deferred images remain unopened during evaluation. It uses offline
`eval()` / `no_grad()` inference with a batch size of two; no optimizer, training,
weight download or threshold selection occurs.

```sh
uv run --locked --extra baseline pvextent evaluate positive-holdout \
  data/positive-holdout-frozen-v1 \
  runs/baseline-resnet18-001/best.pt \
  runs/baseline-resnet18-negatives-001/best.pt \
  artifacts/positive-holdout-evaluation-v1 --device mps --json
```

It writes `positive-evaluation-report.json`, `samples.json`, and native 0/1
prediction PNGs for each eligible crop and model. Dataset declarations, selected
rasters, freeze/protocol evidence, checkpoint bytes and checksum companions are
rechecked before atomic publication.

Primary counts are TP and FN on label `1`:

- **Recall on consensus PV:** `TP / (TP + FN)`.
- **Missed consensus PV fraction:** `FN / (TP + FN)`.

FP and TN on the independently confirmed background are separate diagnostics.
The background pixel false-positive rate is `FP / (FP + TN)`, or undefined when
there is no confirmed background. PV predictions on label `255` are explicitly
unscored. Zero-TP crop counts describe these particular crops, not array or roof
recall. Full-mask precision, Dice and IoU are not claimed for partial labels.

## Interpretation and next evidence

Consensus intersections preferentially retain clear PV interiors and may omit
hard PV surfaces altogether. A high recall on those pixels can therefore be
optimistic. Report image count, pixel denominators, TP/FN, ignored coverage,
checkpoint identity and acquisition provenance together. The separate class
screening of positive and negative subsets prevents combining their rates into
a representative confusion matrix or prevalence estimate.

All twenty acquired cases remain reserved from future training and tuning.
Human adjudication with fuller labels, more independent regions and a specified
array-matching rule are needed for broader detection claims. The original 421
PV4GER test images stay reserved until a separate final-test protocol is fixed.
