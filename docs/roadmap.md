# Research roadmap

PVExtent starts as a standalone experiment. The first goal is a reproducible
baseline for photovoltaic array segmentation from orthophotos.

## 1. Establish the dataset

Step 1's [data contract and validator](data-format.md),
[annotation rules](annotation-guide.md), and synthetic format demonstration are
implemented. Step 2's [PV4GER-SEG importer](pv4ger-import.md) prepares the first
public segmentation dataset, preserving source provenance and native resolution.
The [local review gallery](label-review.md) and
[unreviewed NRW roof sampler](nrw-candidates.md) prepare annotation work.
An AI visual pre-review of 15 candidate crops suggested 11 likely PV-free crops
and four ambiguous cases. Every canonical mask and annotation status remains
unchanged; none are verified negatives. [Candidate prediction support](candidate-predictions.md)
is implemented and has run on all seven NRW crops for visual inspection
without ground-truth scores.
Completed label review, verified hard negatives, and an independent test set
remain outstanding. Imported publisher splits are retained as provenance;
they are not adopted as independent evaluation splits.

- Record imagery source, capture date, ground sampling distance, geographic
  reference, usage rights, and content hash.
- Define the target consistently: visible PV array regions, with documented
  rules for occlusion, shadows, small arrays, and image boundaries.
- Create reviewed segmentation masks and include hard negatives such as
  skylights, glass roofs, ventilation equipment, and dark roof surfaces.
- Keep uncertain labels explicit. Model predictions can assist annotation but
  are not ground truth.

## 2. Separate training and evaluation

The [geographic split policy](geographic-splits.md) is implemented and executed
on the complete PV4GER-SEG source. It creates a frozen internal holdout with
10 km blocks, a 1 km centre buffer, duplicate/group/building constraints and
explicit exclusions. See the [measured audit](research/geographic-split-audit.md).
This does not establish independence from published model training data or
replace the planned reviewed external test set.

- Group images by building and geographic area before creating splits.
- Keep repeat captures of a building together; reserve additional acquisition
  dates or sources to measure transfer where data permits.
- Freeze an independent test set before choosing models or thresholds. Use a
  separate validation set for experiment selection.
- Record possible overlap with any pretrained model's training data.

## 3. Train a baseline

The [training smoke workflow](training-smoke.md) is implemented and executed on
MPS. A small randomly initialized U-Net verifies data loading, masked loss,
gradients, optimizer updates, checkpoint reload and continuation; CPU regression
tests verify exact resumed state. See the [measured smoke audit](research/training-smoke-audit.md).
This remains a pipeline check rather than a model-quality result.

The [pretrained U-Net/ResNet18 baseline](baseline-training.md) is implemented
using SMP and a pinned ImageNet encoder from `timm`. Its
[weight provenance and licenses](pretrained-weights.md) retain the publisher's
Apache-2.0 weight declaration separately from dependency and dataset licenses.
The versioned [configuration](../configs/baseline-resnet18.toml) fixes native
resolution, ImageNet channel normalization, paired flips, AdamW settings, seed
and validation threshold. Full train/validation tensor views exclude test pixels.

Training records global validation metrics each epoch, selects `best.pt` by
validation pixel IoU, and saves self-contained `last.pt` checkpoints with model,
optimizer, RNG and earlier-best state. Exact epoch-boundary CPU continuation and
recovery of completed epochs are tested. The first full-split run completed
three epochs on 3,282 training and 278 validation examples, with 4,923 optimizer
updates. The best checkpoint is epoch 3, with internal validation pixel IoU
`0.91572` and Dice `0.95600`. The
[baseline audit](research/baseline-resnet18-audit.md) records the experiment and
artifact hashes. All 421 test examples remain unopened, and PVExtent model
weights have not been released. These validation results do not establish
external or roof-level performance.

- Adjudicate the selected validation disagreements and complete label review,
  including reviewed hard-negative annotations, before further training or
  architectural comparisons.
- Add architectural alternatives only when comparison is useful; retain fixed
  evaluation boundaries and explicit experiment provenance.

## 4. Evaluate the resulting model

Global masked pixel confusion counts, IoU, Dice, precision and recall are
implemented and used for validation during baseline training. Counts accumulate
over all supervised pixels before metrics are computed. Validation-based
checkpoint selection is separate from a final evaluation on untouched test data.

[Frozen validation error analysis](validation-errors.md) is implemented and has
run on all 278 validation crops, exactly reproducing the checkpoint's original
confusion counts. Global pixel IoU is `0.915717`, macro crop IoU `0.870849`, and
median crop IoU `0.928061`; five crops have IoU zero. An offline gallery contains
30 difficult examples selected for review. The fixed three-pixel boundary band
contains 66.13% of error pixels. This diagnostic does not establish annotation
errors; those disagreements still require adjudication. The
[analysis audit](research/validation-error-audit.md) records the evidence.
The checkpoint and threshold remain frozen, with no further training and all
421 test examples still unopened.

The following evaluation work remains:

- Evaluate the selected checkpoint under a fixed test protocol after experiment
  selection; report internal and external evidence separately.
- Measure array detection precision and recall using a documented spatial
  matching rule, including false positives on negative images and roof-level
  false-positive rates.
- Adjudicate error-gallery examples against source imagery and annotation
  rules; extend reviewed hard negatives and inspect transfer across sources,
  resolutions and acquisition dates as those data become available.
- Record runtime and hardware separately from prediction quality. Treat
  uncertainty and abstention explicitly if introduced.

## 5. Publish a reproducible research release

- Include a model card describing intended use, data provenance, limitations,
  evaluation protocol, and results on the frozen test set.
- Publish model inference instructions and releasable weights with dependency
  versions, experiment configurations and checkpoint hashes. Training guidance
  and local run artifacts are already implemented.
- State the license of each released artifact separately and retain applicable
  third-party attribution.

Unless explicitly marked implemented above, these are planned milestones, not
implemented capabilities or measured results.
