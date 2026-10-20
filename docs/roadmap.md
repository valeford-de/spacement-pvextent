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
The [two-pass AI negative curator](negative-curation.md) is implemented and
explicitly authorized for experimental training. Independent visual reviews of
15 crops, without model predictions, produce 11 partial background masks from
their polygon intersections; four ambiguous crops remain entirely ignored.
AI provenance is explicit, source snapshots remain unchanged, and this does not
claim human-reviewed benchmark truth. [Candidate prediction support](candidate-predictions.md)
has separately run on all seven NRW crops for visual inspection without
ground-truth scores. Human label adjudication and an independent test set
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

The executed [training augmentation](negative-curation.md) adds 11 eligible
AI-reviewed negatives to training in a new immutable snapshot. It contains
3,293 training, 278 validation, 421 test and 42 unassigned records. Frozen
validation/test declarations are unchanged, and inherited holdout rasters were
hard-linked without opening their pixels. Identity, reserved-block and 1 km
centre-distance checks protect the holdout; the nearest added training centre
is 3,258.635 m away. See the [curation audit](research/negative-curation-audit.md).

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

A fresh [comparison run](baseline-comparison.md) on the augmented snapshot is
complete under the same three-epoch protocol, initialization, seed and
configuration. It used 3,293 training examples and 4,941 updates, selecting
epoch 3 on the unchanged 278 validation crops. Its validation pixel IoU is
`0.91903` and Dice `0.95781`. Precision rose from `0.95477` to `0.96085`, while
recall fell from `0.95724` to `0.95479`. False-positive pixels fell from
276,590 to 237,303 (39,287 fewer), but false-negative pixels rose from 260,814
to 275,765 (14,951 more). The
[comparison audit](research/baseline-negative-comparison-audit.md) records this
candidate separately from the original baseline.

This is a modest IoU/precision gain with a recall trade-off, not a general
quality improvement. Adding records changes shuffling, flips, batches and
optimizer-update count; these runs do not isolate a causal effect of the eleven
new masks. The original baseline artifacts and validation scores are preserved,
and all 421 test examples remain unopened. No weights have been released.

- Inspect paired validation diagnostics and annotation disagreements before
  changing sampling or running further training comparisons.
- Adjudicate the selected validation disagreements and improve annotation
  coverage. Experimental AI-reviewed partial negatives do not replace a
  human-reviewed evaluation set.
- Add architectural alternatives only when comparison is useful; retain fixed
  evaluation boundaries and explicit experiment provenance.

## 4. Evaluate the resulting model

Global masked pixel confusion counts, IoU, Dice, precision and recall are
implemented and used for validation during baseline training. Counts accumulate
over all supervised pixels before metrics are computed. Validation-based
checkpoint selection is separate from a final evaluation on untouched test data.

[Frozen validation error analysis](validation-errors.md) is implemented and has
run on all 278 validation crops, exactly reproducing the original baseline's
confusion counts. For that reference checkpoint, global pixel IoU is
`0.915717`, macro crop IoU `0.870849`, and
median crop IoU `0.928061`; five crops have IoU zero. An offline gallery contains
30 difficult examples selected for review. The fixed three-pixel boundary band
contains 66.13% of error pixels. This diagnostic does not establish annotation
errors; those disagreements still require adjudication. The
[analysis audit](research/validation-error-audit.md) records the evidence.
The original checkpoint and threshold remain frozen. The augmented candidate
is a separate training run; its comparison uses the same validation boundaries
and fixed threshold. All 421 test examples remain unopened.

The [paired comparison](baseline-comparison.md) has also completed, reproducing
the candidate checkpoint's counts exactly. Crop IoU improves on 139 validation
images, worsens on 138 and ties on one; none are undefined. Mean crop IoU rises
from `0.87084945` to `0.87637946`, while median crop IoU falls from `0.92806051`
to `0.92732788`. This mixed result keeps the candidate experimental, with the
original reference retained.

A separate diagnostic on the eleven added training images finds 7,078 predicted
PV pixels within their 217,276 approved background pixels, compared with 7,212
for the reference: only 134 fewer. Both models make these predictions on two
of the eleven crops. Ignored areas remain unscored. This uses training images,
so it does not supply independent evidence of a lower false-positive rate.
See the [comparison audit](research/baseline-negative-comparison-audit.md).

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
