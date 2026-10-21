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
ground-truth scores. Human label adjudication and a representative independent
benchmark remain outstanding. Imported publisher splits are retained as provenance;
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

The [exploratory negative holdout](negative-holdout.md) is also implemented and
executed separately from training. Three NRW regions supplied twenty new crops
under geometry/identity selection and metadata-only exclusion against all
4,034 existing records. Blind AI review and a pre-inference V1-to-V2 protocol
amendment produced seven new test crops and thirteen unassigned records. V1's
50% background gate admitted none; V2 requires 5% without changing images,
review masks or fixed model identities. This is an amended exploratory pilot,
not a result under the original gate. The nearest eligible crop centre is
7,364.646 m from the reference data. The original 421 PV4GER test rasters remain
unopened. See the [pilot audit](research/negative-holdout-audit.md).

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
artifact hashes. All original 421 PV4GER test examples remain unopened, and
PVExtent model weights have not been released. These validation results do not establish
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
and all original 421 PV4GER test examples remain unopened. No weights have been
released.

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
and fixed threshold. All original 421 PV4GER test examples remain unopened.

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

Both fixed checkpoints have now been evaluated on the
[new spatially separate negative pilot](negative-holdout.md). Seven crops
supply 150,374 consensus background pixels (1,503.74 m²), covering 20.9785% of
their combined pixels; 566,426 pixels remain ignored. The reference has 31 FP
and 150,343 TN pixels, with FP / (FP + TN) of `0.0206153%`; the candidate has
zero FP and 150,374 TN pixels. One reference crop and no candidate crops contain
such FP pixels. These are crop diagnostics, not roof-level rates. The 220
reference and 93 candidate PV predictions in ignored regions remain unscored.

This small AI-reviewed partial-label result supplies no recall, Dice or IoU,
human-confirmed benchmark truth or general model-quality claim. The coverage
gate changed from V1 to V2 before predictions, as recorded above; both original
runs and the old holdout are preserved. No new training or tuning occurred.
The [pilot audit](research/negative-holdout-audit.md) records the fixed evidence.

The subsequent [positive pilot](positive-holdout.md) is also complete. Two new
blind AI reviews inspected the thirteen previously unevaluated cases from that
same acquisition cohort. Three pass the unchanged 100-consensus-PV-pixel gate;
ten remain ignored and unassigned. On 3,577 confirmed PV pixels, the original
reference has zero TP and 3,577 FN (**0% recall**); the candidate has 145 TP and
3,432 FN (**4.053676% recall**). There are no confirmed background pixels, so
this subset cannot estimate a background FP rate or full-mask precision/IoU/Dice.
The 457 candidate PV predictions in ignored regions remain unscored.

This exposes poor transfer to these three crops despite high internal validation
scores. A post-result control on two known validation crops exactly reproduced
both models' retained predictions and counts, and the input tensors match the
existing validation loader. No preprocessing or checkpoint-loading defect was
demonstrated; the cause of the source difference remains unresolved. This tiny,
partially annotated AI subset does not estimate complete-array/roof recall or
representative performance. The [positive audit](research/positive-holdout-audit.md)
records the failure, control and frozen evidence.

Neither the negative pilot's low FP counts nor the candidate's small positive
advantage justifies an operational model choice. Keep all twenty acquisition
cases reserved from training and tuning. The immediate development priority is
to inspect source differences using training data, acquire separate NRW training
positives with fuller labels, and register the next comparison before training.
Do not adapt the pilot's labels, probability threshold or model selection to
improve its reported result.

A registered duration control has now continued the unchanged augmented snapshot
from three to six total epochs. The best internal-validation checkpoint is
epoch five: IoU **0.92183825**, Dice **0.95932969**, an IoU increase of
**0.280326 percentage points** over the parent best. The fixed final epoch six
has IoU **0.91783037**. No external-pilot inference or threshold selection was
performed; this small internal gain does not establish a remedy for transfer.
The [continuation audit](research/training-extension-audit.md) records exact
counts, hashes, reload checks and unchanged parent/test data.

Separately, a new training-review acquisition contains **22 native NRW crops**
from Schleiden, Dülmen and the Marsberg area. Identity exclusions cover all
4,054 supplied records; geographic distance protects 761 validation, test and
unassigned centres, including every external-pilot case. Two interrupted
combined attempts were replaced by registered per-region atomic acquisitions
with the same 24 slots; two unsupported full-crop metadata cases were excluded.
Two independent blind AI reviews agree on seven positive proposals, supplying
8,938 proposed PV pixels and 20,899 proposed background pixels. Canonical masks
remain all-ignore and these images were not used in the continuation. Positive
curation, an immutable training snapshot, explicit target-source sampling and a
separate NRW-enriched experiment are the next development work; regular resume
must continue to reject changed dataset fingerprints.

The [bulk collection](bulk-training-data.md) has now added **1,000 distinct
native NRW photos**: 600 current-source crops at original TRAIN locations and
400 distinct ALKIS roofs. Independent source, RGB/hash and spatial checks pass;
all 761 reserved centres stay separated by at least 5,047.008 m. The photos span
2022–2026 and retain acquisition/parent provenance. The earlier 22-image review
pool remains separate. Two independent checks of shared frozen visual proposals
have now completed all forty batches per lane. The initial consensus admitted
485 partial masks. Additional native quality checks withdrew one uncertain PV
case and two contradictory whole-image negatives in a separate immutable stage.
The final V4 snapshot adds **482 partial masks: 246 positive and 236 background
only**; 518 new records remain entirely ignored. It contains **3,775 training,
278 validation, 421 test and 560 unassigned records**. Independent audits
reproduced all new masks, the three all-ignore vetoes, inherited records,
coordinates and protected storage without opening original test or pilot pixels.
See the [final mask audit](research/nrw-bulk-mask-curation-audit.md),
[quality amendment](research/nrw-bulk-mask-quality-amendment.md) and
[collection audit](research/nrw-training-bulk-audit.md).

These sparse AI labels supply less than 1% coverage over the new photos, not
full array boundaries or a representative evaluation set. The next step is a
registered target-source training experiment with explicit sampling and loss
handling, followed by fixed validation. Regular resume must reject changed
dataset fingerprints. The current model and earlier pilot scores remain
unchanged. The [live monitor](live-training.md) is implemented for actual
batch loss and epoch validation; opening it does not launch training.

The following evaluation work remains:

- Evaluate the original reserved test partition under a fixed protocol after
  experiment selection; report internal and external evidence separately.
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
