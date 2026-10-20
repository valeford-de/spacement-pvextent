# Positive holdout pilot audit

Local date: **2026-10-21**, Europe/Berlin. UTC acquisition/evaluation log dates
remain 2026-10-20. Machine-readable aggregate evidence is in
[positive-holdout-v1.json](positive-holdout-v1.json); the
[workflow guide](../positive-holdout.md) describes the commands and contracts.

## Result and interpretation

Both existing models miss most doubly confirmed PV pixels in this small external
pilot. The original reference predicts no PV on any of its three complete
images. The augmented candidate recovers only 145 of 3,577 confirmed PV pixels.
The high internal validation scores therefore do not transfer to these cases.
Neither checkpoint is ready for a general operational detection claim.

| Measure | Original reference | Augmented candidate |
| --- | ---: | ---: |
| Evaluated crops | 3 | 3 |
| Confirmed PV pixels | 3,577 | 3,577 |
| True-positive PV pixels | 0 | 145 |
| False-negative PV pixels | 3,577 | 3,432 |
| Recall on consensus PV pixels | 0% | 4.053676% |
| Missed consensus PV pixels | 100% | 95.946324% |
| Crops with zero TP pixels | 3 | 1 |
| PV predictions in ignored areas, unscored | 0 | 457 |

The candidate-minus-reference recall difference is **+4.053676 percentage
points**. This is still very poor recall, not adequate detection. Zero-TP crop
counts are diagnostics of these particular crops, not array or roof recall.

There are **zero confirmed background pixels**. FP and TN counts of zero have
an empty denominator: the background false-positive rate is undefined. Precision,
IoU and Dice for complete masks cannot be estimated here. The 457 candidate
predictions in ignored regions remain unscored, not additional FP or TP.

## Selection registered before new annotations and inference

The [positive protocol](positive-holdout-protocol-v1.json) was committed as
`083d056a6f03f12dd4a2b067baf9d2cd401bda33` at
**2026-10-21 00:13:18 CEST**, before the new reviews and model predictions,
and pushed to GitHub.
Its SHA-256 is
`c31dfe862a48a958e4cbd0e721cc42c89f290c521be58120e9e784318f3ec14f`.

Selection is exactly the thirteen unassigned cases from the twenty-image
[negative-pilot acquisition](negative-holdout-audit.md), excluding the seven
previously evaluated negative cases. Retained prediction and probe artifacts
contained no predictions for these thirteen IDs. The new reviewers did not
read earlier review notes, other reviewers' decisions or model outputs.
No new imagery was requested.

This is a subsequently registered positive extension of the **same external
acquisition cohort**, not an independently sampled new region. All thirteen
cases were inspected; no replacement sampling or gate amendment occurred.
The protocol fixes their image hashes, both checkpoint hashes, threshold `0.5`
and a minimum of **100 consensus PV pixels = 1 m²** at native 0.1 m/pixel.

The thirteen selected cases comprise five Medebach, three Bad Laasphe and five
Lienen crops. Their nearest reference-centre distance is **6,649.257 m**.
Exclusion uses all 4,034 records in the training snapshot, including every
train, validation, test and unassigned centre. The native affine-centre check
and reference coordinate-hash binding enforce the fixed
`5000 + 32 * sqrt(2)` m buffer. Previous reference rasters and the seven negative
pilot rasters are not opened by the positive workflow.

## Two new blind AI reviews and frozen labels

Both independent review workflows inspected all thirteen original 320 × 320 RGB
crops at native resolution. Review A approved three and deferred ten, using
85 PV polygons. Review B approved four and deferred nine, using 40 PV polygons.
Neither drew background polygons. Their exact retained review hashes are:

- A: `8d3d6011db718ef9e02e7926f0aca28b66cc46bce31a95967547527268b952d9`.
- B: `c2e27ff348c96787b313ea50605d16e38aa1e6f93c2940cc740a74fba6737b56`.

Only the intersections of their PV polygons become label `1`; any background
intersection would become `0`. Everything else is `255`. Both positive
approvals and the fixed 100-pixel gate admit **three crops**; ten remain wholly
ignored, unreviewed and unassigned. No label was expanded from predictions.

The frozen dataset `data/positive-holdout-frozen-v1` contains all thirteen
selected source images. Its fingerprint is
`9726f7b6636f5beae6878e635092557a96d004a1422a210cab7ee43ebbdba8f6`.
Freezing completed at **00:20:20 CEST**, before inference began at 00:20:41.

The three evaluated crops are one Medebach and two Lienen cases, captured on
2025-06-20, 2026-03-18 and 2026-03-19. Their minimum reference-centre distance is
**7,130.312 m**. Individual masks contain 849, 1,033 and 1,695 confirmed PV
pixels, totalling **35.77 m²**. They contain 303,623 ignored pixels and no
confirmed background: only **1.164388%** of their combined raster area is
supervised. Including the ten deferred crops, 1,327,623 pixels remain ignored.

Source RGB response bytes and acquisition notices are preserved. NRW imagery
retains DL-DE-Zero-2.0; the new partial AI masks are copyright 2026 Valeford,
Apache-2.0. Actual imagery, polygons, masks, predictions and raw per-image
reports remain outside Git.

## Fixed evaluation and technical checks

Both existing best-epoch-three U-Net/ResNet18 checkpoints were evaluated once:

- Reference: `runs/baseline-resnet18-001/best.pt`, SHA-256
  `252e5ec9f20557fdce9ecd8a3c97b3421be3f63fb73e6c05f4df4e24e95e3635`.
- Candidate: `runs/baseline-resnet18-negatives-001/best.pt`, SHA-256
  `13dfd074d248ab71deee91b976082ad1ff4d9564f0c49eaf917e4334aff16368`.

Inference ran from **00:20:41 to 00:20:48 CEST** on Apple M2 / MPS, batch size
two, native pixels, RGB float32/255 and the shared ImageNet normalization.
`eval()` / `no_grad()` and strict state loading preserve the fixed models.
Torch 2.14.1, torchvision 0.29.1, segmentation-models-pytorch 0.5.0, timm 1.0.30
and safetensors 0.8.0 match the checkpoint environments. The approximately
6.45-second CLI wall time includes startup/loading and is not a throughput
benchmark. No training, weight download, threshold tuning or pilot reinference
occurred.

Because the near-zero recall was unexpected, a separate **post-result technical
control** used the first two sorted, already evaluated validation crops, with
the same batch size and threshold. For both models, manually constructed input
tensors were exactly equal to the existing validation loader's tensors.
Every predicted pixel and TP/FN/FP/TN count exactly matched the retained
validation results: four model/crop comparisons on two distinct images.
The original test split was not used. Independent code review also found the
same preprocessing, normalization, state loading and inference mode in the
positive and validation evaluators.

These checks reveal no demonstrated input-path or checkpoint-loading defect.
They support limited transfer on these three new crops; they do not establish
whether imagery rendering, acquisition conditions, object scale or another
source difference causes it. Those remain hypotheses, not measured causes.

The evaluator checks the frozen cohort, metadata, protocol, raster bytes,
checkpoint bytes and checksum companions before publication. A separate gallery
recount exactly reproduced all reported counts; native mask and prediction
contact sheets were inspected. The 26 retained source/model artifacts and all
699 original validation/test record dictionaries remain unchanged. All original
421 PV4GER test images stay unopened. The complete suite passed **326 tests**.
Wheel and source-distribution builds passed; their changed module bytes match
the working sources and their package metadata contains the current README.
Raw imagery, masks, reviews, run artifacts and checkpoints are excluded.

## Limits and next work

Only three selected positive crops in two regions qualify, and the masks cover
clear PV interiors rather than complete arrays. Consensus intersections can
preferentially omit difficult PV; separate AI passes can also share systematic
mistakes. These are experimental labels, not human-confirmed benchmark truth.
Even on these selected surfaces, the models miss almost everything.

The earlier seven-case negative pilot's low FP counts cannot establish useful
detection when positive recall is this poor. The class-screened subsets share
an acquisition cohort and cannot be combined into a representative confusion
matrix or prevalence estimate. The candidate's small advantage does not justify
choosing it as a deployable winner.

Keep all twenty acquired cases reserved from training and tuning. Next, inspect
source differences using training data and acquire separate, geographically
excluded NRW **training positives** with fuller visible-module labels. Fix the
next training comparison before running it. Broader, independently reviewed
positive evaluation and a documented array-matching rule are needed before
relying on internal validation scores. Preserve the current pilot as a recorded
failure case; do not lower its threshold or adapt its labels to improve scores.
