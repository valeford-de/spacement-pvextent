# Augmented baseline comparison, 2026-10-20

The fresh run with eleven added partial negative masks reached **0.919035
validation pixel IoU**, compared with **0.915717** for the original baseline:
**+0.3318 percentage points**. It produced **39,287 fewer false-positive pixels**
and **14,951 more false-negative pixels**. Precision improved while recall
fell. This is a development result with a tradeoff, not a general model upgrade
or proof that the negative labels caused the difference.

Both runs completed the fixed three-epoch protocol and selected epoch 3.
The original reference run remains unchanged; the new checkpoint is an
experimental candidate. No weights were released. The
[aggregate evidence](baseline-negative-comparison-v1.json) contains measured
counts, fingerprints, hashes and environment. Real images, per-crop decisions,
predictions, checkpoints and raw logs remain outside Git.

## Frozen experiment

The [protocol](baseline-negative-comparison-protocol.json) was recorded before
the first candidate epoch result, with fixed settings and measurement choices.
See the [comparison guide](../baseline-comparison.md) for reproduction.

- Same U-Net/ResNet18 architecture, pinned ImageNet encoder, fresh seeded decoder
  and optimizer; no continuation from the trained reference.
- Same three epochs, batch size 2, AdamW learning rate/weight decay `0.0001`,
  seed `20261020`, MPS, flips, normalization and threshold `0.5`.
- Native 320 × 320 RGB at 0.1 m/pixel; no resizing.
- Training records: 3,282 → 3,293; every original record unchanged.
- Validation: the same 278 records and 28,467,200 supervised pixels.
- All 421 test examples remain unopened; no test score was computed.

Complete initial-validation records were exactly identical. The configuration
contracts differ only in dataset and training-selection fingerprints; code and
dependency-lock hashes match the recorded protocol. All 699 validation/test
record dictionaries and inherited file identities remained unchanged before
and after training. These identity checks did not open test pixels. Validation
pixels were explicitly used for training-time development evaluation.

The eleven added examples are 0.3340% of training records. Their 217,276
background pixels are 0.06461% of the 336,294,076 supervised pixels processed
per candidate epoch. This share is not their exact loss weight: masked BCE and
batchwise soft Dice depend on batch composition. No oversampling was used.
The larger split also changes shuffling, flip draws and batch composition,
creates a final singleton batch, and performs **4,941 rather than 4,923 updates**.
MPS training is not guaranteed to be bit-identical. A single run per dataset
does not estimate variability or isolate the annotations' causal contribution.

## Measured validation results

| Metric | Original reference | Augmented candidate | Change (pp) |
| --- | ---: | ---: | ---: |
| Global pixel IoU | 0.915717 | 0.919035 | +0.3318 |
| Dice | 0.956005 | 0.957810 | +0.1805 |
| Precision | 0.954771 | 0.960848 | +0.6077 |
| Recall | 0.957241 | 0.954790 | -0.2451 |
| Macro crop IoU | 0.870849 | 0.876379 | +0.5530 |
| Median crop IoU | 0.928061 | 0.927328 | -0.0733 |

Percentage points are absolute changes in the metric multiplied by 100, not
relative percentages. Global scores aggregate pixel counts; macro/median
scores summarize crop IoUs.

| Count | Original reference | Augmented candidate | Change |
| --- | ---: | ---: | ---: |
| True-positive pixels | 5,838,793 | 5,823,842 | −14,951 |
| False-positive pixels | 276,590 | 237,303 | −39,287 |
| False-negative pixels | 260,814 | 275,765 | +14,951 |
| True-negative pixels | 22,091,003 | 22,130,290 | +39,287 |

The paired comparison found **139 crops with higher IoU, 138 with lower IoU,
and one tie**, with no undefined IoUs. The slightly lower median despite higher
global and macro scores reinforces that this is not uniform improvement.
Both frozen best checkpoints reproduce their recorded confusion counts in
separate validation analysis. Pairing verifies matching selection fingerprints,
source-PV/valid-pixel totals, sample identifiers and evaluation environments;
all metrics are recomputed from integer counts.

All epoch results remain visible:

| Epoch | Reference IoU | Candidate IoU | Reference FP / FN | Candidate FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.898400 | 0.895599 | 297,995 / 352,002 | 301,087 / 367,155 |
| 2 | 0.909116 | 0.897156 | 325,297 / 258,622 | 297,447 / 360,451 |
| 3 | 0.915717 | 0.919035 | 276,590 / 260,814 | 237,303 / 275,765 |

The fixed final-epoch comparison and best-checkpoint comparison coincide here
because both selected epoch 3. Earlier candidate epochs performed worse than
the reference; no epoch target, threshold or sampling policy was adjusted in
response. Candidate wall time was **18.53 minutes** on MPS.

## Behavior on the added training negatives

A secondary diagnostic evaluated both best checkpoints on the same eleven
added crops. Only the **217,276 AI-reviewed background pixels** were scored;
909,124 ignored pixels have no ground truth and their predictions were not
classified as errors.

| Training diagnostic | Original reference | Augmented candidate |
| --- | ---: | ---: |
| PV predictions inside approved background | 7,212 | 7,078 |
| Fraction of approved background predicted PV | 3.3193% | 3.2576% |
| Crops with any such prediction | 2 of 11 | 2 of 11 |

This small reduction of 134 pixels does not demonstrate a strong removal of
these failure cases. The candidate trained on all eleven crops; these are
training-set behavior checks, not independent negative-roof performance.
AI-reviewed partial labels remain experimental and do not assert that the
ignored areas or whole properties are PV-free.

## Qualitative inspection and verification

A local gallery shows the four largest paired crop-IoU improvements and the
four largest regressions, each at native resolution. Both contact sheets were
inspected. The examples include recovered source-labelled PV and additional
missed labelled regions, consistent with mixed crop-level effects. Some
source-label ambiguities remain; differences were not used to rewrite labels.
This ranking is diagnostic and not a representative sample.

All **249 tests passed**, including the paired-report comparison and CLI error
path. Comparing the original 278-crop analysis with itself yields 278 ties
and exactly zero differences. Separate code review found no concrete issues.
The new best checkpoint passed exact confusion-count verification after reload
and again in the independent execution of the validation analyzer.
Wheel and source-distribution builds passed; packaged comparison and CLI code
match the source, with dataset pixels and checkpoints excluded.

| Artifact | SHA-256 |
| --- | --- |
| Candidate `best.pt` | `13dfd074d248ab71deee91b976082ad1ff4d9564f0c49eaf917e4334aff16368` |
| Candidate `last.pt` | `cdb958053609ee1b3cb4781f59518da9eb921ab28c5f8d2d379a6fcbd728e720` |
| Candidate frozen runtime config | `a08e1595de910b8dbe97d43af81b220796b03b0d929d5f1194812c3f00f5cbf5` |
| Shared validation selection | `18f47f06d8382037f16c0322ff8475b3b9794ad198cf4c81cb171adeaf17e5a2` |
| Candidate dataset declaration | `1e78b1ce00d826e22d60c50f4c2f7c1b337bc3eeb9f36303b51e53daf0e54f14` |
| Paired local sample measurements | `f9218f4084ef2c9ea39b11716852e772d786f44d828b4456dc83e72b64855b54` |

## Interpretation and next work

Retain both runs and this fixed result. The candidate is interesting for its
higher precision and IoU, while its lower recall needs explicit consideration.
The same validation set selected both checkpoints, and its upstream labels
have not been fully adjudicated. No roof-level false-positive rate, independent
benchmark, statistical significance or production-readiness claim follows.

Next, freeze a geographically separate negative evaluation collection before
adjusting sampling weights or thresholds, and repeat fixed protocols across
seeds to measure run variability. The test partition stays reserved for a
separately fixed evaluation protocol. More training-negative coverage and label
adjudication remain useful; this small enrichment alone does not settle either
false positives or false negatives.
