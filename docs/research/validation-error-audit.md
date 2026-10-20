# Validation error analysis and candidate triage, 2026-10-20

The frozen epoch-3 baseline was analyzed on all **278 validation crops**.
Its original TP/FP/FN/TN counts were reproduced exactly on MPS with the same
libraries, batch size and threshold. No training, threshold search, label edits
or test evaluation occurred. The 421 test examples remain reserved.

The [curated aggregate audit](validation-errors-001.json) records measurements,
execution context and artifact hashes. Per-sample measurements, real pixels,
predictions and review observations remain outside Git. See the
[validation analysis guide](../validation-errors.md) for the executable workflow.

## Score distribution and error location

| Measure | Observed value |
| --- | ---: |
| Global pixel IoU | 0.915717 |
| Mean crop IoU, equal weight per crop | 0.870849 |
| Median crop IoU | 0.928061 |
| Crops with zero source/prediction overlap | 5 |
| Crops with IoU below 0.5 | 12 |
| Crops with IoU below 0.75 | 31 |
| FP plus FN pixels | 537,404 |
| Errors in the three-pixel source-boundary band | 355,404 (66.13%) |
| Errors outside that band | 182,000 (33.87%) |

All 278 crop IoUs were defined. Global IoU weights foreground pixels through
aggregated confusion counts; macro IoU gives a crop with little annotated PV
the same weight as a crop with extensive PV. Their difference exposes failures
hidden by the global score.

The fixed boundary diagnostic uses a three-pixel Chebyshev radius, 0.3 m along
the image axes. It does not change the original score or excuse boundary
errors. This validation source contained no ignored pixels, so no errors were
unclassified. Boundary concentration alone does not establish label error.

| Total source-labelled PV area per crop | Crops | Global IoU within stratum | Recall |
| --- | ---: | ---: | ---: |
| Below 10 m² | 3 | 0.774221 | 0.886578 |
| 10 to below 50 m² | 51 | 0.717185 | 0.774418 |
| At least 50 m² | 224 | 0.920789 | 0.961736 |

These are total annotated areas within each crop, not measured individual
array sizes. The smallest bin has only three examples. Different content and
annotation quality can confound these comparisons; they do not establish a
causal effect of array size.

Source PV touched the crop boundary in 258 examples (stratum IoU 0.917677),
versus 20 without edge contact (0.848790). This heavily cropped source does not
support a balanced causal comparison of truncated and complete arrays.

## Review queue and visual findings

The local gallery contains **30 distinct examples**: the union of the twelve
lowest-IoU crops, twelve with most FP pixels and twelve with most FN pixels.
The largest twelve FP cases account for 30.66% of all FP pixels; the largest
twelve FN cases account for 34.72% of FN pixels. Rankings prioritize review;
they are deliberately not a representative evaluation sample.

Visual inspection of the worst examples found unresolved collector-like,
bright framed/glazing-like and low-contrast roof structures under positive
source polygons, often cut by crop edges. Some predicted-empty cases therefore
need annotation and target-definition adjudication before being classified as
confirmed model misses. Existing source polygons also bridge module gaps,
whereas the [project annotation guide](../annotation-guide.md) targets visible
module surfaces. These are provisional visual observations, not corrected
ground truth. Canonical labels remain unchanged.

All 30 ranked examples received visual AI triage. Other recurring differences
included connected predicted bands between source-labelled rows, filled roof
openings, and edge-width disagreements. Shadowed and clipped module-like
sections also contained plausible real omissions. Thin FN lines sometimes
followed gaps between modules. Neither annotation ambiguity nor boundary
concentration explains away all remaining model failures.

## Candidate inspection

A separate [offline prediction export](../candidate-predictions.md) applied the
same verified checkpoint at threshold 0.5 to all seven unassigned, unreviewed
NRW crops. Their original masks remained entirely `255` ignore. Three crops
had nonempty predictions, totaling 1,724 predicted PV pixels. The largest was
1,621 pixels, or 16.21 m² at the declared native GSD.

The largest proposal covered three rectangular structures on a technical
installation. This is a useful potential false-alarm review case, not a measured
false positive without an adjudicated annotation. Prediction panels start
closed so the RGB evidence can be inspected first. No IoU, precision, recall
or roof-level false-positive rate is calculated from ignored masks.

Separately, a visual AI precheck inspected all seven NRW crops and eight
PV4GER zero-mask crops. All eight zero-mask IDs were confirmed unassigned
against the frozen geographic manifest before their pixels were opened.

| Visual precheck only | PV4GER zero-mask crops | NRW crops | Total |
| --- | ---: | ---: | ---: |
| Likely no visible PV | 5 | 6 | 11 |
| Ambiguous | 3 | 1 | 4 |

The ambiguous cases involved dark or blue collector/glazing-like structures,
cropping, blur or strong shadows. AI prechecks are not human approval or
verified negative labels. All source statuses and splits are unchanged; no new
negative training masks were created. The NRW pool is dominated by industrial
structures and does not represent residential roofs or real-world prevalence.

## Provenance and verification

The checkpoint SHA-256 is
`252e5ec9f20557fdce9ecd8a3c97b3421be3f63fb73e6c05f4df4e24e95e3635`.
The frozen source fingerprint is
`597b9facb7f115448e81af3b88de7e1081d510c38108c294195de05763d841c5`.
Both workflows ran on MPS with the baseline's recorded library versions.
Dataset declarations and every artifact in the original training run were
hashed before and after analysis and remained identical. The visual precheck
also verified 23 source/review mask files remained unchanged.

All **210 tests passed**, including known global/macro confusion counts,
boundary/ignore rules, strict validation-only pixel access with deliberately
unreadable test fixtures, immutable checkpoints, evaluation mode, cleanup on
failure, and refusal to score or relabel candidate placeholder masks.

The next data step is to adjudicate the difficult validation cases under an
explicit target definition, then create a separately versioned reviewed
negative/ambiguous collection. Any later correction must preserve the original
snapshot and its published score. These development observations do not turn
validation into an independent final test.
