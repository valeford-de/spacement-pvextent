# NRW training data v1 protocol

**Status**: fixed before triage and training, 2026-10-22.
**Purpose**: add current NRW roofs, above all the hard negatives behind most false
alarms on [NRW test set v1](nrw-test-v1-protocol.md) (glass roofs, skylight bands,
conservatories, dark roofing), to the training data with little manual drawing,
and measure how accurate those labels are.

## Pool

- Built by [`scripts/build_nrw_train_pool.py`](../../scripts/build_nrw_train_pool.py),
  seed `20261023`, into the untracked `data/nrw-train-pool-v1`.
- Up to 80 randomly drawn ALKIS buildings per district (same footprint filter and
  320 × 320 px at 0.1 m/px as the test set) in 28 districts that NRW test set v1
  does not use. Every crop centre lies at least 1,045 m from every evaluation
  location: PV4GER validation and test crops, NRW test set v1 and the frozen
  holdouts. The colour-infrared crop (`nw_dop_cir`, same flight) is stored for
  review; models receive only RGB.
- Four whole districts (Warendorf, Olpe, Heinsberg, Lemgo) form an NRW
  validation split for checkpoint and threshold selection; the rest is training.

## Triage

[`scripts/triage_nrw_pool.py`](../../scripts/triage_nrw_pool.py) runs the frozen
release models and records candidate components: 4-connected regions of at
least 50 px where Tinden (ensemble profile) reaches 0.3 or Vidda (single
profile) reaches 0.5. The low threshold and the higher-recall Vidda keep weak
detections in review instead of labelling them background.

- **No candidate**: the crop becomes an automatic negative; every pixel where
  Tinden exceeds 0.1 is set to ignore.
- **Candidates**: each component is judged visually from the RGB crop and the
  CIR crop by the AI assistant that runs the pipeline, without the human test
  labels: `pv`, `not_pv` (hard negative) or `unsure`. A crop with visible PV that
  no candidate covers is marked `missed` and excluded from training.

## Label construction

- `pv` component: PV where Tinden ≥ 0.505; the rest of the component and a
  2 px ring around it are ignore.
- `not_pv` component: background, including a 3 px ring.
- `unsure` component: ignore.
- All other pixels: background, except Tinden > 0.1, which is ignore.

## Label audit

A random sample of 80 crops, stratified into 40 automatic negatives and 40
reviewed crops, is labelled blind by the human annotator in the same iPad page
used for the test set, with the CIR view available. Reported: crop-level
agreement on PV presence, the share of automatic negatives that contain PV, and
pixel IoU of the constructed PV labels against the human polygons.

## Training and evaluation

- V7 repeats the Vidda (V5) run exactly, with
  [its config](../../configs/deeplabv3plus-resnet50-v5-photometric-v1.toml) from the
  same ImageNet encoder, on Vidda's training data (`pvextent-training-v4`) plus
  the new NRW training crops, so the comparison isolates the added data. The
  checkpoint is chosen on the PV4GER validation split plus the NRW validation
  districts; the threshold stays 0.5.
- NRW test set v1 and the sealed PV4GER test set are each evaluated once with the
  final frozen checkpoint, using
  [`scripts/evaluate_nrw_test.py`](../../scripts/evaluate_nrw_test.py) and the
  existing sealed benchmark, and reported next to Vidda and Tinden.

## Known limitations

- PV that no model detects at all and that the reviewer cannot see stays
  unlabelled in the automatic negatives; the audit measures this rate.
- PV boundaries in the new labels come from Tinden; boundary quality still rests
  on the PV4GER labels.
- The reviewer of the candidates is an AI system; the audit, not the review,
  carries the accuracy claim.

## Amendment 1: label rules v2 and V8 (2026-10-22, after the V7 evaluation)

The audit showed that the v1 rules label about half of the true PV area in the
sample as background, and V7 lost NRW recall. Almost all of those pixels have a
weak Vidda response (≥ 0.1) while Tinden stays below 0.1. This amendment was
designed on the audit sample, so the audit no longer measures v2 labels
independently.

- Label rules v2: every pixel within 3 px of Tinden > 0.1 or Vidda ≥ 0.15 is
  ignore; reviewed `not_pv` components stay background, `pv` and `unsure`
  components are built as in v1. On the audit sample this lowers human PV pixels
  labelled background from 47 % to 7 %, and background falls from 99 % to 63 %
  of pixels.
- The 80 audit crops use the human labels (PV 1, ignore 255, rest background),
  including the two crops the reviewer had excluded as `missed`.
- V8 repeats the V7 run with
  [its own config](../../configs/deeplabv3plus-resnet50-v8-nrw-hardneg-v2.toml)
  (identical to V7's) on `pvextent-training-v8`. The NRW test set v1 and the
  sealed PV4GER test set are evaluated once with the final checkpoint.
