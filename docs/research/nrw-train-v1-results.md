# NRW training data v1 results

Results for the protocol in [nrw-train-v1-protocol.md](nrw-train-v1-protocol.md),
measured 2026-10-22. Imagery, labels and checkpoints stay outside Git.

## Pool and triage

- 1,832 crops from 24 districts (4 of the 28 listed had no roof far enough from
  every evaluation location): 1,512 train, 320 validation.
- 969 crops had at least one candidate component (1,912 components); 863 became
  automatic negatives.
- Review verdicts: 352 components `pv` (236 crops), 1,298 `not_pv`, 262 `unsure`;
  29 crops marked `missed` and excluded. Most `not_pv` components were cars,
  skylights, glass roofs, conservatories, shadows and dark roofing.
- Assembled dataset `pvextent-training-v7`: 5,263 train and 593 validation crops
  (Vidda's `pvextent-training-v4` plus 1,488 train and 315 validation NRW crops),
  0 validation errors.

## Label audit (80 crops, blind human labels with CIR view)

| | Result |
|---|---|
| Automatic negatives with PV (human) | 1 of 40 (2.5 %, Wilson 95 % CI 0.4–12.9 %); flagged unsure by the annotator |
| Reviewed crops, PV presence: both PV | 4 |
| Reviewed crops: human PV, crop excluded as `missed` | 2 |
| Reviewed crops: human PV, reviewer `unsure` (ignore) | 1 |
| Reviewed crops: human PV, reviewer `not_pv` | 2 (one clear module field; one glass roof the annotator called PV) |
| Reviewed crops: reviewer PV, human no PV | 1 (small dark field, possibly a roof window) |
| Pixel precision of constructed PV labels | 84.2 % |
| Human PV pixels labelled PV / ignore / background | 24.7 % / 27.9 % / 47.4 % |

The constructed labels are conservative: PV labels are rarely wrong, but about
half of the true PV area in the sample is labelled background. (A first version of
this report gave 34.3 / 38.6 / 27.1 %, computed with the wrong denominator.)

## Model results

V7 (`runs/deeplabv3plus-resnet50-v7-nrw-hardneg-001/best.pt`, epoch 3, validation
IoU 0.9038), single model, no TTA, threshold 0.5. Each test set was evaluated once.

**NRW test set v1** (321 crops, 42 with PV; 95 % bootstrap CIs over crops):

| Model | IoU | Precision | Recall | PV crops detected | False alarms on PV-free crops |
|---|---|---|---|---|---|
| Vidda (V5) | 44.1 % [33.3–52.5] | 49.2 % | 81.2 % | 83.3 % | 33.7 % [28.4–38.9] (94/279) |
| Tinden (V5+V6, D4 TTA) | 53.3 % [42.5–61.8] | 65.9 % | 73.6 % | 81.0 % | 16.5 % [12.0–21.1] (46/279) |
| **V7** | **54.1 % [42.4–63.3]** | **74.8 %** | 66.2 % | 64.3 % | **6.5 % [3.9–9.4] (18/279)** |

Paired bootstrap differences: V7 − Vidda IoU +3.6 to +16.3 points, false alarms
−32.5 to −21.6 points; V7 − Tinden IoU −3.9 to +5.4 points (no measurable
difference), false alarms −14.0 to −6.5 points.

**Sealed PV4GER test set** (421 crops, single model, no TTA):

| Model | IoU | Precision | Recall |
|---|---|---|---|
| Vidda (V5) | 88.16 % | 93.89 % | 93.53 % |
| V7 | 88.83 % | 94.46 % | 93.71 % |

## Reading

- The hard negatives work: V7 cuts false alarms on PV-free NRW roofs from 34 %
  (Vidda) and 17 % (Tinden) to 6.5 %, with a single forward pass instead of
  Tinden's 16, and does not lose accuracy on PV4GER.
- The cost is recall on NRW: V7 detects 27 of 42 PV crops against 35 for Vidda,
  and misses some large arrays Vidda finds (20–70 m²). This matches the audit:
  the constructed labels mark part of the real PV area as background, which
  teaches the model to under-call PV.
- Next step: label rules v2 and V8, see amendment 1 of the protocol.

## Amendment 1 results: V8 (label rules v2)

V8 (`runs/deeplabv3plus-resnet50-v8-nrw-hardneg-v2-001/best.pt`, epoch 2,
validation IoU 0.8895), single model, no TTA, threshold 0.5, evaluated once on
each test set on 2026-10-23.

**NRW test set v1**:

| Model | IoU | Precision | Recall | PV crops detected | False alarms on PV-free crops |
|---|---|---|---|---|---|
| Vidda (V5) | 44.1 % [33.3–52.5] | 49.2 % | 81.2 % | 35/42 | 33.7 % (94/279) |
| Tinden | 53.3 % [43.0–61.9] | 65.9 % | 73.6 % | 34/42 | 16.5 % (46/279) |
| V7 | 54.1 % [42.1–63.1] | 74.8 % | 66.2 % | 27/42 | 6.5 % (18/279) |
| V8 | 48.3 % [38.2–56.9] | 58.4 % | 73.6 % | 31/42 | 8.6 % [5.6–12.0] (24/279) |

Paired bootstrap, V8 − V7: IoU −12.0 to +0.8, recall −0.0 to +15.5, false alarms
−1.1 to +5.5 points. V8 − Tinden: false alarms −11.7 to −4.3 points, IoU and
recall not distinguishable.

**Sealed PV4GER test set**: V8 IoU 87.91 %, precision 91.88 %, recall 95.32 %
(Vidda 88.16 / 93.89 / 93.53 %, V7 88.83 / 94.46 / 93.71 %).

Reading: v2 labels recover recall (31 instead of 27 detected PV crops) and keep
false alarms low, but V8 spills over PV boundaries: false-positive pixels on PV
crops rise from 16,116 (V7) to 62,792. The ignore band around weak responses
removes the background supervision next to modules. Neither V7 nor V8 dominates;
better PV boundaries need human PV polygons, not wider ignore regions.
