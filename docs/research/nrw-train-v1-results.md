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
| Human PV pixels labelled PV / ignore / background | 34.3 % / 38.6 % / 27.1 % |

The constructed labels are conservative: PV labels are rarely wrong, but about a
quarter of the true PV area in the sample is labelled background.

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
- Next step: relabel the background-labelled PV, for example by setting all
  pixels where Tinden exceeds 0.3 in `not_pv` crops of buildings with accepted PV
  to ignore, and add the 80 human-labelled audit crops to training.
