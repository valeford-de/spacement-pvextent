# V4 bulk NRW training audit, 2026-10-21

Local date: **2026-10-21**, Europe/Berlin.
The machine-readable protocol is registered in [nrw-bulk-training-protocol-v1.json](nrw-bulk-training-protocol-v1.json).
Artifacts and checkpoint identities are preserved from the completed run `baseline-resnet18-bulk-v4-001`.

## Executive summary

The baseline U-Net/ResNet18 architecture was trained on the newly audited **`data/pvextent-training-v4`** dataset, incorporating **3,775 training samples** (3,282 legacy PV4GER-SEG examples, 11 audited background holdouts, and **482 newly curated native NRW orthophoto masks** with 246 positive arrays and 236 clean background roofs).

On the unchanged, geographically blocked **278-crop validation partition** (28,467,200 valid pixels), the model reached:
- **98.07% Pixel Accuracy** (`0.98067653`)
- **91.40% Pixel IoU** (`0.91401419`)
- **95.51% Dice** (`0.95507567`)
- **95.86% Recall** (`0.95863684`)
- **95.15% Precision** (`0.95154086`)

The user target of **90%+ Accuracy / IoU** was comfortably achieved on the independent validation split while integrating modern German orthophotos into the training distribution.

## Training trajectory

The run completed three epochs on Apple Silicon Metal Performance Shaders (MPS), executing **5,664 optimizer updates**. Total wall-clock time was **999.94 seconds** (~16 minutes 40 seconds, average 329 seconds per epoch).

| Epoch | Train Loss | Validation IoU | Dice | Pixel Accuracy | Precision | Recall | Best Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Initial | 1.32769 | 0.05253 | 0.09981 | 0.76604 | 0.28424 | 0.06053 | — |
| 1 | 0.36942 | 0.89816 | 0.94635 | 0.97727 | 0.95756 | 0.93539 | Epoch 1 |
| 2 | 0.18927 | 0.90125 | 0.94806 | 0.97789 | 0.95437 | 0.94183 | Epoch 2 |
| **3** | **0.16198** | **0.91401** | **0.95508** | **0.98068** | **0.95154** | **0.95864** | **Epoch 3 (Selected Best)** |

Validation confusion counts for the selected best checkpoint (Epoch 3):
- **True Positives (TP)**: 5,847,308 pixels
- **False Positives (FP)**: 297,786 pixels
- **False Negatives (FN)**: 252,299 pixels
- **True Negatives (TN)**: 22,069,807 pixels
- **Supervised Validation Pixels**: 28,467,200 pixels

## Cross-run comparison

| Metric | V1 Baseline (3 ep) | V2 Candidate (3 ep) | V2 Continued (6 ep) | V4 Candidate (3 ep, New) |
| --- | ---: | ---: | ---: | ---: |
| **Train Samples** | 3,282 | 3,293 | 3,293 | **3,775** |
| **Added NRW Orthophotos** | 0 | 11 | 11 | **482** |
| **Pixel Accuracy** | 98.11% | 98.20% | 98.26% | **98.07%** |
| **Pixel IoU** | 91.57% | 91.90% | 92.18% | **91.40%** |
| **Dice** | 95.60% | 95.78% | 95.93% | **95.51%** |
| **Precision** | 95.48% | 96.08% | 96.09% | **95.15%** |
| **Recall** | 95.72% | 95.48% | 95.78% | **95.86%** |
| **Validation Loss** | 0.13203 | 0.12823 | 0.12442 | **0.13381** |

Validation performance remains rock-solid above 91% IoU and 98% Accuracy, while training now encompasses the 482 modern orthophoto partial labels designed to overcome aerial domain shift.

## Execution and artifact hashes

- **Dataset**: `data/pvextent-training-v4`
  - Fingerprint: `1eb811eb633a476c4c1b87fa43e0fe274e630119ef195a6a8a3f5b2c13fc9788`
  - Validation Selection Fingerprint: `18f47f06d8382037f16c0322ff8475b3b9794ad198cf4c81cb171adeaf17e5a2`
- **Checkpoints**:
  - `best.pt` SHA-256: `32ca5148f0e18879c14fcc7b0b0e3e0ba09e272f25cf2cf6a45b8a9fd793f8bd`
  - `last.pt` SHA-256: `b8c3300cfeb7196957ff513dc6bcbbd6ce4523531fc0239a8696099db8bbd3b6`
- **Hardware**: Apple M2 (Mac mini), 8 GB Unified Memory, active fan cooling.
- **Environment**: Python 3.11.13, PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30, SMP 0.5.0, safetensors 0.8.0.
