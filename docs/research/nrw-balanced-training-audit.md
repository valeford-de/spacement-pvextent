# Schema 1.1 domain-balanced & photometric training audit, 2026-10-21

Local date: **2026-10-21**, Europe/Berlin.  
The machine-readable protocol is registered in [nrw-balanced-training-protocol-v1.json](nrw-balanced-training-protocol-v1.json).  
Artifacts and checkpoint identities are preserved from the completed run `baseline-resnet18-balanced-v4-001`.

## Executive summary

Following the successful baseline training on `data/pvextent-training-v4` (which achieved 98.07% Accuracy and 91.40% IoU), an experimental run (`baseline-resnet18-balanced-v4-001`) was evaluated under **Schema 1.1** to test whether forced domain balancing (40% representation of modern NRW tiles per epoch) combined with photometric jittering (15% brightness and contrast) would further boost cross-domain generalization.

Key conclusions:
1. **Target Achieved**: The user's requirement of **90%+ Accuracy / IoU** was again surpassed on the independent validation split:
   - **97.77% Pixel Accuracy** (`0.97769450`)
   - **90.13% Pixel IoU** (`0.90131543`)
   - **94.81% Dice** (`0.94809669`)
   - **94.54% Precision** (`0.94542351`)
   - **95.08% Recall** (`0.95078503`)
2. **Mechanistic Finding & Model Selection**:
   - In `data/pvextent-training-v4`, native NRW tiles are ALKIS building-specific partial crops where **98.07% of pixels are ignored** (`255`), and >50% of supervised footprints are clean background roofs (`0` PV).
   - Artificially concentrating 40% of all batches on these sparse partial masks with photometric jittering caused over-regularization: the model learned a strong conservative bias toward predicting background, dampening positive recall on unseen aerial crops.
   - **Champion Model**: The **V4 Baseline (`baseline-resnet18-bulk-v4-001`)** with its natural data distribution (~11.3% NRW inclusion) remains the superior model:
     - Higher Validation IoU (**91.40%** vs 90.13%)
     - Higher Validation Accuracy (**98.07%** vs 97.77%)
     - Significantly higher recall on unseen NRW holdout crops (**93.5%** on Crop 1 vs 17.9%).

## Training trajectory

The run completed 3 full epochs on Apple Silicon Metal Performance Shaders (MPS), executing **5,664 optimizer updates**. Total wall-clock time was **1,003.28 seconds** (~16 minutes 43 seconds, 330.5 s/epoch).

| Epoch | Train Loss | Validation IoU | Dice | Pixel Accuracy | Precision | Recall | Best Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Initial | 1.32769 | 0.05253 | 0.09981 | 0.76604 | 0.28424 | 0.06053 | — |
| 1 | 0.41814 | 0.88503 | 0.93901 | 0.97402 | 0.94456 | 0.93352 | Epoch 1 |
| 2 | 0.23543 | 0.88776 | 0.94055 | 0.97406 | 0.92395 | 0.95775 | Epoch 2 |
| **3** | **0.20505** | **0.90132** | **0.94810** | **0.97769** | **0.94542** | **0.95079** | **Epoch 3 (Selected Best)** |

Validation confusion counts for the selected best checkpoint (Epoch 3):
- **True Positives (TP)**: 5,799,415 pixels
- **False Positives (FP)**: 334,783 pixels
- **False Negatives (FN)**: 300,192 pixels
- **True Negatives (TN)**: 22,032,810 pixels
- **Supervised Validation Pixels**: 28,467,200 pixels

## Cross-run comparison

| Metric | V4 Baseline (`bulk-v4-001`) | V4 Balanced (`balanced-v4-001`) | Difference |
| --- | ---: | ---: | ---: |
| **Domain Balance (NRW)** | Natural (~11.3%) | **40.0%** | +28.7 pp |
| **Photometric Jitter** | None (0.0) | **15.0%** (Brightness & Contrast) | +15 pp |
| **Validation Pixel Accuracy** | **98.07%** | 97.77% | -0.30 pp |
| **Validation Pixel IoU** | **91.40%** | 90.13% | -1.27 pp |
| **Validation Dice** | **95.51%** | 94.81% | -0.70 pp |
| **Validation Precision** | **95.15%** | 94.54% | -0.61 pp |
| **Validation Recall** | **95.86%** | 95.08% | -0.78 pp |
| **Validation Loss** | **0.13381** | 0.16215 | +0.02834 |
| **NRW Holdout Crop 1 Recall@0.5** | **93.5%** | 17.9% | -75.6 pp |

## Unseen NRW Positive Holdout Evaluation

Evaluated against the reviewed positive NRW holdout crops (`data/positive-holdout-frozen-v1`):

| Crop ID | Supervised PV Pixels | Metric | V4 Baseline | V4 Balanced (40% + Jitter) |
| :--- | :--- | :--- | :---: | :---: |
| **Crop 1** (`nrw-DENW07AL0000ytXZBL`) | 849 px | Mean PV Prob<br>Recall @ 0.5<br>Recall @ 0.2 | 0.8819<br>**93.5%**<br>**97.9%** | 0.1686<br>17.9%<br>25.3% |
| **Crop 2** (`nrw-DENW53AL0000StB8BL`) | 1,033 px | Mean PV Prob<br>Recall @ 0.5<br>Recall @ 0.2 | 0.1508<br>**7.7%**<br>**22.8%** | 0.0062<br>0.0%<br>0.0% |
| **Crop 3** (`nrw-DENW53AL0000mo4NBL`) | 1,695 px | Mean PV Prob<br>Recall @ 0.5<br>Recall @ 0.2 | 0.1391<br>**14.2%**<br>**22.4%** | 0.0418<br>0.0%<br>7.5% |

## Checkpoint & protocol provenance

- **Configuration**: `configs/baseline-resnet18-balanced-v1.toml`
  - SHA-256: `701886988cb91f1c558f9e970e102fc819d7b81a639de356a73e8625058079b6`
- **Dataset**: `data/pvextent-training-v4`
  - Fingerprint: `1eb811eb633a476c4c1b87fa43e0fe274e630119ef195a6a8a3f5b2c13fc9788`
- **Output Checkpoints**:
  - `best.pt` SHA-256: `bbcc8495c534699e4f2f85f8b2fb3872157e30176756124bb2cbea293e0511c3`
  - `last.pt` SHA-256: `5b79a2567927c2d00329b37e2899d48a97a6455a5b45f64c3ee07c99e9deba84`
- **Champion Checkpoint**:
  - Retained as `runs/baseline-resnet18-bulk-v4-001/best.pt`
  - SHA-256: `32ca5148f0e18879c14fcc7b0b0e3e0ba09e272f25cf2cf6a45b8a9fd793f8bd`
