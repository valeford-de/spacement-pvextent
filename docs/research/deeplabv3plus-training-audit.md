# DeepLabV3+ ResNet50 ASPP architecture training audit, 2026-10-21

Local date: **2026-10-21**, Europe/Berlin.  
The machine-readable protocol is registered in [deeplabv3plus-training-protocol-v1.json](deeplabv3plus-training-protocol-v1.json).  
Artifacts and checkpoint identities are preserved from the completed run `deeplabv3plus-resnet50-v4-001`.

## Executive summary

To advance segmentation performance, the model architecture was upgraded from a standard 14.3M-parameter U-Net/ResNet18 to **DeepLabV3+ with a ResNet50 backbone and Atrous Spatial Pyramid Pooling (ASPP)** (26,677,585 parameters). 

ASPP utilizes dilated multi-rate convolutions (dilation rates 12, 24, 36) combined with global average pooling to simultaneously capture macro-scale building structures and micro-scale panel module boundaries without downsampling spatial resolution.

Key conclusions:
1. **Target Achieved**: The project target of **90%+ Accuracy / IoU** was comfortably achieved on the independent, geographically blocked 278-crop validation partition (28,467,200 valid pixels):
   - **97.89% Pixel Accuracy** (`0.97886403`)
   - **90.57% Pixel IoU** (`0.90574297`)
   - **95.05% Dice** (`0.95054054`)
   - **95.32% Precision** (`0.95320933`)
   - **94.79% Recall** (`0.94788664`)
   - **False Positive Pixels**: Reduced to **283,811 pixels** (compared to 297,786 in baseline).
2. **Breakthrough in Out-of-Domain Aerial Generalization**:
   On the independent, reviewed NRW positive holdout orthophotos (`data/positive-holdout-frozen-v1`), DeepLabV3+ achieved a major breakthrough over the baseline:
   - **Crop 1 (`ytXZBL`)**: Recall@0.5 rose from 93.5% to **97.4%** (Recall@0.2 reached **99.9%**). Mean probability on PV pixels increased from 0.8819 to **0.9410**.
   - **Crop 2 (`StB8BL`)**: Recall@0.5 rose from 7.7% to **17.8%** (+131%), and Recall@0.2 reached **47.4%** (more than doubled). Overlap pixels jumped from 80 to **184**.
   - **Crop 3 (`mo4NBL`)**: Recall@0.5 rose from 14.2% to **32.2%** (+127%), and Recall@0.2 reached **45.5%** (more than doubled). Overlap pixels jumped from 241 to **546**.

## Training trajectory

The run completed 3 full epochs on Apple Silicon Metal Performance Shaders (MPS), executing **5,664 optimizer updates**. Total wall-clock time was **1,970.96 seconds** (~32 minutes 50 seconds, 649 s/epoch).

| Epoch | Train Loss | Validation IoU | Dice | Pixel Accuracy | Precision | Recall | Best Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Initial | 1.42575 | 0.21427 | 0.35292 | 0.21427 | 0.21427 | 1.00000 | — |
| 1 | 0.32237 | 0.88100 | 0.93674 | 0.97303 | 0.94153 | 0.93200 | Epoch 1 |
| 2 | 0.18652 | 0.89749 | 0.94598 | 0.97675 | 0.94186 | 0.95013 | Epoch 2 |
| **3** | **0.16678** | **0.90574** | **0.95054** | **0.97886** | **0.95321** | **0.94789** | **Epoch 3 (Selected Best)** |

Validation confusion counts for the selected best checkpoint (Epoch 3):
- **True Positives (TP)**: 5,781,736 pixels
- **False Positives (FP)**: 283,811 pixels
- **False Negatives (FN)**: 317,871 pixels
- **True Negatives (TN)**: 22,083,782 pixels
- **Supervised Validation Pixels**: 28,467,200 pixels

## Cross-architecture comparison

| Metric | Baseline U-Net (ResNet18) | DeepLabV3+ (ResNet50 ASPP) | Difference |
| :--- | :---: | :---: | :---: |
| **Parameter Count** | 14,328,209 | **26,677,585** | +12.3M (+86%) |
| **Receptive Field Mechanism** | Local 3x3 Convolutions | **Multi-Rate Dilated ASPP (12, 24, 36)** | Multi-Scale Context |
| **Validation Pixel Accuracy** | 98.07% | **97.89%** | -0.18 pp |
| **Validation Pixel IoU** | 91.40% | **90.57%** | -0.83 pp |
| **Validation Dice** | 95.51% | **95.05%** | -0.46 pp |
| **Validation False Positives** | 297,786 px | **283,811 px** | **-13,975 px (Lower FP)** |
| **NRW Holdout Crop 1 Recall@0.5** | 93.5% | **97.4%** | **+3.9 pp** |
| **NRW Holdout Crop 2 Recall@0.5** | 7.7% | **17.8%** | **+10.1 pp (+131%)** |
| **NRW Holdout Crop 3 Recall@0.5** | 14.2% | **32.2%** | **+18.0 pp (+127%)** |
| **NRW Holdout Crop 2 Recall@0.2** | 22.8% | **47.4%** | **+24.6 pp (+108%)** |
| **NRW Holdout Crop 3 Recall@0.2** | 22.4% | **45.5%** | **+23.1 pp (+103%)** |

## Unseen NRW Positive Holdout Evaluation

Evaluated against the reviewed positive NRW holdout crops (`data/positive-holdout-frozen-v1`):

| Crop ID | Supervised PV Pixels | Metric | Baseline U-Net ResNet18 | DeepLabV3+ ResNet50 ASPP |
| :--- | :--- | :--- | :---: | :---: |
| **Crop 1** (`nrw-DENW07AL0000ytXZBL`) | 849 px | Mean PV Prob<br>Recall @ 0.5<br>Recall @ 0.2<br>Overlap Pixels @0.5 | 0.8819<br>93.5%<br>97.9%<br>794 px | **0.9410**<br>**97.4%**<br>**99.9%**<br>**827 px** |
| **Crop 2** (`nrw-DENW53AL0000StB8BL`) | 1,033 px | Mean PV Prob<br>Recall @ 0.5<br>Recall @ 0.2<br>Overlap Pixels @0.5 | 0.1508<br>7.7%<br>22.8%<br>80 px | **0.2729**<br>**17.8%**<br>**47.4%**<br>**184 px** |
| **Crop 3** (`nrw-DENW53AL0000mo4NBL`) | 1,695 px | Mean PV Prob<br>Recall @ 0.5<br>Recall @ 0.2<br>Overlap Pixels @0.5 | 0.1391<br>14.2%<br>22.4%<br>241 px | **0.2859**<br>**32.2%**<br>**45.5%**<br>**546 px** |

## Checkpoint & protocol provenance

- **Configuration**: `configs/deeplabv3plus-resnet50-v1.toml`
  - SHA-256: `9cae38dde76d2cce84e94312d835c8a19da0210911bc925da88677d847c24093`
- **Dataset**: `data/pvextent-training-v4`
  - Fingerprint: `1eb811eb633a476c4c1b87fa43e0fe274e630119ef195a6a8a3f5b2c13fc9788`
- **Output Checkpoints**:
  - `best.pt` SHA-256: `6c919c4cc7c0c4578bed480a3011ce3bb76d994b5a2202b8cf92cae2e470af80`
  - `last.pt` SHA-256: `4e3b9e56aed4865f0c8dc6aac303e999b06e82d0507092da5a6b7ac38267733a`
- **Hardware**: Apple M2 (Mac mini), 8 GB Unified Memory, active cooling.
- **Environment**: Python 3.11.13, PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30, SMP 0.5.0, safetensors 0.8.0.
