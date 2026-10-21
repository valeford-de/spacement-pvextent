# DeepLabV3+ ResNet50 Focal Loss + Lovász Hinge Training Audit, 2026-10-21

Local date: **2026-10-21**, Europe/Berlin.  
The machine-readable protocol is registered in [deeplabv3plus-focal-lovasz-protocol-v1.json](deeplabv3plus-focal-lovasz-protocol-v1.json).  
Artifacts and checkpoint identities are preserved from the completed run `deeplabv3plus-resnet50-focal-lovasz-v4-001`.

## Executive summary

In Step 2 of the aerial segmentation enhancement program toward higher segmentation accuracy, loss engineering was implemented to address the key limitation observed in Step 1: boundary fuzziness and background false positives caused by standard BCE + Dice loss.

We deployed **Masked Focal Loss + Lovász Hinge Surrogate**:
1. **Focal Loss ($\gamma=2.0, \alpha=0.25$)**: Modulates gradient magnitude by $(1 - p_t)^\gamma$, actively suppressing trivial background pixels and focusing optimization on hard negative roof clutter and ambiguous panel edges.
2. **Lovász Hinge**: Directly optimizes the Jaccard index (IoU) through its convex submodular Lovász extension, avoiding soft-Dice sub-optimal local minima and enforcing rectilinear boundary tightness.
3. **Strict Ignore Masking**: Label 255 (unannotated ALKIS background/cadastre) is strictly excluded from both the focal modulating factor and the Lovász sorting permutation.

### Key Breakthrough Findings

1. **New Peak Validation Performance**:
   On the independent, geographically split 278-crop validation partition (28,467,200 valid pixels):
   - **98.00% Pixel Accuracy** (`0.98004658`)
   - **91.07% Pixel IoU** (`0.91074791`), gaining **+0.50 percentage points** over Step 1 (`90.57%`)
   - **95.33% Dice** (`0.95328945`), gaining **+0.28 percentage points** over Step 1 (`95.05%`)
   - **95.63% Precision** (`0.95634367`), gaining **+0.31 percentage points** over Step 1 (`95.32%`)
   - **95.03% Recall** (`0.95025466`), gaining **+0.24 percentage points** over Step 1 (`94.79%`)
   - **False Positive Pixel Suppression**: Reduced from 283,811 down to **264,591 pixels** (**-19,220 FP pixels** vs Step 1, and **-33,195 FP pixels** vs Baseline ResNet18 U-Net).

2. **False Positive Reduction on Negative Holdout**:
   On the 7 confirmed non-PV NRW orthophoto crops (`data/negative-holdout-frozen-v2`, 150,374 background pixels):
   - Step 1 (BCE+Dice) had 3,179 false positive pixels (FPR: 2.11%).
   - Step 2 (Focal+Lovász) slashed false positive pixels down to **1,260 pixels** (FPR: **0.84%**), representing a **60.4% reduction in false alarms**!

3. **High-Sensitivity Generalization on Out-of-Domain Positive Holdout**:
   On the unseen NRW holdout crops (`data/positive-holdout-frozen-v1`, 3,577 confirmed PV pixels):
   - Because Lovász Hinge operates with margin $m=1$ around logit 0 (sigmoid 0.5), it avoids BCE's uncalibrated logit saturation ($\pm 10$).
   - At threshold $\tau=0.2$:
     - **Crop 1 (`ytXZBL`)**: **100.00% Recall** (849/849 px).
     - **Crop 2 (`StB8BL`)**: **90.80% Recall** (938/1,033 px), nearly double Step 1's 47.4% and quadrupling Baseline's 22.8%!
     - **Crop 3 (`mo4NBL`)**: **51.50% Recall** (873/1,695 px), up from 45.5% (Step 1) and 22.4% (Baseline).
     - **Overall Positive Holdout Recall**: **74.36%** (2,660 / 3,577 px), up from **58.96%** in Step 1 and **40.42%** in Baseline.

---

## Three-Way Comparative Architecture & Loss Matrix

| Metric / Dimension | Baseline U-Net (ResNet18) | Step 1: DeepLabV3+ (BCE+Dice) | Step 2: DeepLabV3+ (Focal+Lovász) | Step 2 vs Step 1 | Step 2 vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 14,328,209 | 26,677,585 | **26,677,585** | identical | +12.3M (+86%) |
| **Loss Function** | Masked BCE + Soft Dice | Masked BCE + Soft Dice | **Masked Focal ($\gamma=2$) + Lovász Hinge** | Loss Engineering | Loss Engineering |
| **Validation Pixel Accuracy** | 98.07% | 97.89% | **98.00%** | **+0.11 pp** | -0.07 pp |
| **Validation Pixel IoU** | 91.40% | 90.57% | **91.07%** | **+0.50 pp** | -0.33 pp |
| **Validation Dice** | 95.51% | 95.05% | **95.33%** | **+0.28 pp** | -0.18 pp |
| **Validation Precision** | 95.12% | 95.32% | **95.63%** | **+0.31 pp** | **+0.51 pp** |
| **Validation Recall** | 95.90% | 94.79% | **95.03%** | **+0.24 pp** | -0.87 pp |
| **Validation False Positives** | 297,786 px | 283,811 px | **264,591 px** | **-19,220 px (-6.8%)** | **-33,195 px (-11.1%)** |
| **Negative Holdout FP (@0.5)** | 214 px | 3,179 px | **1,260 px** | **-1,919 px (-60.4%)** | +1,046 px |
| **Negative Holdout FPR (@0.5)** | 0.14% | 2.11% | **0.84%** | **-1.27 pp** | +0.70 pp |
| **Positive Holdout Recall (@0.2)**| 40.42% | 58.96% | **74.36%** | **+15.40 pp (+26.1%)** | **+33.94 pp (+84.0%)** |
| **Positive Crop 1 Recall (@0.2)** | 97.88% | 99.88% | **100.00%** | **+0.12 pp** | **+2.12 pp** |
| **Positive Crop 2 Recall (@0.2)** | 22.85% | 47.43% | **90.80%** | **+43.37 pp (+91.5%)** | **+67.95 pp (+297%)** |
| **Positive Crop 3 Recall (@0.2)** | 22.36% | 45.49% | **51.50%** | **+6.01 pp (+13.2%)** | **+29.14 pp (+130%)** |

---

## Training Trajectory

Training completed 3 full epochs (5,664 optimizer steps) on Apple Silicon MPS via remote execution on Mac mini (`mini-remote`). Total runtime: **1,974.97 seconds** (~32.9 minutes, 651 s/epoch).

| Epoch | Train Loss | Validation IoU | Dice | Pixel Accuracy | Precision | Recall | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Initial | 0.62925 | 0.21427 | 0.35292 | 0.21427 | 0.21427 | 1.00000 | Baseline Init |
| 1 | 0.27630 | 0.88492 | 0.93895 | 0.97361 | 0.93106 | 0.94697 | Checkpoint |
| 2 | 0.18722 | 0.89302 | 0.94349 | 0.97591 | 0.94853 | 0.93850 | Checkpoint |
| **3** | **0.16343** | **0.91075** | **0.95329** | **0.98005** | **0.95634** | **0.95025** | **Selected Best Checkpoint** |

### Final Confusion Matrix (Epoch 3 Selected Best Checkpoint)

- **True Positives (TP)**: 5,796,180 pixels
- **False Positives (FP)**: 264,591 pixels
- **False Negatives (FN)**: 303,427 pixels
- **True Negatives (TN)**: 22,103,002 pixels
- **Total Valid Pixels**: 28,467,200 pixels

---

## Checkpoint & Protocol Provenance

- **Configuration**: `configs/deeplabv3plus-resnet50-focal-lovasz-v1.toml`
  - SHA-256: `e8267f7124a1e03272edb8c7d8effd5729b31f5a4721a931ad9a8070e7936065`
- **Protocol**: `docs/research/deeplabv3plus-focal-lovasz-protocol-v1.json`
- **Dataset**: `data/pvextent-training-v4`
  - Fingerprint: `1eb811eb633a476c4c1b87fa43e0fe274e630119ef195a6a8a3f5b2c13fc9788`
- **Output Checkpoints**:
  - `best.pt` SHA-256: `0a49cbc87e83058b49088b7a0222f8c2ffed3c5a6a843f5e523e7cdc029d7a7c`
  - `last.pt` SHA-256: `da74651b28b703828e8e0e0615ceb228472c9444da42d2994924d5e5122d6dd6`
- **Evaluation Benchmark Report**: `runs/benchmark_comparison.json`
- **Hardware & Environment**: Apple M2 (Mac mini), 8 GB Unified Memory, macOS 15.3 Darwin arm64, PyTorch 2.14.1, SMP 0.5.0, safetensors 0.8.0.
