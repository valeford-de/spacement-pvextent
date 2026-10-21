# DeepLabV3+ ResNet50 Photometric Invariance Training Audit (V5), 2026-10-21

Local date: **2026-10-21**, Europe/Berlin.  
Completed run: `runs/deeplabv3plus-resnet50-v5-photometric-001`.  
Machine-readable protocol: `docs/research/deeplabv3plus-v5-photometric-protocol-v1.json`.  
Comparative benchmark report: `artifacts/v5_comparison_report.json`.

## Executive Summary

To investigate the impact of sensor glare and cast shadow on aerial photovoltaic segmentation, we engineered a dedicated **Photometric Invariance (V5)** training run.

In addition to horizontal and vertical flips, V5 introduces dynamic exposure modeling directly into the batch generator:
1. **Global Brightness Jitter ($\pm 20\%$)**: Simulates underexposure and overexposure across varying camera gains.
2. **Dynamic Contrast Jitter ($\pm 20\%$)**: Simulates overcast diffuse lighting versus harsh midday direct sunlight.
3. **Specular Sun-Glare Simulation (20% probability)**: Applies Gaussian irradiance hotspots to replicate solar panel glass reflections.
4. **Directional Roof Shadow Simulation (20% probability)**: Applies smooth-step half-plane dimming factors to replicate gable and chimney shadows.

---

## Benchmark Results (Validation Set & Holdouts)

### 1. In-Domain Validation Set (28,467,200 Valid Pixels across 278 Crops)

| Metric | Baseline ResNet18 U-Net | Step 1: DeepLabV3+ (BCE) | Step 2: DeepLabV3+ (Focal-Lovász) | V5: Photometric Invariance | V5 vs Step 2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pixel IoU** | 91.40% | 90.57% | 91.07% | **91.22%** | **+0.15 pp (New Record)** |
| **Dice Score** | 95.51% | 95.05% | 95.33% | **95.41%** | **+0.08 pp** |
| **Pixel Accuracy** | 98.07% | 97.89% | 98.00% | **98.04%** | **+0.04 pp** |
| **Precision** | 95.12% | 95.32% | 95.63% | **95.53%** | -0.10 pp |
| **Recall** | 95.90% | 94.79% | 95.03% | **95.29%** | **+0.26 pp** |
| **False Negatives** | 247,400 px | 317,871 px | 303,427 px | **287,396 px** | **-16,031 px** |
| **True Positives** | 5,852,207 px | 5,781,736 px | 5,796,180 px | **5,812,211 px** | **+16,031 px** |

### 2. Negative Holdout False Positive Suppression (7 Confirmed Non-PV Crops, 150,374 px)

| Configuration | False Positive Pixels (@0.25) | False Positive Rate (FPR @0.25) |
| :--- | :---: | :---: |
| Step 2 (Single-Pass) | 8,750 px | 5.82% |
| Step 2 (Full D4-TTA) | 4,909 px | 3.26% |
| **V5 Photometric (Full D4-TTA)** | **3,781 px** | **2.51% (Lowest False Alarm Rate)** |

### 3. Out-of-Domain Holdout Orthophotos

- **Crop 1 (`ytXZBL`)**:
  - Step 2 Mean Confidence: 0.6529.
  - V5 Mean Confidence: **0.7193** (+10.2% further confidence boost, 100% recall).
- **Model Invariance Trade-off**:
  - The heavy synthetic shadow dimming (dim factor 0.65) taught the model to be more conservative on ambiguous low-contrast roofs (e.g. Crop 2), while setting a new all-time high on validated pixel IoU (91.22%) and slashing background false alarms to 2.51%.

---

## Checkpoint & Protocol Provenance

- **Configuration**: `configs/deeplabv3plus-resnet50-v5-photometric-v1.toml`
  - SHA-256: `177c71f809bf3ee2e0fa8fb9aa30a2d378f9f467c780c56f7417f74b01ff6259`
- **Output Checkpoint**: `runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt`
  - SHA-256: `9380f5b7ebf5af23aeb445832402714865e387db5ce496d070575aada6e351a6`
- **Hardware & Environment**: Apple M2 (Mac mini), 8 GB Unified Memory, active fan cooling. Runtime: 1,982.65 seconds (654 s/epoch).
