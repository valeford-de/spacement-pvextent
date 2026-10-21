# Probability Calibration & Temperature Scaling Audit

**Project:** `valeford-de/spacement-pvextent`  
**Model Architecture:** DeepLabV3+ ResNet-50 ASPP Photometric Invariance V5  
**Checkpoint:** `runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt`  
**Checkpoint SHA-256:** `9380f5b7ebf5af23aeb445832402714865e387db5ce496d070575aada6e351a6`  
**Dataset Split:** Strictly isolated in-domain validation split (`data/pvextent-training-v4`, `split="val"`, 278 crops, 28,467,200 valid pixels)  
**Hardware Accelerator:** Apple M2 Silicon MPS  

---

## 1. Executive Summary

Standard deep learning segmentation models trained with loss functions like Focal Loss and Lovász Hinge surrogate frequently suffer from **probability miscalibration**. While they excel at ranking and separating hard boundaries, their unscaled sigmoid probabilities do not represent true Bayesian posterior probabilities $P(Y=1 \mid X)$.

In operational geospatial analysis, calibrated probabilities are strictly mandatory:
1. When a model predicts $p = 0.90$, exactly 90 out of 100 pixels must empirically contain photovoltaic modules.
2. Downstream physical solvers (shadow simulation, CAD vectorization, financial yield projection, grid feeder sizing) rely on calibrated confidence thresholds to prevent false capital expenditure allocations.

This audit establishes:
- Validation calibration across **28,467,200 supervised validation pixels** (6,099,607 PV-positive pixels).
- Optimal scalar **Temperature Scaling ($T^* = 0.4068$)** and **Platt Scaling ($a^* = 2.3983, b^* = -0.1618$)** parameters fitted via L-BFGS.
- **Expected Calibration Error (ECE)** dropped from **8.75% down to 0.47%** (**-94.59% error reduction**).
- **Negative Log-Likelihood (NLL / Log Loss)** dropped from **0.13374 down to 0.05858** (**-56.19% reduction**).
- **Bayes-Optimal Decision Thresholds** $\tau^*_{\text{IoU}} = 0.505$ (Max IoU = **91.22%**, Dice = **95.41%**), eliminating ad-hoc threshold guessing.

---

## 2. Mathematical Calibration Framework

### 2.1 Reliability Diagrams & Expected Calibration Error (ECE)

Let predicted probability $p_i \in [0, 1]$ and ground truth $y_i \in \{0, 1\}$ for each pixel $i \in \{1, \dots, N\}$.
We partition the confidence interval $[0, 1]$ into $M = 15$ equal-width bins $B_m = \left(\frac{m-1}{M}, \frac{m}{M}\right]$.

For each bin $B_m$:
- **Confidence:** $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$
- **Accuracy (Empirical Precision):** $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$

The **Expected Calibration Error (ECE)** is the bin-count-weighted difference between empirical accuracy and predicted confidence:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

The **Maximum Calibration Error (MCE)** captures worst-case bin deviation:
$$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 2.2 Temperature Scaling

Given raw unnormalized network logits $z_i \in \mathbb{R}$, temperature scaling introduces a strictly positive scalar $T > 0$ such that:
$$p_i(T) = \sigma\left(\frac{z_i}{T}\right) = \frac{1}{1 + e^{-z_i / T}}$$

The parameter $T^*$ is learned by minimizing the Negative Log-Likelihood (Binary Cross-Entropy) over all valid pixels:
$$T^* = \arg\min_{T > 0} \left[ - \frac{1}{N} \sum_{i=1}^N \left( y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log \left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right) \right]$$

Crucially, because temperature scaling is a strictly monotonic transformation of $z$, it preserves the ROC curve, PR curve, and pixel rankings while aligning probabilities to the real world.

### 2.3 Why $T^* = 0.4068 < 1$? (Focal + Lovász Logit Compression)

Unlike Cross-Entropy models which are notoriously overconfident ($T > 1$), models trained with **Focal Loss + Lovász Hinge** penalize extreme confident errors and suppress easy negative gradients. Consequently, the network produces moderate logit magnitudes ($|z| \in [1, 3]$) instead of saturated logits ($|z| > 10$).
Because the raw logits were mildly compressed towards the decision boundary, the raw sigmoid probabilities understated the model's true accuracy.
Dividing by $T^* \approx 0.4068$ acts as a logit amplifier ($z / 0.4068 \approx 2.458 \times z$), correctly mapping high-confidence detections to $p > 0.95$ and background non-PV slate to $p < 0.01$.

---

## 3. Empirical Results

### 3.1 Global Metric Comparison

| Calibration Method | Temperature $T$ / Parameters $(a, b)$ | ECE | MCE | Negative Log-Likelihood (NLL) | Brier Score | ECE Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Uncalibrated** | $T = 1.000$ | **8.75%** | 15.83% | 0.13374 | 0.02494 | Baseline |
| **Temperature Scaled** | **$T^* = 0.4068$** | **0.47%** | **9.44%** | **0.05858** | **0.01537** | **-94.59%** |
| **Affine Platt Scaled** | $a^* = 2.3983, b^* = -0.1618$ | **0.39%** | **7.26%** | **0.05841** | **0.01532** | **-95.53%** |

Both Temperature Scaling and Platt Scaling achieve near-identical reductions, dropping ECE below **0.5%**. Temperature scaling is selected as the production standard because it preserves the zero-crossing decision boundary ($z=0 \iff p=0.5$).

### 3.2 Reliability Diagram Bin Analysis (15 Equal-Width Bins)

#### Raw Model Bins:
- Bin $[0.133, 0.200]$: Mean Conf = 16.02%, Empirical Accuracy = 1.24% ($\Delta = +14.78\%$)
- Bin $[0.733, 0.800]$: Mean Conf = 77.12%, Empirical Accuracy = 91.33% ($\Delta = -14.21\%$)
- Bin $[0.800, 0.867]$: Mean Conf = 84.23%, Empirical Accuracy = 98.25% ($\Delta = -14.01\%$)

#### Calibrated Model Bins ($T^* = 0.4068$):
- Bin $[0.000, 0.067]$: Count = 21,655,339 px, Mean Conf = 0.65%, Empirical Accuracy = 0.48% ($\Delta = \mathbf{0.16\%}$)
- Bin $[0.467, 0.533]$: Count = 60,650 px, Mean Conf = 49.99%, Empirical Accuracy = 50.15% ($\Delta = \mathbf{0.16\%}$)
- Bin $[0.933, 1.000]$: Count = 5,696,555 px, Mean Conf = 99.46%, Empirical Accuracy = 99.47% ($\Delta = \mathbf{0.01\%}$)

The calibrated model displays virtually perfect diagonal calibration across the low, middle, and high confidence regimes.

---

## 4. Bayes-Optimal Threshold Grid Search

Evaluating decision thresholds $\tau \in [0.01, 0.99]$ across the calibrated validation split demonstrates remarkable stability:

| Threshold $\tau$ | Validation IoU | Validation F1 / Dice | Precision | Recall | Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0.250 | 0.8924 | 0.9431 | 0.9080 | 0.9811 | High Recall Sensitivity Mode |
| 0.350 | 0.9048 | 0.9500 | 0.9304 | 0.9705 | Balanced Conservative |
| 0.455 | 0.9118 | 0.9539 | 0.9483 | 0.9596 | Near-optimal trade-off |
| **0.485** | **0.9122** | **0.9541** | **0.9530** | **0.9551** | **Balanced Precision / Recall Point** |
| **0.505** | **0.9122** | **0.9541** | **0.9557** | **0.9525** | **Bayes-Optimal Global Max IoU** |
| 0.604 | 0.9116 | 0.9538 | 0.9620 | 0.9457 | High Precision Mode |
| 0.703 | 0.9086 | 0.9521 | 0.9682 | 0.9365 | Clutter Rejection Mode |

Because the model is calibrated, the optimal decision threshold naturally converges to **$\tau^* = 0.505$** (essentially $\tau = 0.50$), completely validating the statistical coherence of the learned representations.

---

## 5. Command-Line Interface (CLI) Integration

### 5.1 Calibrate Any Checkpoint
To run calibration across the validation split and output full JSON diagnostics:
```bash
pvextent evaluate calibrate data/pvextent-training-v4 runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt runs/calibration/ --device mps
```

### 5.2 Vectorize with Calibrated Inference & D4-TTA
To apply temperature scaling and optimal thresholding automatically during CAD vectorization:
```bash
pvextent vectorize input.png output_dir/ \
  --checkpoint runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt \
  --calibrate \
  --tta d4 \
  --gsd 0.10
```
This automatically applies $T^* = 0.407$, resolves the optimal threshold $\tau^* = 0.505$, executes 8-fold dihedral ensembling, computes physical module metrics, and exports EPSG:25832 GeoJSON.
