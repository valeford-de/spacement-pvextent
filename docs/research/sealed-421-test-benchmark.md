# Sealed 421 PV4GER Test Benchmark Audit

**Date**: 2026-10-22  
**Device**: Apple M2 Mac mini (MPS, macOS 15.3, PyTorch 2.14.1)  
**Partition**: 421 reserved PV4GER test crops (43,110,400 pixels), unsealed only for this formal benchmark evaluation.

---

## 1. Executive Summary

This audit records the final benchmark across all model generations and the production ensemble.

```
+---------------------------------------------------------------------------------------------------------+
| Architecture / Generation             | Test IoU  | Test Dice | Precision | Recall    | Test Pixel Acc. |
+---------------------------------------+-----------+-----------+-----------+-----------+-----------------+
| Baseline V1 (ResNet18 U-Net)          | 85.92%    | 92.42%    | 93.98%    | 90.92%    | 97.08%          |
| DeepLabV3+ V4 (Focal+Lovasz)          | 87.87%    | 93.55%    | 93.88%    | 93.21%    | 97.48%          |
| DeepLabV3+ V5 Raw                     | 88.16%    | 93.71%    | 93.89%    | 93.53%    | 97.54%          |
| DeepLabV3+ V5 + D4 TTA + Calibrated   | 87.95%    | 93.59%    | 94.06%    | 93.12%    | 97.50%          |
| DeepLabV3+ V6 + D4 TTA + Calibrated   | 89.07%    | 94.22%    | 92.21%    | 96.32%    | 97.68%          |
| Production Ensemble (V5 + V6 + D4)    | 89.15%    | 94.26%    | 93.61%    | 94.93%    | 97.74%          |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Quantitative Breakthroughs

1. **Missed PV Panels Reduced by 59.5%**:
   - Baseline V1 missed **767,417** PV pixels on the test set.
   - DeepLabV3+ V6 missed only **311,121** PV pixels.
   - Test set recall increased by **+5.40%** to **96.32%**.

2. **Negative Holdout False Positive Suppression**:
   - Evaluated on 7 difficult negative holdout roofs with dormers, skylights, and complex roof tiles (716,800 pixels):
   - Baseline V1 FPR: **17.8%**
   - V6 FPR ($\tau = 0.50$): **0.045%** (only 325 pixels out of 716,800).
   - Achieved **99.955% background specificity**.

3. **Production Ensemble Performance**:
   - Combining V5 (photometric invariance) and V6 (self-trained NRW boundary sensitivity) with $D_4$ dihedral test-time augmentation achieved the project record of **89.15% IoU** and **97.74% Pixel Accuracy**.
