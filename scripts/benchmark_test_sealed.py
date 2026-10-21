# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Formal, sealed one-time evaluation across the 421 reserved PV4GER test crops.

Measures computer vision metrics across model generations:
- Baseline V1 (ResNet18 U-Net)
- DeepLabV3+ V4 (Focal+Lovasz)
- DeepLabV3+ V5 (Photometric Invariance: Glare+Shadow Jitter)
- DeepLabV3+ V6 (Bayesian NRW Self-Training)
- Multi-Model Calibrated Ensemble with Dihedral D4 TTA
"""

import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import build_resnet_unet, normalize_rgb
from pvextent.inference.tta import predict_with_tta
from pvextent.inference.tiler import predict_tiled


def evaluate_dataset_on_model(model, records, dataset_dir, device, tta_mode="none", temperature=1.0, threshold=0.5):
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    valid_pixels_total = 0

    per_crop_metrics = []

    for idx, r in enumerate(records, 1):
        img_p = dataset_dir / r["image"]
        mask_p = dataset_dir / r["mask"]

        img = np.array(Image.open(img_p).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_p))

        # Binary ground truth: 1 for PV, 0 for background, 255 for ignore
        valid_mask = (mask != 255)
        gt_pv = (mask == 1) & valid_mask
        gt_bg = (mask == 0) & valid_mask

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            continue

        if tta_mode == "none":
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            norm_t = normalize_rgb(img_t)
            with torch.no_grad():
                logits = model(norm_t)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        else:
            res = predict_tiled(
                model=model,
                image=img,
                tile_size=400,
                overlap=0.50,
                tta_mode=tta_mode,
                temperature=temperature,
                device=device,
                normalize_fn=normalize_rgb,
            )
            probs = res["probabilities"]

        pred_pv = (probs >= threshold) & valid_mask
        pred_bg = (probs < threshold) & valid_mask

        tp = int((pred_pv & gt_pv).sum())
        fp = int((pred_pv & gt_bg).sum())
        fn = int((pred_bg & gt_pv).sum())
        tn = int((pred_bg & gt_bg).sum())

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn
        valid_pixels_total += n_valid

        denom_iou = tp + fp + fn
        crop_iou = (tp / denom_iou) if denom_iou > 0 else (1.0 if (gt_pv.sum() == 0 and fp == 0) else 0.0)

        per_crop_metrics.append({
            "id": r["id"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "crop_iou": crop_iou,
            "has_pv": bool(gt_pv.sum() > 0)
        })

    iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0
    dice = (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total) if (2.0 * tp_total + fp_total + fn_total) > 0 else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    accuracy = (tp_total + tn_total) / valid_pixels_total if valid_pixels_total > 0 else 0.0
    specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "tn": tn_total,
        "valid_pixels": valid_pixels_total,
        "evaluated_crops": len(per_crop_metrics),
    }


def evaluate_ensemble(models_with_weights, records, dataset_dir, device, tta_mode="d4", temperature=1.0, threshold=0.5):
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    valid_pixels_total = 0

    for idx, r in enumerate(records, 1):
        img_p = dataset_dir / r["image"]
        mask_p = dataset_dir / r["mask"]

        img = np.array(Image.open(img_p).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_p))

        valid_mask = (mask != 255)
        gt_pv = (mask == 1) & valid_mask
        gt_bg = (mask == 0) & valid_mask

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            continue

        ensemble_probs = np.zeros(img.shape[:2], dtype=np.float64)
        for model, weight in models_with_weights:
            res = predict_tiled(
                model=model,
                image=img,
                tile_size=400,
                overlap=0.50,
                tta_mode=tta_mode,
                temperature=temperature,
                device=device,
                normalize_fn=normalize_rgb,
            )
            ensemble_probs += weight * res["probabilities"]

        pred_pv = (ensemble_probs >= threshold) & valid_mask
        pred_bg = (ensemble_probs < threshold) & valid_mask

        tp = int((pred_pv & gt_pv).sum())
        fp = int((pred_pv & gt_bg).sum())
        fn = int((pred_bg & gt_pv).sum())
        tn = int((pred_bg & gt_bg).sum())

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn
        valid_pixels_total += n_valid

    iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0
    dice = (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total) if (2.0 * tp_total + fp_total + fn_total) > 0 else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    accuracy = (tp_total + tn_total) / valid_pixels_total if valid_pixels_total > 0 else 0.0
    specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "tn": tn_total,
        "valid_pixels": valid_pixels_total,
        "evaluated_crops": len(records),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== FORMAL BENCHMARK ON SEALED 421 PV4GER TEST CROPS ({device.upper()}) ===")

    dataset_dir = Path("data/pvextent-training-v5")
    manifest_p = dataset_dir / "manifest.jsonl"

    with open(manifest_p, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    test_records = [r for r in records if r.get("split") == "test"]
    print(f"Loaded {len(test_records)} verified test crops from sealed partition.")
    assert len(test_records) == 421, f"Expected exactly 421 test crops, found {len(test_records)}"

    benchmark_results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "PV4GER-test-sealed-421",
        "crop_count": len(test_records),
        "device": device,
        "evaluations": {},
    }

    # Calibration parameters
    temperature = 0.4068
    optimal_threshold = 0.505

    # 1. Baseline V1 (ResNet18 U-Net)
    v1_ckpt = Path("runs/baseline-resnet18-balanced-v4-001/best.pt")
    if v1_ckpt.exists():
        print("Evaluating Model 1: Baseline V1 (ResNet18 U-Net)...")
        doc_v1 = torch.load(v1_ckpt, map_location="cpu", weights_only=True)
        m_v1 = build_resnet_unet(None)
        m_v1.load_state_dict(doc_v1["model_state_dict"])
        m_v1.to(device).eval()
        res_v1 = evaluate_dataset_on_model(m_v1, test_records, dataset_dir, device, tta_mode="none", threshold=0.5)
        benchmark_results["evaluations"]["Baseline_V1_ResNet18"] = res_v1
        print(f"  -> V1 IoU: {res_v1['iou']*100:.2f}%, Dice: {res_v1['dice']*100:.2f}%, Acc: {res_v1['accuracy']*100:.2f}%")

    # 2. DeepLabV3+ V4 (Focal+Lovasz)
    v4_ckpt = Path("runs/deeplabv3plus-resnet50-focal-lovasz-v4-001/best.pt")
    if v4_ckpt.exists():
        print("Evaluating Model 2: DeepLabV3+ V4 (Focal+Lovasz)...")
        doc_v4 = torch.load(v4_ckpt, map_location="cpu", weights_only=True)
        m_v4 = build_deeplabv3plus_resnet50(None)
        m_v4.load_state_dict(doc_v4["model_state_dict"])
        m_v4.to(device).eval()
        res_v4 = evaluate_dataset_on_model(m_v4, test_records, dataset_dir, device, tta_mode="none", threshold=0.5)
        benchmark_results["evaluations"]["DeepLabV3Plus_V4"] = res_v4
        print(f"  -> V4 IoU: {res_v4['iou']*100:.2f}%, Dice: {res_v4['dice']*100:.2f}%, Acc: {res_v4['accuracy']*100:.2f}%")

    # 3. DeepLabV3+ V5 (Photometric Invariance)
    v5_ckpt = Path("runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt")
    if v5_ckpt.exists():
        print("Evaluating Model 3: DeepLabV3+ V5 (Photometric Invariance)...")
        doc_v5 = torch.load(v5_ckpt, map_location="cpu", weights_only=True)
        m_v5 = build_deeplabv3plus_resnet50(None)
        m_v5.load_state_dict(doc_v5["model_state_dict"])
        m_v5.to(device).eval()

        # Without TTA
        res_v5_raw = evaluate_dataset_on_model(m_v5, test_records, dataset_dir, device, tta_mode="none", threshold=0.5)
        benchmark_results["evaluations"]["DeepLabV3Plus_V5_raw"] = res_v5_raw
        print(f"  -> V5 Raw IoU: {res_v5_raw['iou']*100:.2f}%, Dice: {res_v5_raw['dice']*100:.2f}%")

        # With D4 TTA and Temperature Calibration
        print("Evaluating Model 3 with D4 TTA + Calibration...")
        res_v5_tta = evaluate_dataset_on_model(m_v5, test_records, dataset_dir, device, tta_mode="d4", temperature=temperature, threshold=optimal_threshold)
        benchmark_results["evaluations"]["DeepLabV3Plus_V5_D4_Calibrated"] = res_v5_tta
        print(f"  -> V5 Calibrated D4 IoU: {res_v5_tta['iou']*100:.2f}%, Dice: {res_v5_tta['dice']*100:.2f}%, Acc: {res_v5_tta['accuracy']*100:.2f}%")

    # 4. DeepLabV3+ V6 (Self-Trained NRW Pseudo-Labels)
    v6_ckpt = Path("runs/deeplabv3plus-resnet50-v6-nrw-pseudo-001/best.pt")
    m_v6 = None
    if v6_ckpt.exists():
        print("Evaluating Model 4: DeepLabV3+ V6 (NRW Pseudo-Labels)...")
        doc_v6 = torch.load(v6_ckpt, map_location="cpu", weights_only=True)
        m_v6 = build_deeplabv3plus_resnet50(None)
        m_v6.load_state_dict(doc_v6["model_state_dict"])
        m_v6.to(device).eval()
        res_v6_tta = evaluate_dataset_on_model(m_v6, test_records, dataset_dir, device, tta_mode="d4", temperature=temperature, threshold=optimal_threshold)
        benchmark_results["evaluations"]["DeepLabV3Plus_V6_D4_Calibrated"] = res_v6_tta
        print(f"  -> V6 Calibrated D4 IoU: {res_v6_tta['iou']*100:.2f}%, Dice: {res_v6_tta['dice']*100:.2f}%, Acc: {res_v6_tta['accuracy']*100:.2f}%")

    # 5. Heterogeneous Ensemble (V5 + V6 + V4)
    if v5_ckpt.exists() and v6_ckpt.exists():
        print("Evaluating Calibrated Production Ensemble (V5 + V6 with D4 TTA)...")
        models_weights = [(m_v5, 0.50), (m_v6, 0.50)]
        res_ensemble = evaluate_ensemble(models_weights, test_records, dataset_dir, device, tta_mode="d4", temperature=temperature, threshold=optimal_threshold)
        benchmark_results["evaluations"]["Production_Ensemble_V5_V6_D4"] = res_ensemble
        print(f"  -> Production Ensemble IoU: {res_ensemble['iou']*100:.2f}%, Dice: {res_ensemble['dice']*100:.2f}%, Acc: {res_ensemble['accuracy']*100:.2f}%")

    # Save benchmark JSON
    out_json = Path("runs/benchmark_test_sealed_421.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nBenchmark completed successfully! Saved to {out_json}")


if __name__ == "__main__":
    main()
