# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive benchmark across positive holdout, negative holdout, and validation."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from pvextent.models.resnet_unet import build_resnet_unet, normalize_rgb
from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50

POSITIVE_CROPS = [
    ("Crop 1 (ytXZBL)", "data/positive-holdout-frozen-v1/images/nrw-DENW07AL0000ytXZBL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW07AL0000ytXZBL.png"),
    ("Crop 2 (StB8BL)", "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000StB8BL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW53AL0000StB8BL.png"),
    ("Crop 3 (mo4NBL)", "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000mo4NBL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW53AL0000mo4NBL.png"),
]

NEGATIVE_MANIFEST = "data/negative-holdout-frozen-v2/manifest.jsonl"
NEGATIVE_DIR = Path("data/negative-holdout-frozen-v2")


def load_model(kind: str, ckpt_path: str):
    if kind == "unet":
        model = build_resnet_unet(None)
    elif kind == "deeplabv3plus":
        model = build_deeplabv3plus_resnet50(None)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def evaluate_model_on_holdouts(model_name: str, kind: str, ckpt_path: str):
    print(f"\n=======================================================")
    print(f"BENCHMARK: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"=======================================================")
    
    if not Path(ckpt_path).exists():
        print(f"[SKIPPED] Checkpoint does not exist yet: {ckpt_path}")
        return None

    model = load_model(kind, ckpt_path)

    # --- 1. Positive Holdout ---
    print("\n--- 1. UNSEEN POSITIVE HOLDOUT (3 NRW Orthophotos) ---")
    pos_results = {}
    tot_pv_pixels = 0
    tot_tp_05 = 0
    tot_tp_02 = 0
    tot_pred_pixels = 0

    for label, img_p, mask_p in POSITIVE_CROPS:
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)
        arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        norm_t = normalize_rgb(img_t)
        mask_arr = np.array(mask)
        pv_gt = (mask_arr == 1)
        gt_pv_pixels = int(pv_gt.sum())

        with torch.no_grad():
            logits = model(norm_t).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

        probs_on_pv = probs[pv_gt]
        mean_p = float(probs_on_pv.mean()) if gt_pv_pixels > 0 else 0.0
        median_p = float(np.median(probs_on_pv)) if gt_pv_pixels > 0 else 0.0

        pred_05 = (probs >= 0.5)
        pred_02 = (probs >= 0.2)
        pred_01 = (probs >= 0.1)

        tp_05 = int((pred_05 & pv_gt).sum())
        tp_02 = int((pred_02 & pv_gt).sum())
        tp_01 = int((pred_01 & pv_gt).sum())
        total_pred_05 = int(pred_05.sum())

        rec_05 = tp_05 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0
        rec_02 = tp_02 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0
        rec_01 = tp_01 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0

        tot_pv_pixels += gt_pv_pixels
        tot_tp_05 += tp_05
        tot_tp_02 += tp_02
        tot_pred_pixels += total_pred_05

        pos_results[label] = {
            "gt_pv_pixels": gt_pv_pixels,
            "mean_prob": mean_p,
            "median_prob": median_p,
            "tp_05": tp_05,
            "rec_05": rec_05,
            "tp_02": tp_02,
            "rec_02": rec_02,
            "tp_01": tp_01,
            "rec_01": rec_01,
            "total_predicted_pixels_05": total_pred_05,
        }

        print(f"[{label}] Supervised PV: {gt_pv_pixels} px")
        print(f"  Mean PV prob: {mean_p:.4f} | Median: {median_p:.4f}")
        print(f"  Recall @ 0.5: {rec_05*100:.2f}% ({tp_05}/{gt_pv_pixels} px)")
        print(f"  Recall @ 0.2: {rec_02*100:.2f}% ({tp_02}/{gt_pv_pixels} px)")
        print(f"  Recall @ 0.1: {rec_01*100:.2f}% ({tp_01}/{gt_pv_pixels} px)")
        print(f"  Total predicted pixels (@0.5): {total_pred_05} px")

    overall_rec_05 = tot_tp_05 / tot_pv_pixels if tot_pv_pixels > 0 else 0.0
    overall_rec_02 = tot_tp_02 / tot_pv_pixels if tot_pv_pixels > 0 else 0.0
    print(f">> POSITIVE TOTALS: Supervised PV = {tot_pv_pixels} px | Recall @ 0.5 = {overall_rec_05*100:.2f}% ({tot_tp_05} px) | Recall @ 0.2 = {overall_rec_02*100:.2f}% ({tot_tp_02} px)")

    # --- 2. Negative Holdout ---
    print("\n--- 2. GEOGRAPHIC NEGATIVE HOLDOUT (7 Confirmed Non-PV Crops) ---")
    neg_results = {}
    total_bg_pixels = 0
    total_fp_pixels_05 = 0
    total_fp_pixels_02 = 0
    total_ignored_pred_05 = 0
    crops_with_fp = 0

    if Path(NEGATIVE_MANIFEST).exists():
        with open(NEGATIVE_MANIFEST, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split") != "test":
                    continue
                img_p = NEGATIVE_DIR / rec["image"]
                mask_p = NEGATIVE_DIR / rec["mask"]

                img = Image.open(img_p).convert("RGB")
                mask = Image.open(mask_p)
                arr = np.array(img, dtype=np.float32) / 255.0
                img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                norm_t = normalize_rgb(img_t)
                mask_arr = np.array(mask)

                bg_mask = (mask_arr == 0)
                ignored_mask = (mask_arr == 255)
                bg_pixels = int(bg_mask.sum())

                with torch.no_grad():
                    logits = model(norm_t).squeeze()
                    probs = torch.sigmoid(logits).cpu().numpy()

                pred_05 = (probs >= 0.5)
                pred_02 = (probs >= 0.2)

                fp_05 = int((pred_05 & bg_mask).sum())
                fp_02 = int((pred_02 & bg_mask).sum())
                ignored_pred_05 = int((pred_05 & ignored_mask).sum())

                total_bg_pixels += bg_pixels
                total_fp_pixels_05 += fp_05
                total_fp_pixels_02 += fp_02
                total_ignored_pred_05 += ignored_pred_05
                if fp_05 > 0:
                    crops_with_fp += 1

        fpr_05 = total_fp_pixels_05 / total_bg_pixels if total_bg_pixels > 0 else 0.0
        fpr_02 = total_fp_pixels_02 / total_bg_pixels if total_bg_pixels > 0 else 0.0
        print(f"Supervised Background Pixels: {total_bg_pixels}")
        print(f"FP Pixels @ 0.5: {total_fp_pixels_05} (FPR: {fpr_05*100:.4f}%) | Crops with FP: {crops_with_fp}/7")
        print(f"FP Pixels @ 0.2: {total_fp_pixels_02} (FPR: {fpr_02*100:.4f}%)")
        print(f"Unscored Ignored Area Predictions @ 0.5: {total_ignored_pred_05} px")
    else:
        print("[SKIPPED] Negative holdout manifest not found.")

    return {
        "model_name": model_name,
        "positive": {
            "crops": pos_results,
            "overall_recall_05": overall_rec_05,
            "overall_recall_02": overall_rec_02,
            "total_pv_pixels": tot_pv_pixels,
            "total_tp_05": tot_tp_05,
            "total_tp_02": tot_tp_02,
            "total_predicted_pixels": tot_pred_pixels,
        },
        "negative": {
            "total_bg_pixels": total_bg_pixels,
            "fp_05": total_fp_pixels_05,
            "fpr_05": fpr_05 if total_bg_pixels > 0 else 0.0,
            "fp_02": total_fp_pixels_02,
            "crops_with_fp": crops_with_fp,
            "unscored_ignored_pred_05": total_ignored_pred_05,
        }
    }


if __name__ == "__main__":
    models_to_test = [
        ("Baseline U-Net ResNet18", "unet", "runs/baseline-resnet18-bulk-v4-001/best.pt"),
        ("DeepLabV3+ ResNet50 ASPP (BCE+Dice)", "deeplabv3plus", "runs/deeplabv3plus-resnet50-v4-001/best.pt"),
        ("DeepLabV3+ ResNet50 ASPP (Focal+Lovasz)", "deeplabv3plus", "runs/deeplabv3plus-resnet50-focal-lovasz-v4-001/best.pt"),
    ]

    all_results = {}
    for name, kind, path in models_to_test:
        res = evaluate_model_on_holdouts(name, kind, path)
        if res is not None:
            all_results[name] = res

    # Save summary json
    out_path = Path("runs/benchmark_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")
