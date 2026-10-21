# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive head-to-head evaluation: Step 2 Focal+Lovasz vs Step 4 Photometric V5."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from pvextent.inference.tta import predict_with_tta
from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import normalize_rgb

MODELS = [
    ("Step 2 (Focal+Lovasz)", "runs/deeplabv3plus-resnet50-focal-lovasz-v4-001/best.pt"),
    ("V5 (Photometric Glare+Shadow)", "runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt"),
]

POSITIVE_CROPS = [
    ("Crop 1 (ytXZBL)", "data/positive-holdout-frozen-v1/images/nrw-DENW07AL0000ytXZBL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW07AL0000ytXZBL.png"),
    ("Crop 2 (StB8BL)", "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000StB8BL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW53AL0000StB8BL.png"),
    ("Crop 3 (mo4NBL)", "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000mo4NBL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW53AL0000mo4NBL.png"),
]

NEGATIVE_MANIFEST = "data/negative-holdout-frozen-v2/manifest.jsonl"
NEGATIVE_DIR = Path("data/negative-holdout-frozen-v2")


def evaluate(model, tta_mode):
    # Positive
    tot_pv = 0
    tot_tp_025 = 0
    tot_tp_020 = 0
    crops = {}
    for cname, img_p, mask_p in POSITIVE_CROPS:
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p)
        arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        norm_t = normalize_rgb(img_t)

        mask_arr = np.array(mask)
        pv_gt = (mask_arr == 1)
        gt_pv_pixels = int(pv_gt.sum())

        res = predict_with_tta(model, norm_t, mode=tta_mode)
        probs = res["probabilities"].cpu().numpy()

        probs_on_pv = probs[pv_gt]
        mean_p = float(probs_on_pv.mean()) if gt_pv_pixels > 0 else 0.0

        tp_025 = int(((probs >= 0.25) & pv_gt).sum())
        tp_020 = int(((probs >= 0.20) & pv_gt).sum())

        rec_025 = tp_025 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0
        rec_020 = tp_020 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0

        tot_pv += gt_pv_pixels
        tot_tp_025 += tp_025
        tot_tp_020 += tp_020
        crops[cname] = {
            "mean_pv_prob": mean_p,
            "tp_025": tp_025,
            "recall_025": rec_025,
            "tp_020": tp_020,
            "recall_020": rec_020,
            "gt_pv": gt_pv_pixels,
        }

    # Negative
    total_bg = 0
    total_fp_025 = 0
    total_fp_050 = 0
    crops_with_fp = 0

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
            bg_pixels = int(bg_mask.sum())

            res = predict_with_tta(model, norm_t, mode=tta_mode)
            probs = res["probabilities"].cpu().numpy()

            fp_025 = int(((probs >= 0.25) & bg_mask).sum())
            fp_050 = int(((probs >= 0.50) & bg_mask).sum())

            total_bg += bg_pixels
            total_fp_025 += fp_025
            total_fp_050 += fp_050
            if fp_025 > 0:
                crops_with_fp += 1

    return {
        "positive": {
            "overall_recall_025": tot_tp_025 / tot_pv,
            "overall_recall_020": tot_tp_020 / tot_pv,
            "total_tp_025": tot_tp_025,
            "total_tp_020": tot_tp_020,
            "crops": crops,
        },
        "negative": {
            "fp_050": total_fp_050,
            "fpr_050": total_fp_050 / total_bg,
            "fp_025": total_fp_025,
            "fpr_025": total_fp_025 / total_bg,
            "crops_with_fp_025": crops_with_fp,
        }
    }


def main():
    comparison = {}
    for mname, ckpt in MODELS:
        print(f"\n==========================================")
        print(f"MODEL: {mname}")
        print(f"==========================================")
        model = build_deeplabv3plus_resnet50(None)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
        model.load_state_dict(sd)
        model.eval()

        for tta in ["none", "d4"]:
            key = f"{mname} ({'Single-Pass' if tta=='none' else 'Full D4-TTA'})"
            print(f"--- Running {key} ---")
            res = evaluate(model, tta)
            comparison[key] = res
            p = res["positive"]
            n = res["negative"]
            print(f"  Holdout Recall@0.25: {p['overall_recall_025']*100:.2f}% | Recall@0.20: {p['overall_recall_020']*100:.2f}%")
            for cname, cdata in p["crops"].items():
                print(f"    {cname}: Rec@0.25={cdata['recall_025']*100:.2f}% ({cdata['tp_025']}/{cdata['gt_pv']}), MeanP={cdata['mean_pv_prob']:.4f}")
            print(f"  Negative FP@0.25: {n['fp_025']} (FPR: {n['fpr_025']*100:.2f}%) | FP@0.50: {n['fp_050']}")

    out = Path("artifacts/v5_comparison_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
