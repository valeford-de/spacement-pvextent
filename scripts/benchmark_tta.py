# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Quantitative evaluation of Test-Time Augmentation (TTA) across holdouts."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from pvextent.inference.tta import predict_with_tta
from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import normalize_rgb

CKPT_PATH = "runs/deeplabv3plus-resnet50-focal-lovasz-v4-001/best.pt"

POSITIVE_CROPS = [
    ("Crop 1 (ytXZBL)", "data/positive-holdout-frozen-v1/images/nrw-DENW07AL0000ytXZBL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW07AL0000ytXZBL.png"),
    ("Crop 2 (StB8BL)", "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000StB8BL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW53AL0000StB8BL.png"),
    ("Crop 3 (mo4NBL)", "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000mo4NBL.png", "data/positive-holdout-frozen-v1/masks/nrw-DENW53AL0000mo4NBL.png"),
]

NEGATIVE_MANIFEST = "data/negative-holdout-frozen-v2/manifest.jsonl"
NEGATIVE_DIR = Path("data/negative-holdout-frozen-v2")


def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    model = build_deeplabv3plus_resnet50(None)
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    modes = [("Single-Pass (none)", "none"), ("Flips-TTA (4x)", "flips"), ("Rotations-TTA (4x)", "rot4"), ("Full D4-TTA (8x)", "d4")]
    results = {}

    for mode_label, mode in modes:
        print(f"\n=======================================================")
        print(f"EVALUATING: {mode_label}")
        print(f"=======================================================")

        # 1. Positive Holdout
        tot_pv = 0
        tot_tp_025 = 0
        tot_tp_020 = 0
        crop_stats = {}

        for cname, img_p, mask_p in POSITIVE_CROPS:
            img = Image.open(img_p).convert("RGB")
            mask = Image.open(mask_p)
            arr = np.array(img, dtype=np.float32) / 255.0
            img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            norm_t = normalize_rgb(img_t)

            mask_arr = np.array(mask)
            pv_gt = (mask_arr == 1)
            gt_pv_pixels = int(pv_gt.sum())

            tta_res = predict_with_tta(model, norm_t, mode=mode)
            probs = tta_res["probabilities"].cpu().numpy()

            probs_on_pv = probs[pv_gt]
            mean_p = float(probs_on_pv.mean()) if gt_pv_pixels > 0 else 0.0

            pred_025 = (probs >= 0.25)
            pred_020 = (probs >= 0.20)

            tp_025 = int((pred_025 & pv_gt).sum())
            tp_020 = int((pred_020 & pv_gt).sum())

            rec_025 = tp_025 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0
            rec_020 = tp_020 / gt_pv_pixels if gt_pv_pixels > 0 else 0.0

            tot_pv += gt_pv_pixels
            tot_tp_025 += tp_025
            tot_tp_020 += tp_020

            crop_stats[cname] = {
                "mean_pv_prob": mean_p,
                "recall_025": rec_025,
                "recall_020": rec_020,
                "tp_025": tp_025,
                "tp_020": tp_020,
                "gt_pv_pixels": gt_pv_pixels,
            }
            print(f"  [{cname}] Mean PV Prob: {mean_p:.4f} | Recall@0.25: {rec_025*100:.2f}% ({tp_025}/{gt_pv_pixels} px) | Recall@0.20: {rec_020*100:.2f}%")

        overall_rec_025 = tot_tp_025 / tot_pv if tot_pv > 0 else 0.0
        overall_rec_020 = tot_tp_020 / tot_pv if tot_pv > 0 else 0.0
        print(f"  >> POSITIVE TOTALS: Recall@0.25 = {overall_rec_025*100:.2f}% ({tot_tp_025}/{tot_pv} px) | Recall@0.20 = {overall_rec_020*100:.2f}% ({tot_tp_020}/{tot_pv} px)")

        # 2. Negative Holdout
        total_bg = 0
        total_fp_025 = 0
        total_fp_050 = 0
        crops_with_fp_025 = 0

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

                tta_res = predict_with_tta(model, norm_t, mode=mode)
                probs = tta_res["probabilities"].cpu().numpy()

                pred_025 = (probs >= 0.25)
                pred_050 = (probs >= 0.50)

                fp_025 = int((pred_025 & bg_mask).sum())
                fp_050 = int((pred_050 & bg_mask).sum())

                total_bg += bg_pixels
                total_fp_025 += fp_025
                total_fp_050 += fp_050
                if fp_025 > 0:
                    crops_with_fp_025 += 1

        fpr_025 = total_fp_025 / total_bg if total_bg > 0 else 0.0
        fpr_050 = total_fp_050 / total_bg if total_bg > 0 else 0.0
        print(f"  >> NEGATIVE TOTALS: FP@0.50 = {total_fp_050} (FPR: {fpr_050*100:.4f}%) | FP@0.25 = {total_fp_025} (FPR: {fpr_025*100:.4f}%) | Crops with FP: {crops_with_fp_025}/7")

        results[mode_label] = {
            "mode": mode,
            "positive_holdout": {
                "crops": crop_stats,
                "overall_recall_025": overall_rec_025,
                "overall_recall_020": overall_rec_020,
                "total_tp_025": tot_tp_025,
                "total_tp_020": tot_tp_020,
            },
            "negative_holdout": {
                "fp_050": total_fp_050,
                "fpr_050": fpr_050,
                "fp_025": total_fp_025,
                "fpr_025": fpr_025,
                "crops_with_fp_025": crops_with_fp_025,
            }
        }

    out_file = Path("artifacts/tta_benchmark_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved TTA benchmark results to {out_file}")


if __name__ == "__main__":
    main()
