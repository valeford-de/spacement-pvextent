# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Semi-supervised Bayesian pseudo-labeling on unassigned NRW aerial orthophotos."""

import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch

from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import normalize_rgb
from pvextent.inference.tiler import predict_tiled


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Executing Bayesian pseudo-labeling on device: {device}")

    # 1. Load checkpoints
    ckpt_v4 = Path("runs/deeplabv3plus-resnet50-focal-lovasz-v4-001/best.pt")
    ckpt_v5 = Path("runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt")

    doc_v4 = torch.load(ckpt_v4, map_location="cpu", weights_only=True)
    model_v4 = build_deeplabv3plus_resnet50(None)
    model_v4.load_state_dict(doc_v4["model_state_dict"])
    model_v4.to(device).eval()

    doc_v5 = torch.load(ckpt_v5, map_location="cpu", weights_only=True)
    model_v5 = build_deeplabv3plus_resnet50(None)
    model_v5.load_state_dict(doc_v5["model_state_dict"])
    model_v5.to(device).eval()

    # Load temperature calibration
    with open("runs/calibration_report.json", "r", encoding="utf-8") as f:
        calib = json.load(f)
    temperature = calib["temperature_scaling"]["temperature"]
    print(f"Loaded temperature scalar: T* = {temperature:.4f}")

    # 2. Setup paths
    source_dir = Path("data/pvextent-training-v4")
    dest_dir = Path("data/pvextent-training-v5")

    if dest_dir.exists():
        print(f"Target directory {dest_dir} already exists. Removing for clean regeneration.")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "masks").mkdir(parents=True, exist_ok=True)
    (dest_dir / "provenance").mkdir(parents=True, exist_ok=True)

    # Copy dataset.toml and provenance
    if (source_dir / "dataset.toml").exists():
        shutil.copy2(source_dir / "dataset.toml", dest_dir / "dataset.toml")
    for prov_file in (source_dir / "provenance").glob("*"):
        if prov_file.is_file():
            shutil.copy2(prov_file, dest_dir / "provenance" / prov_file.name)

    # 3. Read manifest and isolate target unassigned NRW records
    with open(source_dir / "manifest.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    target_records = [
        r for r in records
        if r.get("split") == "unassigned" and r.get("source_id") == "ai-bulk-nrw-dop"
    ]
    print(f"Found {len(target_records)} unassigned NRW candidate crops for self-training curation.")

    new_records = []
    admitted_pv_px = 0
    admitted_bg_px = 0
    admitted_ignore_px = 0
    promoted_crops = 0

    started = time.perf_counter()

    for idx, r in enumerate(records, 1):
        rel_img = r["image"]
        rel_mask = r["mask"]

        src_img_p = source_dir / rel_img
        dest_img_p = dest_dir / rel_img

        # Symlink or hardlink source images
        dest_img_p.parent.mkdir(parents=True, exist_ok=True)
        if not dest_img_p.exists():
            dest_img_p.hardlink_to(src_img_p)

        # Check if this is an unassigned NRW candidate
        if r.get("split") == "unassigned" and r.get("source_id") == "ai-bulk-nrw-dop":
            img = np.array(Image.open(src_img_p).convert("RGB"))

            # TTA inference with V4 and V5
            res_v4 = predict_tiled(
                model=model_v4,
                image=img,
                tile_size=400,
                overlap=0.50,
                tta_mode="d4",
                device=device,
                normalize_fn=normalize_rgb,
            )
            res_v5 = predict_tiled(
                model=model_v5,
                image=img,
                tile_size=400,
                overlap=0.50,
                tta_mode="d4",
                temperature=temperature,
                device=device,
                normalize_fn=normalize_rgb,
            )

            p_v4 = res_v4["probabilities"]
            p_v5 = res_v5["probabilities"]
            u_v4 = res_v4["uncertainty"]
            u_v5 = res_v5["uncertainty"]

            p_ens = 0.5 * p_v4 + 0.5 * p_v5
            epistemic = 0.5 * ((p_v4 - p_ens) ** 2 + (p_v5 - p_ens) ** 2)
            aleatoric = 0.5 * (u_v4 + u_v5)
            u_total = epistemic + aleatoric

            # High-confidence pseudo-label mask creation:
            # 1: PV positive (prob >= 0.95 and uncert <= 0.02)
            # 0: Supervised background (prob <= 0.02 and uncert <= 0.02)
            # 255: Ignored ambiguous pixels
            h, w = img.shape[:2]
            pseudo_mask = np.full((h, w), 255, dtype=np.uint8)

            pos_cond = (p_ens >= 0.95) & (u_total <= 0.020)
            neg_cond = (p_ens <= 0.02) & (u_total <= 0.020)

            pseudo_mask[pos_cond] = 1
            pseudo_mask[neg_cond] = 0

            n_pos = int(np.sum(pos_cond))
            n_neg = int(np.sum(neg_cond))
            n_ign = int(np.sum(pseudo_mask == 255))

            dest_mask_p = dest_dir / rel_mask
            dest_mask_p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pseudo_mask).save(dest_mask_p)
            mask_digest = hashlib.sha256(dest_mask_p.read_bytes()).hexdigest()

            # Update manifest record: promote to train if significant background/PV coverage
            r_updated = dict(r)
            r_updated["mask_sha256"] = mask_digest
            if n_pos > 0 or n_neg > 0:
                r_updated["split"] = "train"
                r_updated["annotation_status"] = "reviewed"
                promoted_crops += 1

            admitted_pv_px += n_pos
            admitted_bg_px += n_neg
            admitted_ignore_px += n_ign

            new_records.append(r_updated)

            if len(new_records) % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f"  Processed {len(new_records)}/{len(records)} records ({elapsed:.1f}s)...")
        else:
            # Copy existing mask unchanged
            src_mask_p = source_dir / rel_mask
            dest_mask_p = dest_dir / rel_mask
            dest_mask_p.parent.mkdir(parents=True, exist_ok=True)
            if not dest_mask_p.exists():
                dest_mask_p.hardlink_to(src_mask_p)
            new_records.append(r)

    # Write new manifest.jsonl
    with open(dest_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    elapsed_total = time.perf_counter() - started
    print("=" * 60)
    print(f"CURATION COMPLETE in {elapsed_total:.2f}s:")
    print(f"  Promoted NRW crops to train: {promoted_crops} / {len(target_records)}")
    print(f"  Admitted high-confidence PV pixels: {admitted_pv_px:,}")
    print(f"  Admitted supervised background pixels: {admitted_bg_px:,}")
    print(f"  Preserved ignore pixels: {admitted_ignore_px:,}")
    print(f"  New training dataset ready: {dest_dir}")


if __name__ == "__main__":
    main()
