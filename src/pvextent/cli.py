# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry points for standalone PVExtent workflows."""

import argparse
import json
from pathlib import Path
import sys

from pvextent import __version__
from pvextent.data import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pvextent")
    parser.add_argument("--version", action="version", version=f"PVExtent {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("monitor", help="Show read-only live training and mask-check progress on localhost")
    monitor.add_argument("--repo-dir", default=".")
    monitor.add_argument("--port", type=int, default=0)
    data = commands.add_parser("data", help="Dataset preparation and validation")
    actions = data.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate", help="Validate a dataset directory offline")
    validate.add_argument("directory", help="Directory containing dataset.toml and manifest.jsonl")
    validate.add_argument("--json", action="store_true", help="Write a machine-readable JSON report")
    importer = actions.add_parser("import-pv4ger-seg", help="Import pinned GEO-Bench PV4GER segmentation data")
    importer.add_argument("source_dir", help="Cache directory for the four pinned upstream files")
    importer.add_argument("output_dir", help="New dataset directory; must not already exist")
    importer.add_argument("--download", action="store_true", help="Download missing source files from Zenodo")
    importer.add_argument("--limit", type=int, help="Convert only the first N sorted IDs for a smoke run")
    importer.add_argument("--json", action="store_true", help="Write a machine-readable import report")
    split = actions.add_parser("split", help="Freeze geographic blocks with a buffer between assigned splits")
    split.add_argument("source_dir", help="Unassigned dataset with provenance/samples.jsonl centres")
    split.add_argument("output_dir", help="New immutable split snapshot")
    split.add_argument("--seed", type=int, default=20261020)
    split.add_argument("--block-size-m", type=float, default=10000.0)
    split.add_argument("--buffer-m", type=float, default=1000.0)
    split.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    split.add_argument("--json", action="store_true")
    review = actions.add_parser("review", help="Export a local visual review bundle; no automatic label changes")
    review.add_argument("source_dir", help="Dataset to inspect")
    review.add_argument("output_dir", help="New self-contained review directory")
    review.add_argument("--json", action="store_true")
    candidates = actions.add_parser("fetch-nrw-candidates", help="Download a bounded official NRW roof review pool")
    candidates.add_argument("source_dir", help="Existing dataset whose locations must remain geographically separate")
    candidates.add_argument("output_dir", help="New unreviewed candidate dataset; masks contain only ignore pixels")
    candidates.add_argument("--limit", type=int, default=24)
    candidates.add_argument("--json", action="store_true")
    holdout = actions.add_parser("fetch-negative-holdout", help="Acquire a spatially separate review pool under a frozen protocol")
    holdout.add_argument("output_dir")
    holdout.add_argument("--exclude", action="append", required=True, dest="exclude_dirs")
    holdout.add_argument("--protocol", required=True)
    holdout.add_argument("--json", action="store_true")
    training_pool = actions.add_parser("fetch-training-candidates", help="Acquire an unlabelled NRW training-review pool separate from reserved data")
    training_pool.add_argument("output_dir")
    training_pool.add_argument("--exclude", action="append", required=True, dest="exclude_dirs")
    training_pool.add_argument("--protocol", required=True)
    training_pool.add_argument("--json", action="store_true")
    bulk = actions.add_parser("fetch-training-bulk", help="Acquire a resumable, bounded NRW training-review collection")
    bulk.add_argument("training_dir", help="Existing training snapshot; only original train pixels may be read")
    bulk.add_argument("output_dir", help="New unlabelled dataset directory")
    bulk.add_argument("--protected", action="append", required=True, dest="protected_dirs")
    bulk.add_argument("--identity-only", action="append", default=[], dest="identity_only_dirs")
    bulk.add_argument("--protocol", required=True)
    bulk.add_argument("--cache-dir", required=True, help="Persistent verified response cache for restarts")
    bulk.add_argument("--allow-partial", action="store_true", help="Explicitly publish below-quota results")
    bulk.add_argument("--json", action="store_true")
    queue = actions.add_parser("review-queue", help="Prepare two blind review lanes without opening images")
    queue.add_argument("dataset_dir")
    queue.add_argument("output_dir", help="New review queue directory")
    queue.add_argument("--batch-size", type=int, default=25)
    queue.add_argument("--seed", type=int, default=20261021)
    queue.add_argument("--json", action="store_true")
    proposals = actions.add_parser("prepare-mask-proposals", help="Freeze visual draft polygons for two independent mask checks")
    proposals.add_argument("dataset_dir")
    proposals.add_argument("queue_dir")
    proposals.add_argument("draft_dir")
    proposals.add_argument("output_dir")
    proposals.add_argument("--protocol", required=True)
    proposals.add_argument("--json", action="store_true")
    bulk_curate = actions.add_parser("curate-training-bulk", help="Curate conservative masks from two complete blind bulk reviews")
    bulk_curate.add_argument("dataset_dir")
    bulk_curate.add_argument("queue_dir")
    bulk_curate.add_argument("review_dir")
    bulk_curate.add_argument("output_dir")
    bulk_curate.add_argument("--protocol", required=True)
    bulk_curate.add_argument("--dataset-version", required=True)
    bulk_curate.add_argument("--allow-ai-review", action="store_true")
    bulk_curate.add_argument("--json", action="store_true")
    bulk_augment = actions.add_parser("augment-training-bulk", help="Add curated bulk masks while preserving frozen evaluation records")
    bulk_augment.add_argument("base_dir")
    bulk_augment.add_argument("curated_dir")
    bulk_augment.add_argument("output_dir")
    bulk_augment.add_argument("--protected", action="append", required=True, dest="protected_dirs")
    bulk_augment.add_argument("--protocol", required=True)
    bulk_augment.add_argument("--dataset-version", required=True)
    bulk_augment.add_argument("--json", action="store_true")
    quarantine = actions.add_parser("quarantine-training-quality", help="Exclude disputed new bulk TRAIN masks under a frozen quality protocol")
    quarantine.add_argument("source_dir")
    quarantine.add_argument("base_dir")
    quarantine.add_argument("output_dir")
    quarantine.add_argument("--protocol", required=True)
    quarantine.add_argument("--evidence", required=True)
    quarantine.add_argument("--dataset-version", required=True)
    quarantine.add_argument("--json", action="store_true")
    freeze = actions.add_parser("freeze-negative-holdout", help="Reserve reviewed negatives for evaluation before predictions")
    freeze.add_argument("curated_dir")
    freeze.add_argument("candidate_dir")
    freeze.add_argument("output_dir")
    freeze.add_argument("--exclude", action="append", required=True, dest="exclude_dirs")
    freeze.add_argument("--protocol", required=True)
    freeze.add_argument("--acquisition-protocol", required=True)
    freeze.add_argument("--json", action="store_true")
    positives = actions.add_parser("freeze-positive-holdout", help="Freeze two blind positive reviews on the untouched cohort complement")
    positives.add_argument("candidate_dir")
    positives.add_argument("negative_holdout_dir")
    positives.add_argument("output_dir")
    positives.add_argument("--exclude", action="append", required=True, dest="exclude_dirs")
    positives.add_argument("--review", action="append", required=True, dest="review_paths")
    positives.add_argument("--protocol", required=True)
    positives.add_argument("--json", action="store_true")
    curate = actions.add_parser("curate-negatives", help="Build a versioned negative collection from two explicit reviews")
    curate.add_argument("output_dir")
    curate.add_argument("--source", action="append", required=True, dest="source_dirs")
    curate.add_argument("--review", action="append", required=True, dest="review_paths")
    curate.add_argument("--dataset-version", required=True)
    curate.add_argument("--allow-ai-review", action="store_true", help="Explicitly permit documented AI reviews for experimental labels")
    curate.add_argument("--json", action="store_true")
    augment = actions.add_parser("augment-training", help="Add curated negatives while preserving a frozen holdout")
    augment.add_argument("base_dir")
    augment.add_argument("negative_dir")
    augment.add_argument("output_dir")
    augment.add_argument("--dataset-version", required=True)
    augment.add_argument("--buffer-m", type=float, default=1000.0)
    augment.add_argument("--json", action="store_true")
    train = commands.add_parser("train", help="Standalone training workflows")
    training_actions = train.add_subparsers(dest="action", required=True)
    smoke = training_actions.add_parser("smoke", help="Check training, gradients, checkpoint reload and resume")
    smoke.add_argument("dataset_dir", help="Canonical split dataset; only train pixels are loaded")
    smoke.add_argument("output_dir", help="New run directory; existing runs are never overwritten")
    smoke.add_argument("--steps", type=int, default=8, help="Total target global steps, including any resumed steps (1–100)")
    smoke.add_argument("--samples", type=int, default=8, help="Maximum selected training examples (1–64)")
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=20261020)
    smoke.add_argument("--lr", type=float, default=0.001)
    smoke.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    smoke.add_argument("--resume", help="Local checkpoint.pt with its .sha256 companion")
    smoke.add_argument("--json", action="store_true")
    baseline = training_actions.add_parser("baseline", help="Fine-tune pretrained U-Net/ResNet18 using train/validation only")
    baseline.add_argument("dataset_dir")
    baseline.add_argument("output_dir", help="New run directory; existing runs are never overwritten")
    baseline.add_argument("--config", required=True, help="Versioned baseline TOML configuration")
    baseline.add_argument("--weights-dir", required=True, help="Cache directory for pinned encoder weights")
    baseline.add_argument("--download-weights", action="store_true", help="Explicitly download missing pinned encoder files")
    baseline.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    baseline.add_argument("--resume", help="Local last.pt epoch checkpoint and .sha256 companion")
    baseline.add_argument("--json", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Frozen validation analysis, comparisons and candidate review")
    evaluations = evaluate.add_subparsers(dest="action", required=True)
    analysis = evaluations.add_parser("validation", help="Analyze validation errors without opening test pixels")
    proposals = evaluations.add_parser("candidates", help="Inspect unreviewed candidate predictions; no accuracy scores")
    for action in (analysis, proposals):
        action.add_argument("dataset_dir")
        action.add_argument("checkpoint", help="Local best.pt with its .sha256 companion")
        action.add_argument("output_dir", help="New offline gallery directory")
        action.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        action.add_argument("--json", action="store_true")
    analysis.add_argument("--top-k", type=int, default=12, help="Examples per error ranking (1–24)")
    comparison = evaluations.add_parser("compare-validations", help="Compare paired validation reports without opening pixels")
    comparison.add_argument("reference_dir", help="Reference validation analysis directory")
    comparison.add_argument("candidate_dir", help="Candidate validation analysis directory")
    comparison.add_argument("output_dir", help="New comparison report directory")
    comparison.add_argument("--json", action="store_true")
    negatives = evaluations.add_parser("negative-holdout", help="Score two frozen models on experimental negative pixels")
    negatives.add_argument("dataset_dir")
    negatives.add_argument("reference_checkpoint")
    negatives.add_argument("candidate_checkpoint")
    negatives.add_argument("output_dir")
    negatives.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    negatives.add_argument("--json", action="store_true")
    positives = evaluations.add_parser("positive-holdout", help="Measure recall on independently reviewed experimental PV pixels")
    positives.add_argument("dataset_dir")
    positives.add_argument("reference_checkpoint")
    positives.add_argument("candidate_checkpoint")
    positives.add_argument("output_dir")
    positives.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    positives.add_argument("--json", action="store_true")
    calibrate_p = evaluations.add_parser("calibrate", help="Calibrate model probabilities (Temperature & Platt scaling) on validation data")
    calibrate_p.add_argument("dataset_dir")
    calibrate_p.add_argument("checkpoint", help="Local best.pt checkpoint")
    calibrate_p.add_argument("output_dir", help="Output directory for calibration report")
    calibrate_p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    calibrate_p.add_argument("--subsample-pixels", type=int, default=0, help="Max pixels per crop to evaluate (0 for all)")
    calibrate_p.add_argument("--json", action="store_true")
    vectorize = commands.add_parser("vectorize", help="CAD-grade geospatial polygon vectorization and physical metrics")
    vectorize.add_argument("mask_or_image", help="Input binary mask PNG or RGB image")
    vectorize.add_argument("output_dir", help="Output directory for GeoJSON and CAD overlay")
    vectorize.add_argument("--profile", choices=("fjell", "vidda", "tinden"),
                           help="Model profile: 'fjell' (NRW, fewest false alarms), 'vidda' (Standard) or 'tinden' (High-End Flagship)")
    vectorize.add_argument("--checkpoint", nargs="+", help="One or more model checkpoints to ensemble")
    vectorize.add_argument("--ensemble-weights", nargs="+", type=float, help="Weights for ensemble models (default: equal)")
    vectorize.add_argument("--scales", nargs="+", type=float, default=[1.0], help="Scale factors for multi-scale pyramid TTA (default: 1.0)")
    vectorize.add_argument("--threshold", type=float, default=0.25, help="Probability threshold (default: 0.25)")
    vectorize.add_argument("--calibrate", action="store_true", help="Apply temperature scaling and calibrated Bayes-optimal threshold")
    vectorize.add_argument("--temperature", type=float, default=1.0, help="Temperature scalar T for logit calibration (default: 1.0)")
    vectorize.add_argument("--tile-size", type=int, default=400, help="Tile size for sliding-window inference (default: 400)")
    vectorize.add_argument("--overlap", type=float, default=0.50, help="Tile overlap fraction in [0.0, 0.75] (default: 0.50)")
    vectorize.add_argument("--tta", choices=("none", "flips", "rot4", "d4"), default="d4",
                           help="Test-Time Augmentation mode (default: d4, 8-fold dihedral symmetry)")
    vectorize.add_argument("--gsd", type=float, default=0.10, help="Ground sampling distance in meters (default: 0.10)")
    vectorize.add_argument("--transform", nargs=6, type=float, help="GDAL 6-parameter affine transform [a, b, c, d, e, f]")
    vectorize.add_argument("--crs", default="EPSG:25832", help="Coordinate Reference System (default: EPSG:25832)")
    vectorize.add_argument("--min-area-m2", type=float, default=1.5, help="Minimum PV array surface area in m² (default: 1.5, filter out sub-module noise)")
    vectorize.add_argument("--dsm", help="Path to Digital Surface Model (DSM / DOM) GeoTIFF or .npy elevation raster")
    vectorize.add_argument("--roof-tilt", type=float, default=0.0, help="Manual roof pitch / tilt angle in degrees (default: 0.0, horizontal)")
    vectorize.add_argument("--roof-azimuth", type=float, help="Manual roof compass azimuth in degrees (0=North, 90=East, 180=South, 270=West)")
    vectorize.add_argument("--ghi", type=float, default=1050.0, help="Annual Global Horizontal Irradiation in kWh/m² (default: 1050.0)")
    vectorize.add_argument("--nir", help="Path to Near-Infrared (NIR) single-channel GeoTIFF or .npy raster")
    vectorize.add_argument("--veto-vegetation", action="store_true", help="Automatically veto/suppress detected arrays on vegetated roofs (NDVI > 0.35)")
    vectorize.add_argument("--no-regularize", action="store_true", help="Disable 90-degree rectilinear Manhattan regularization")
    vectorize.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    vectorize.add_argument("--json", action="store_true")
    spectral = commands.add_parser("spectral", help="Multispectral analysis, NDVI extraction and Color Infrared (CIR) false-color composite export")
    spectral.add_argument("input_image", help="Input 4-band GeoTIFF/PNG or standard 3-band RGB image")
    spectral.add_argument("output_dir", help="Output directory for CIR composite, NDVI raster, and spectral report")
    spectral.add_argument("--nir", help="Optional separate Near-Infrared (NIR) band image or .npy file")
    spectral.add_argument("--cir", action="store_true", default=True, help="Export standard German Color Infrared (CIR) composite")
    spectral.add_argument("--ndvi-threshold", type=float, default=0.35, help="Vegetation NDVI threshold (default: 0.35)")
    spectral.add_argument("--json", action="store_true")
    temporal = commands.add_parser("temporal", help="Multi-temporal change detection & 4D asset tracking across survey epochs")
    temporal.add_argument("input_t1", help="GeoJSON or RGB image from survey epoch T1 (Historical)")
    temporal.add_argument("input_t2", help="GeoJSON or RGB image from survey epoch T2 (Current)")
    temporal.add_argument("output_dir", help="Output directory for multi-temporal GeoJSON, change audit, and CAD overlays")
    temporal.add_argument("--date-t1", help="Date string for epoch T1 (e.g. 2024-06-15)")
    temporal.add_argument("--date-t2", help="Date string for epoch T2 (e.g. 2025-06-20)")
    temporal.add_argument("--checkpoint", nargs="+", help="Checkpoints to run segmentation if inputs are RGB images")
    temporal.add_argument("--gsd", type=float, default=0.10, help="Ground sampling distance in meters (default: 0.10)")
    temporal.add_argument("--threshold", type=float, default=0.25, help="Probability threshold (default: 0.25)")
    temporal.add_argument("--iou-stable", type=float, default=0.70, help="Spatial overlap threshold to classify as stable array (default: 0.70)")
    temporal.add_argument("--iou-match", type=float, default=0.15, help="Spatial overlap threshold to classify as expansion (default: 0.15)")
    temporal.add_argument("--no-side-by-side", action="store_true", help="Render overlay only on T2 rather than side-by-side comparison")
    temporal.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    temporal.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "monitor":
        from pvextent.monitoring import serve
        try:
            serve(args.repo_dir, port=args.port)
        except KeyboardInterrupt:
            return 0
        except (OSError, ValueError) as error:
            print(f"MONITOR FAILED: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command == "vectorize":
        from pvextent.cad.vectorize import vectorize_mask, render_cad_overlay
        import numpy as np
        from PIL import Image
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        input_path = Path(args.mask_or_image).resolve()

        if getattr(args, "profile", None) == "fjell":
            if not args.checkpoint:
                cand = Path("runs/deeplabv3plus-resnet50-v7-nrw-hardneg-001/best.pt")
                if cand.exists():
                    args.checkpoint = [str(cand)]
            # The settings NRW test set v1 and the sealed benchmark measured
            args.tta = "none"
            args.scales = [1.0]
            args.overlap = 0.50
            args.threshold = 0.5
        elif getattr(args, "profile", None) == "vidda":
            if not args.checkpoint:
                cand = [
                    Path("runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt"),
                    Path("runs/baseline-resnet18-balanced-v4-001/best.pt"),
                ]
                for c in cand:
                    if c.exists():
                        args.checkpoint = [str(c)]
                        break
            args.tta = "none"
            args.scales = [1.0]
            args.overlap = 0.25
        elif getattr(args, "profile", None) == "tinden":
            if not args.checkpoint:
                cand1 = Path("runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt")
                cand2 = Path("runs/deeplabv3plus-resnet50-v6-nrw-pseudo-001/best.pt")
                if cand1.exists() and cand2.exists():
                    args.checkpoint = [str(cand1), str(cand2)]
                    args.ensemble_weights = [0.5, 0.5]
                elif cand1.exists():
                    args.checkpoint = [str(cand1)]
            args.tta = "d4"
            args.scales = [0.9, 1.0, 1.1]
            args.overlap = 0.50
            args.calibrate = True
            args.temperature = 0.4068
            args.veto_vegetation = True

        final_probs = None
        final_uncert = None

        if args.checkpoint:
            import torch
            from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
            from pvextent.models.resnet_unet import build_resnet_unet, normalize_rgb
            from pvextent.training.smoke import _device
            from pvextent.inference.tiler import predict_tiled

            img = Image.open(input_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            dev = _device(torch, args.device)

            ckpts = args.checkpoint
            n_models = len(ckpts)
            if args.ensemble_weights:
                weights = [w / sum(args.ensemble_weights) for w in args.ensemble_weights]
            else:
                weights = [1.0 / n_models] * n_models

            probs_list = []
            uncert_list = []
            threshold = args.threshold

            for ckpt_path_str in ckpts:
                ckpt_p = Path(ckpt_path_str).resolve()
                doc = torch.load(ckpt_p, map_location="cpu", weights_only=True)
                sd = doc["model_state_dict"]
                is_deeplab = any("aspp" in k for k in sd)
                is_segformer = any("patch_embed" in k or "decode_head" in k for k in sd)
                if is_deeplab:
                    model = build_deeplabv3plus_resnet50(None)
                elif is_segformer:
                    from pvextent.models.segformer import build_segformer_mit_b2
                    model = build_segformer_mit_b2(None)
                else:
                    model = build_resnet_unet(None)
                model.load_state_dict(sd)
                model.to(dev)
                model.eval()

                # Calibration handling
                temp = args.temperature
                if args.calibrate:
                    calib_cand = [
                        ckpt_p.parent / "calibration_report.json",
                        Path("runs/calibration_report.json"),
                    ]
                    calib_data = None
                    for c_path in calib_cand:
                        if c_path.exists():
                            try:
                                with open(c_path, "r", encoding="utf-8") as cf:
                                    calib_data = json.load(cf)
                                break
                            except Exception:
                                pass
                    if calib_data:
                        temp = calib_data.get("temperature_scaling", {}).get("temperature", 1.0)
                        if args.threshold == 0.25:
                            threshold = calib_data.get("temperature_scaling", {}).get("optimal_threshold_iou", 0.25)

                tiled_res = predict_tiled(
                    model=model,
                    image=arr,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    tta_mode=args.tta,
                    temperature=temp,
                    scales=args.scales,
                    device=dev,
                    normalize_fn=normalize_rgb,
                )
                probs_list.append(tiled_res["probabilities"])
                uncert_list.append(tiled_res["uncertainty"])

            # Weighted ensemble combination
            final_probs = np.zeros_like(probs_list[0], dtype=np.float64)
            for w, p in zip(weights, probs_list):
                final_probs += w * p
            final_probs = np.clip(final_probs, 0.0, 1.0).astype(np.float32)

            # Combined Bayesian predictive uncertainty: epistemic (cross-model disagreement) + aleatoric (TTA variance)
            if len(probs_list) > 1:
                stacked_probs = np.stack(probs_list, axis=0)
                epistemic_var = np.var(stacked_probs, axis=0)
                mean_aleatoric = np.mean(np.stack(uncert_list, axis=0), axis=0)
                final_uncert = (epistemic_var + mean_aleatoric).astype(np.float32)
            else:
                final_uncert = uncert_list[0]

            binary_mask = (final_probs >= threshold).astype(np.uint8)
        else:
            img = Image.open(input_path).convert("RGB")
            raw_mask = Image.open(input_path)
            binary_mask = (np.array(raw_mask) > 0).astype(np.uint8)

        # Georeference resolution: CLI explicit override -> auto-extract GeoTIFF/Worldfile
        from pvextent.cad.georef import extract_georeference
        if args.transform:
            georef = {"crs": args.crs, "transform": args.transform}
        else:
            georef = extract_georeference(input_path, default_crs=args.crs)

        geojson = vectorize_mask(
            binary_mask,
            georeference=georef,
            gsd_m=args.gsd,
            min_area_m2=args.min_area_m2,
            regularize=not args.no_regularize,
            probabilities=final_probs,
            uncertainty=final_uncert,
            dsm=args.dsm,
            default_tilt_deg=args.roof_tilt,
            default_azimuth_deg=args.roof_azimuth,
            ghi_kwh_m2=args.ghi,
            rgb=img,
            nir=args.nir,
            veto_vegetated_roofs=args.veto_vegetation,
        )
        geojson_path = output / "vectorized.geojson"
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        overlay_path = output / "cad_overlay.png"
        render_cad_overlay(img, geojson, overlay_path)

        # Export raw masks and probability heatmaps
        Image.fromarray((binary_mask * 255).astype(np.uint8)).save(output / "binary_mask.png")
        if final_probs is not None:
            Image.fromarray((np.clip(final_probs, 0.0, 1.0) * 255).astype(np.uint8)).save(output / "probabilities.png")
        if final_uncert is not None:
            u_max = max(1e-4, float(np.max(final_uncert)))
            uncert_scaled = (np.clip(final_uncert / u_max, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(uncert_scaled).save(output / "uncertainty.png")

        if args.json:
            print(json.dumps(geojson["summary"], indent=2))
        else:
            summary = geojson["summary"]
            spec_info = ""
            if summary.get("multispectral_nir_enabled"):
                spec_info = f", NIR Verified: {summary.get('spectral_silicon_confirmed', 0)} silicon, {summary.get('vegetated_roof_vetoes', 0)} vegetated vetoes"
            print(f"VECTORIZE OK: {summary['detected_pv_arrays']} arrays "
                  f"({summary.get('high_confidence_arrays', 0)} high-confidence, "
                  f"{summary.get('review_recommended_arrays', 0)} review-flagged), "
                  f"2D Area: {summary['total_pv_area_m2']} m², "
                  f"True 3D Area: {summary.get('total_true_3d_area_m2', summary['total_pv_area_m2'])} m² "
                  f"(tilt {summary.get('mean_roof_tilt_deg', 0.0)}°), "
                  f"{summary['total_capacity_kwp']} kWp, "
                  f"Annual Yield: {summary.get('total_annual_yield_mwh', 0.0)} MWh/a "
                  f"({summary.get('total_annual_co2_savings_tons', 0.0)} t CO2/a){spec_info}. Output: {output}")
        return 0
    if args.command == "spectral":
        from pvextent.cad.spectral import calculate_ndvi, compose_cir_composite
        import numpy as np
        from PIL import Image

        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        inp_p = Path(args.input_image).resolve()

        raw_img = Image.open(inp_p)
        if raw_img.mode in ("RGBA", "CMYK") or len(raw_img.getbands()) >= 4:
            arr = np.array(raw_img)
            rgb_arr = arr[..., :3]
            nir_arr = arr[..., 3]
        else:
            rgb_arr = np.array(raw_img.convert("RGB"))
            if args.nir:
                nir_p = Path(args.nir).resolve()
                if nir_p.suffix == ".npy":
                    nir_arr = np.load(nir_p)
                else:
                    nir_arr = np.array(Image.open(nir_p).convert("L"))
            else:
                nir_arr = (rgb_arr[..., 0].astype(np.float32) * 0.7 + rgb_arr[..., 1].astype(np.float32) * 0.3).astype(np.uint8)

        r_norm = rgb_arr[..., 0].astype(np.float32) / 255.0
        nir_norm = nir_arr.astype(np.float32) / 255.0
        ndvi = calculate_ndvi(nir_norm, r_norm)

        veg_mask = (ndvi > args.ndvi_threshold).astype(np.uint8) * 255
        Image.fromarray(veg_mask).save(output / "vegetation_mask.png")

        ndvi_vis = (np.clip((ndvi + 0.2) / 1.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(ndvi_vis).save(output / "ndvi.png")

        cir_img = compose_cir_composite(rgb_arr[..., 0], rgb_arr[..., 1], nir_arr)
        cir_img.save(output / "cir_composite.png")

        report = {
            "input_image": str(inp_p),
            "output_directory": str(output),
            "mean_ndvi": round(float(np.mean(ndvi)), 4),
            "max_ndvi": round(float(np.max(ndvi)), 4),
            "vegetated_pixel_percentage": round(float(np.mean(ndvi > args.ndvi_threshold) * 100), 2),
            "spectral_bands": 4,
            "cir_composite": str(output / "cir_composite.png"),
            "ndvi_raster": str(output / "ndvi.png"),
        }
        with open(output / "spectral_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"SPECTRAL OK: Mean NDVI: {report['mean_ndvi']}, "
                  f"Vegetated coverage: {report['vegetated_pixel_percentage']}%. "
                  f"Outputs written to: {output}")
        return 0
    if args.command == "temporal":
        from pvextent.cad.temporal import detect_temporal_changes, render_temporal_cad_overlay
        from pvextent.cad.vectorize import vectorize_mask
        from pvextent.cad.georef import extract_georeference
        import numpy as np
        from PIL import Image

        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        def _resolve_input(path_str: str):
            p = Path(path_str).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Input path not found: {p}")
            if p.suffix.lower() in (".json", ".geojson"):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                img_cand = [
                    p.with_suffix(".png"),
                    p.with_suffix(".jpg"),
                    p.parent / "cad_overlay.png",
                    p.parent / "input.png",
                ]
                img = None
                for c in img_cand:
                    if c.exists():
                        try:
                            img = Image.open(c).convert("RGB")
                            break
                        except Exception:
                            pass
                return data, img, None, None

            img = Image.open(p).convert("RGB")
            georef = extract_georeference(p)
            probs = None
            uncert = None
            if args.checkpoint:
                import torch
                from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
                from pvextent.models.resnet_unet import build_resnet_unet, normalize_rgb
                from pvextent.training.smoke import _device
                from pvextent.inference.tiler import predict_tiled

                arr = np.array(img, dtype=np.float32) / 255.0
                dev = _device(torch, args.device)
                ckpts = args.checkpoint
                probs_list = []
                uncert_list = []
                for ckpt_path_str in ckpts:
                    ckpt_p = Path(ckpt_path_str).resolve()
                    doc = torch.load(ckpt_p, map_location="cpu", weights_only=True)
                    sd = doc["model_state_dict"]
                    if any("aspp" in k for k in sd):
                        model = build_deeplabv3plus_resnet50(None)
                    elif any("patch_embed" in k or "decode_head" in k for k in sd):
                        from pvextent.models.segformer import build_segformer_mit_b2
                        model = build_segformer_mit_b2(None)
                    else:
                        model = build_resnet_unet(None)
                    model.load_state_dict(sd)
                    model.to(dev)
                    model.eval()
                    tiled_res = predict_tiled(
                        model=model,
                        image=arr,
                        device=dev,
                        normalize_fn=normalize_rgb,
                    )
                    probs_list.append(tiled_res["probabilities"])
                    uncert_list.append(tiled_res["uncertainty"])
                probs = np.mean(probs_list, axis=0).astype(np.float32)
                uncert = np.mean(uncert_list, axis=0).astype(np.float32)
                binary_mask = (probs >= args.threshold).astype(np.uint8)
            else:
                gray = Image.open(p).convert("L")
                binary_mask = (np.array(gray) > 127).astype(np.uint8)

            geojson = vectorize_mask(
                binary_mask,
                georeference=georef,
                gsd_m=args.gsd,
                probabilities=probs,
                uncertainty=uncert,
            )
            return geojson, img, probs, uncert

        gj_t1, img_t1, p_t1, u_t1 = _resolve_input(args.input_t1)
        gj_t2, img_t2, p_t2, u_t2 = _resolve_input(args.input_t2)

        temporal_geojson = detect_temporal_changes(
            geojson_t1=gj_t1,
            geojson_t2=gj_t2,
            date_t1=args.date_t1,
            date_t2=args.date_t2,
            probabilities_t1=p_t1,
            probabilities_t2=p_t2,
            uncertainty_t1=u_t1,
            uncertainty_t2=u_t2,
            iou_stable_threshold=args.iou_stable,
            iou_match_threshold=args.iou_match,
        )

        geojson_out = output / "temporal_changes.geojson"
        with open(geojson_out, "w", encoding="utf-8") as f:
            json.dump(temporal_geojson, f, indent=2)

        if img_t2 is not None:
            overlay_out = output / "temporal_cad_overlay.png"
            render_temporal_cad_overlay(
                image_t2=img_t2,
                temporal_geojson=temporal_geojson,
                output_path=overlay_out,
                image_t1=img_t1 if (not args.no_side_by_side and img_t1 is not None) else None,
            )

        if args.json:
            print(json.dumps(temporal_geojson["summary"], indent=2))
        else:
            s = temporal_geojson["summary"]
            print(
                f"TEMPORAL OK: {s['total_events']} events detected "
                f"({s['new_installations']} new, {s['expansions']} expansions, "
                f"{s['decommissioned_arrays']} decommissioned, {s['stable_arrays']} stable) | "
                f"Net ΔCapacity: {s['net_delta_capacity_kwp']:+.3f} kWp, "
                f"Net ΔModules: {s['net_delta_module_count']:+d}, "
                f"Net ΔYield: {s['net_delta_annual_yield_mwh']:+.3f} MWh/a. "
                f"Output: {output}"
            )
        return 0
    if args.command == "evaluate":
        try:
            if args.action == "validation":
                from pvextent.evaluation.validation import analyze_validation
                result = analyze_validation(args.dataset_dir, args.checkpoint, args.output_dir,
                                            device=args.device, top_k=args.top_k,
                                            progress=lambda message: print(message, file=sys.stderr))
            elif args.action == "compare-validations":
                from pvextent.evaluation.comparison import compare_validation_analyses
                result = compare_validation_analyses(args.reference_dir, args.candidate_dir, args.output_dir)
            elif args.action == "negative-holdout":
                from pvextent.evaluation.negative import evaluate_negative_holdout
                result = evaluate_negative_holdout(args.dataset_dir, args.reference_checkpoint,
                                                   args.candidate_checkpoint, args.output_dir,
                                                   device=args.device,
                                                   progress=lambda message: print(message, file=sys.stderr))
            elif args.action == "positive-holdout":
                from pvextent.evaluation.positive import evaluate_positive_holdout
                result = evaluate_positive_holdout(args.dataset_dir, args.reference_checkpoint,
                                                   args.candidate_checkpoint, args.output_dir,
                                                   device=args.device,
                                                   progress=lambda message: print(message, file=sys.stderr))
            elif args.action == "calibrate":
                from pvextent.data.segmentation import SegmentationDataset
                from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
                from pvextent.models.resnet_unet import build_resnet_unet, normalize_rgb
                from pvextent.training.smoke import _device
                from pvextent.inference.calibration import calibrate_validation_split
                import torch

                dev = _device(torch, args.device)
                out_dir = Path(args.output_dir).resolve()
                out_dir.mkdir(parents=True, exist_ok=True)

                val_dataset = SegmentationDataset(args.dataset_dir, split="val")
                doc = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                sd = doc["model_state_dict"]
                is_deeplab = any("aspp" in k for k in sd)
                is_segformer = any("patch_embed" in k or "decode_head" in k for k in sd)
                if is_deeplab:
                    model = build_deeplabv3plus_resnet50(None)
                elif is_segformer:
                    from pvextent.models.segformer import build_segformer_mit_b2
                    model = build_segformer_mit_b2(None)
                else:
                    model = build_resnet_unet(None)
                model.load_state_dict(sd)
                model.to(dev)

                result = calibrate_validation_split(
                    model=model,
                    dataset=val_dataset,
                    device=dev,
                    normalize_fn=normalize_rgb,
                    max_samples=0,
                    subsample_pixels=args.subsample_pixels,
                )
                report_path = out_dir / "calibration_report.json"
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            else:
                from pvextent.inference.candidates import export_candidate_predictions
                result = export_candidate_predictions(args.dataset_dir, args.checkpoint, args.output_dir,
                                                      device=args.device,
                                                      progress=lambda message: print(message, file=sys.stderr))
        except ValueError as error:
            if args.json:
                print(json.dumps({"ok": False, "error": str(error)}))
            else:
                print(f"{args.action.upper()} FAILED: {error}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.action == "calibrate":
            raw = result["raw_metrics"]
            t_scal = result["temperature_scaling"]
            print(f"CALIBRATE OK: Evaluated {result['samples_evaluated']} validation samples ({result['total_pixels_evaluated']} px).")
            print(f"  Raw: ECE={raw['ece']:.4f}, NLL={raw['nll']:.4f}, Bayes-opt IoU threshold={raw['optimal_threshold_iou']:.3f} (Max IoU={raw['max_iou']:.4f})")
            print(f"  Temperature Scaled (T={t_scal['temperature']:.3f}): ECE={t_scal['ece']:.4f} (-{t_scal['ece_reduction_pct']}%), NLL={t_scal['nll']:.4f}, Bayes-opt IoU threshold={t_scal['optimal_threshold_iou']:.3f} (Max IoU={t_scal['max_iou']:.4f})")
            print(f"  Report written to: {args.output_dir}/calibration_report.json")
        else:
            print(f"{args.action.upper()} OK. Output: {args.output_dir}")
        return 0
    if args.command == "train" and args.action == "baseline":
        from pvextent.training.baseline import BaselineTrainingError, run_baseline
        try:
            result = run_baseline(
                args.dataset_dir, args.output_dir, config_path=args.config, weights_dir=args.weights_dir,
                device=args.device, download_weights=args.download_weights, resume=args.resume,
                progress=lambda message: print(message, file=sys.stderr),
            )
        except BaselineTrainingError as error:
            if args.json:
                print(json.dumps({"ok": False, "error": str(error)}))
            else:
                print(f"BASELINE TRAINING FAILED: {error}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print(f"BASELINE OK: {result['epochs_completed']} epochs on {result['device']}; "
                  f"best validation pixel IoU={result['best_val']['iou']:.4f}. Output: {args.output_dir}")
        return 0
    if args.command == "train":
        from pvextent.training.smoke import TrainingSmokeError, run_smoke
        try:
            result = run_smoke(
                args.dataset_dir, args.output_dir, steps=args.steps, max_samples=args.samples,
                batch_size=args.batch_size, seed=args.seed, learning_rate=args.lr,
                device=args.device, resume=args.resume,
                progress=lambda message: print(message, file=sys.stderr),
            )
        except TrainingSmokeError as error:
            if args.json:
                print(json.dumps({"ok": False, "error": str(error)}))
            else:
                print(f"TRAINING SMOKE FAILED: {error}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print(f"TRAINING SMOKE OK: {result['steps_completed']} updates on {result['device']}; "
                  f"checkpoint reload verified. Output: {args.output_dir}")
        return 0
    if args.action != "validate":
        from pvextent.data.pv4ger import DatasetImportError

        try:
            if args.action == "import-pv4ger-seg":
                from pvextent.data.pv4ger import import_pv4ger_seg
                result = import_pv4ger_seg(
                    args.source_dir, args.output_dir, download=args.download, limit=args.limit,
                    progress=lambda message: print(message, file=sys.stderr),
                )
            elif args.action == "split":
                from pvextent.data.splitting import split_dataset
                result = split_dataset(args.source_dir, args.output_dir, seed=args.seed,
                                       block_size_m=args.block_size_m, buffer_m=args.buffer_m,
                                       ratios=tuple(args.ratios))
            elif args.action == "review":
                from pvextent.data.review import export_review_bundle
                result = export_review_bundle(args.source_dir, args.output_dir)
            elif args.action == "curate-negatives":
                from pvextent.data.curation import curate_negative_collection
                result = curate_negative_collection(args.source_dirs, args.review_paths, args.output_dir,
                                                    dataset_version=args.dataset_version, allow_ai_review=args.allow_ai_review)
            elif args.action == "augment-training":
                from pvextent.data.augmentation import augment_training_snapshot
                result = augment_training_snapshot(args.base_dir, args.negative_dir, args.output_dir,
                                                   dataset_version=args.dataset_version, buffer_m=args.buffer_m)
            elif args.action == "fetch-negative-holdout":
                from pvextent.data.negative_holdout import fetch_negative_holdout_candidates
                result = fetch_negative_holdout_candidates(args.exclude_dirs, args.output_dir, protocol_path=args.protocol)
            elif args.action == "fetch-training-candidates":
                from pvextent.data.nrw_training import fetch_training_candidates
                result = fetch_training_candidates(args.exclude_dirs, args.output_dir, protocol_path=args.protocol)
            elif args.action == "fetch-training-bulk":
                from pvextent.data.bulk_training import fetch_bulk_training
                result = fetch_bulk_training(
                    args.training_dir, args.protected_dirs, args.identity_only_dirs, args.output_dir,
                    protocol_path=args.protocol, cache_dir=args.cache_dir, allow_partial=args.allow_partial,
                    progress=lambda event: print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True),
                )
            elif args.action == "review-queue":
                from pvextent.data.review_queue import export_training_review_queue
                result = export_training_review_queue(args.dataset_dir, args.output_dir,
                                                      batch_size=args.batch_size, seed=args.seed)
            elif args.action == "curate-training-bulk":
                from pvextent.data.bulk_curation import curate_bulk_training
                result = curate_bulk_training(args.dataset_dir, args.queue_dir, args.review_dir,
                                              args.output_dir, dataset_version=args.dataset_version,
                                              protocol_path=args.protocol, allow_ai_review=args.allow_ai_review)
            elif args.action == "prepare-mask-proposals":
                from pvextent.data.bulk_curation import export_bulk_mask_proposals
                result = export_bulk_mask_proposals(args.dataset_dir, args.queue_dir, args.draft_dir,
                                                    args.output_dir, protocol_path=args.protocol)
            elif args.action == "augment-training-bulk":
                from pvextent.data.bulk_curation import augment_bulk_training
                result = augment_bulk_training(args.base_dir, args.curated_dir, args.protected_dirs,
                                               args.output_dir, dataset_version=args.dataset_version,
                                               protocol_path=args.protocol)
            elif args.action == "quarantine-training-quality":
                from pvextent.data.quality_quarantine import quarantine_training_quality
                result = quarantine_training_quality(args.source_dir, args.base_dir, args.output_dir,
                                                      protocol_path=args.protocol, evidence_path=args.evidence,
                                                      dataset_version=args.dataset_version)
            elif args.action == "freeze-negative-holdout":
                from pvextent.data.negative_freeze import freeze_negative_holdout
                result = freeze_negative_holdout(args.curated_dir, args.candidate_dir, args.exclude_dirs, args.output_dir,
                                                 protocol_path=args.protocol, acquisition_protocol_path=args.acquisition_protocol)
            elif args.action == "freeze-positive-holdout":
                from pvextent.data.positive_holdout import freeze_positive_holdout
                result = freeze_positive_holdout(args.candidate_dir, args.negative_holdout_dir,
                                                 args.exclude_dirs, args.output_dir,
                                                 review_paths=args.review_paths, protocol_path=args.protocol)
            else:
                from pvextent.data.nrw import fetch_nrw_roof_candidates
                result = fetch_nrw_roof_candidates(args.source_dir, args.output_dir, limit=args.limit)
        except DatasetImportError as error:
            if args.json:
                print(json.dumps({"ok": False, "error": str(error)}))
            else:
                print(f"{args.action.upper()} FAILED: {error}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.action == "prepare-mask-proposals":
            print(f"MASK PROPOSALS: {result['ready_batches']}/{result['total_batches']} batches to {args.output_dir}.")
        else:
            count = result["sample_count"] if args.action == "review-queue" else result["records"]
            print(f"{args.action.upper()}: {count} records to {args.output_dir}.")
        return 0
    report = validate_dataset(args.directory)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
    else:
        print(f"{'VALID' if report.ok else 'INVALID'}: {report.stats['records']} records, "
              f"{len(report.errors)} errors, {report.warning_count} warnings")
        for label, issues in (("ERROR", report.errors), ("WARNING", report.warnings)):
            for issue in issues:
                print(f"{label} {issue.location!r} [{issue.code}]: {issue.message}")
        if report.stats["stopped_early"]:
            print("Validation stopped at a configured limit; remaining records were not checked.")
        if report.warning_count > len(report.warnings):
            print("Additional warnings omitted; see warning_count in the JSON report.")
    return 0 if report.ok else 1
