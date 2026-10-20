# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry points for standalone PVExtent workflows."""

import argparse
import json
import sys

from pvextent import __version__
from pvextent.data import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pvextent")
    parser.add_argument("--version", action="version", version=f"PVExtent {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
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
    args = parser.parse_args(argv)
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
        else:
            print(f"{args.action.upper()}: {result['records']} records to {args.output_dir}.")
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
