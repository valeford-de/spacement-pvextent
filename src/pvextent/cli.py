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
    args = parser.parse_args(argv)
    if args.action == "import-pv4ger-seg":
        from pvextent.data.pv4ger import DatasetImportError, import_pv4ger_seg

        try:
            result = import_pv4ger_seg(
                args.source_dir, args.output_dir, download=args.download, limit=args.limit,
                progress=lambda message: print(message, file=sys.stderr),
            )
        except DatasetImportError as error:
            if args.json:
                print(json.dumps({"ok": False, "error": str(error)}))
            else:
                print(f"IMPORT FAILED: {error}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print(f"IMPORTED: {result['records']} records to {args.output_dir}; "
                  f"{result['validation']['error_count']} errors, "
                  f"{result['validation']['warning_count']} warnings. Splits remain unassigned.")
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
