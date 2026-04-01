#!/usr/bin/env python3
"""
combined_csvs.py
================
Step 3c of the LLMAO evaluation pipeline.

For each fold, reads the filtered CSV from ``filter_csvs.py`` and the best
``step_*.json`` snapshot from ``--json-logs-dir``, then adds a ``probability``
column by aligning the flat ``prob`` array in the JSON to rows in order.

The flat ``prob`` array in ``step_*.json`` is produced by the validation loop
in ``training.py`` with ``shuffle=False``, so its entries are in the same
order as the rows in the filtered CSV (one probability per non-padding line,
per sample, in ``validation_set_files.json`` order).

Output columns:

    pt_number, line_number, line, is_buggy, probability

Usage
-----
    python custom/combined_csvs.py \\
        --csv-dir        data/filtered_csv \\
        --json-logs-dir  model_logs/swebench \\
        --output-dir     data/combined_csvs
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

csv.field_size_limit(2 ** 31 - 1)


def _best_step_json(fold_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the single step_*.json in fold_dir, or None."""
    matches = list(fold_dir.glob("step_*.json"))
    if not matches:
        return None
    # Training keeps only the best checkpoint, so there should be exactly one.
    # If somehow there are multiple, pick the one with the highest step number.
    return max(matches, key=lambda p: int(p.stem.split("_")[1]))


def combine_fold(
    filtered_csv: pathlib.Path,
    fold_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> int:
    step_json = _best_step_json(fold_dir)
    if step_json is None:
        print(f"  [skip] no step_*.json in {fold_dir.name}")
        return 0

    data = json.loads(step_json.read_text(encoding="utf-8"))
    probs: list[float] = data["prob"]

    out_path = output_dir / filtered_csv.name
    rows_written = 0

    with (
        open(filtered_csv, "r", encoding="utf-8", newline="") as in_fh,
        open(out_path, "w", encoding="utf-8", newline="") as out_fh,
    ):
        reader = csv.DictReader(in_fh)
        fieldnames = ["pt_number", "line_number", "line", "is_buggy", "probability"]
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        writer.writeheader()

        prob_idx = 0
        for row in reader:
            prob = probs[prob_idx] if prob_idx < len(probs) else 0.0
            writer.writerow({
                "pt_number":   row["pt_number"],
                "line_number": row["line_number"],
                "line":        row["line"],
                "is_buggy":    row["is_buggy"],
                "probability": prob,
            })
            prob_idx += 1
            rows_written += 1

    if prob_idx < len(probs):
        print(
            f"  [warn] {fold_dir.name}: used {prob_idx}/{len(probs)} probabilities "
            f"({len(probs) - prob_idx} unused — padding tokens filtered by training)"
        )

    print(f"  {filtered_csv.name} → {out_path.name}  ({rows_written} rows)")
    return rows_written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--csv-dir",
        default="data/filtered_csv",
        help="Directory with filtered CSVs from filter_csvs.py.",
    )
    ap.add_argument(
        "--json-logs-dir",
        default="model_logs/swebench",
        help="Directory containing per-fold subdirectories (each with step_*.json).",
    )
    ap.add_argument(
        "--output-dir",
        default="data/combined_csvs",
        help="Directory to write combined CSVs with probability column.",
    )
    args = ap.parse_args()

    csv_dir = pathlib.Path(args.csv_dir)
    json_logs = pathlib.Path(args.json_logs_dir)
    out_dir = pathlib.Path(args.output_dir)

    if not csv_dir.is_dir():
        sys.exit(f"ERROR: --csv-dir not found: {csv_dir}")
    if not json_logs.is_dir():
        sys.exit(f"ERROR: --json-logs-dir not found: {json_logs}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} filtered CSV(s) in {csv_dir}")
    total_rows = 0
    for csv_path in csv_files:
        fold_dir = json_logs / csv_path.stem
        if not fold_dir.is_dir():
            print(f"  [warn] no matching fold dir for {csv_path.name}, skipping")
            continue
        total_rows += combine_fold(csv_path, fold_dir, out_dir)

    print(f"\nDone. Total rows written: {total_rows}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
