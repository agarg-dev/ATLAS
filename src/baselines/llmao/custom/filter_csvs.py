#!/usr/bin/env python3
"""
filter_csvs.py
==============
Step 3b of the LLMAO evaluation pipeline.

Reads per-fold master CSVs written by ``process_csvs.py`` and filters each
one so only rows belonging to the validation ``pt_numbers`` for that fold
are retained.  This removes training samples that were included in the master
CSV if the fold directory contained more data than just the validation set.

(In practice, ``process_csvs.py`` already emits only validation samples, so
this step is a safety filter / explicit checkpoint rather than a heavy
transformation.)

Output columns (same as input):

    pt_number, line_number, line, is_buggy

Usage
-----
    python custom/filter_csvs.py \\
        --merged-csv-dir      data/processed_csv \\
        --validation-logs-dir model_logs/swebench \\
        --output-dir          data/filtered_csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import sys

csv.field_size_limit(2 ** 31 - 1)

_PT_NUM = re.compile(r"(\d+)\.pt$", re.IGNORECASE)


def _valid_pt_numbers(fold_dir: pathlib.Path) -> set[int]:
    val_file = fold_dir / "validation_set_files.json"
    if not val_file.exists():
        return set()
    paths: list[str] = json.loads(val_file.read_text(encoding="utf-8"))
    nums: set[int] = set()
    for p in paths:
        m = _PT_NUM.search(p.replace("\\", "/"))
        if m:
            nums.add(int(m.group(1)))
    return nums


def filter_fold(
    merged_csv: pathlib.Path,
    fold_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> int:
    valid_pts = _valid_pt_numbers(fold_dir)
    if not valid_pts:
        print(f"  [skip] no valid pt numbers for {fold_dir.name}")
        return 0

    out_path = output_dir / merged_csv.name
    rows_kept = 0

    with (
        open(merged_csv, "r", encoding="utf-8", newline="") as in_fh,
        open(out_path, "w", encoding="utf-8", newline="") as out_fh,
    ):
        reader = csv.DictReader(in_fh)
        writer = csv.DictWriter(
            out_fh,
            fieldnames=["pt_number", "line_number", "line", "is_buggy"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in reader:
            try:
                pt = int(row["pt_number"])
            except (KeyError, ValueError):
                continue
            if pt in valid_pts:
                writer.writerow(row)
                rows_kept += 1

    print(f"  {merged_csv.name} -> {out_path.name}  ({rows_kept} rows kept)")
    return rows_kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--merged-csv-dir",
        default="data/processed_csv",
        help="Directory with per-fold master CSVs from process_csvs.py.",
    )
    ap.add_argument(
        "--validation-logs-dir",
        default="model_logs/swebench",
        help="Directory containing per-fold subdirectories (for validation_set_files.json).",
    )
    ap.add_argument(
        "--output-dir",
        default="data/filtered_csv",
        help="Directory to write filtered CSVs.",
    )
    args = ap.parse_args()

    merged_dir = pathlib.Path(args.merged_csv_dir)
    val_logs = pathlib.Path(args.validation_logs_dir)
    out_dir = pathlib.Path(args.output_dir)

    if not merged_dir.is_dir():
        sys.exit(f"ERROR: --merged-csv-dir not found: {merged_dir}")
    if not val_logs.is_dir():
        sys.exit(f"ERROR: --validation-logs-dir not found: {val_logs}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(merged_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files found in {merged_dir}")

    print(f"Found {len(csv_files)} merged CSV(s) in {merged_dir}")
    total_kept = 0
    for csv_path in csv_files:
        fold_dir = val_logs / csv_path.stem
        if not fold_dir.is_dir():
            print(f"  [warn] no matching fold dir for {csv_path.name}, copying as-is")
            shutil.copy(csv_path, out_dir / csv_path.name)
            continue
        total_kept += filter_fold(csv_path, fold_dir, out_dir)

    print(f"\nDone. Total rows kept: {total_kept}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
