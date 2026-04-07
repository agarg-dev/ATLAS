#!/usr/bin/env python3
"""
process_csvs.py
===============
Step 3a of the LLMAO evaluation pipeline.

For each 10-fold subdirectory under ``--validation-logs-dir``, read
``validation_set_files.json`` to discover which ``.pt`` samples were held out,
then look up the corresponding mirror CSV in ``--csv-data-dir`` (same integer
index, e.g. ``42.pt`` → ``42.csv``).  Parse each mirror CSV and emit one
master CSV per fold with columns:

    pt_number, line_number, line, is_buggy

``top_k_per_pt.py`` reconstructs the full source for each pt by joining the
``line`` column values in line_number order, then parses Python functions via
the ``ast`` module to compute function-level Hit@K metrics.

Usage
-----
    python custom/process_csvs.py \\
        --validation-logs-dir model_logs/swebench \\
        --csv-data-dir        data/codegen_instances_csv/swebench_350M \\
        --output-dir          data/processed_csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import sys

csv.field_size_limit(2 ** 31 - 1)

_PT_NUM = re.compile(r"(\d+)\.pt$", re.IGNORECASE)


def pt_path_to_number(pt_path: str) -> int | None:
    """Extract the integer index from a ``.pt`` file path."""
    m = _PT_NUM.search(pt_path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def process_fold(
    fold_dir: pathlib.Path,
    csv_data_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> int:
    """Build a master CSV for one fold. Returns number of rows written."""
    val_file = fold_dir / "validation_set_files.json"
    if not val_file.exists():
        print(f"  [skip] no validation_set_files.json in {fold_dir.name}")
        return 0

    val_paths: list[str] = json.loads(val_file.read_text(encoding="utf-8"))
    if not val_paths:
        print(f"  [skip] empty validation_set_files.json in {fold_dir.name}")
        return 0

    out_path = output_dir / f"{fold_dir.name}.csv"
    rows_written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(["pt_number", "line_number", "line", "is_buggy"])

        for pt_path in val_paths:
            pt_num = pt_path_to_number(pt_path)
            if pt_num is None:
                print(f"  [warn] could not parse pt number from: {pt_path}")
                continue

            mirror_csv = csv_data_dir / f"{pt_num}.csv"
            if not mirror_csv.exists():
                print(f"  [warn] mirror CSV not found: {mirror_csv}")
                continue

            with open(mirror_csv, "r", encoding="utf-8", newline="") as in_fh:
                reader = csv.reader(in_fh)
                for row in reader:
                    if len(row) < 2:
                        continue
                    source = row[0]
                    try:
                        buggy_lines: list[int] = json.loads(row[1])
                    except Exception:
                        buggy_lines = []

                    buggy_set = set(buggy_lines)
                    for line_idx, line_text in enumerate(source.splitlines()):
                        writer.writerow([
                            pt_num,
                            line_idx,
                            line_text,
                            1 if line_idx in buggy_set else 0,
                        ])
                        rows_written += 1

    print(f"  {fold_dir.name} -> {out_path.name}  ({rows_written} rows)")
    return rows_written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--validation-logs-dir",
        default="model_logs/swebench",
        help="Directory containing per-fold subdirectories (each with validation_set_files.json).",
    )
    ap.add_argument(
        "--csv-data-dir",
        default="data/codegen_instances_csv/swebench_350M",
        help="Directory containing mirror CSVs produced by codegen_loading.py.",
    )
    ap.add_argument(
        "--output-dir",
        default="data/processed_csv",
        help="Directory to write per-fold master CSVs.",
    )
    args = ap.parse_args()

    val_logs = pathlib.Path(args.validation_logs_dir)
    csv_data = pathlib.Path(args.csv_data_dir)
    out_dir = pathlib.Path(args.output_dir)

    if not val_logs.is_dir():
        sys.exit(f"ERROR: --validation-logs-dir not found: {val_logs}")
    if not csv_data.is_dir():
        sys.exit(f"ERROR: --csv-data-dir not found: {csv_data}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(
        d for d in val_logs.iterdir()
        if d.is_dir() and (d / "validation_set_files.json").exists()
    )
    if not fold_dirs:
        sys.exit(f"No fold subdirectories with validation_set_files.json found in {val_logs}")

    print(f"Found {len(fold_dirs)} fold(s) in {val_logs}")
    total_rows = 0
    for fold_dir in fold_dirs:
        total_rows += process_fold(fold_dir, csv_data, out_dir)

    print(f"\nDone. Total rows written: {total_rows}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
