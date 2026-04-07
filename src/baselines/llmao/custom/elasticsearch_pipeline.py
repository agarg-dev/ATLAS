#!/usr/bin/env python3
"""
elasticsearch_pipeline.py
=========================
End-to-end LLMAO pipeline for the Elasticsearch triplet JSON dataset.

Subcommands
-----------
  preprocess   Read the triplet JSON, merge duplicate issue/file rows, chunk
               sources into 128-line bug-containing windows, and write
               LLMAO-format CSVs plus ``metadata.json``.

  run          Print (or execute) codegen_loading + training for one
               ``data_path`` / ``data_name`` (default: ``data`` / ``elasticsearch_pipeline``).

  evaluate     Run the generic CSV evaluation flow over the trained dataset and
               report line-level + Java function-level Hit@K / MRR.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import textwrap
from typing import Any

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from swebench_pipeline import chunk_file


def _load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def _sanitize_path(path: str) -> str:
    return path.replace("/", "__").replace("\\", "__")


def _group_triplets(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        issue_id = str(row.get("issue_id", "")).strip()
        positive_path = str(row.get("positive_path", "")).strip()
        if not issue_id or not positive_path:
            continue
        grouped.setdefault((issue_id, positive_path), []).append(row)
    return grouped


def stage_preprocess(args: argparse.Namespace) -> None:
    input_json = pathlib.Path(args.input_json)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_json)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
        print(f"Limiting to {len(rows)} JSON rows (--max-rows).")
    else:
        print(f"Loaded {len(rows)} JSON rows.")

    grouped = _group_triplets(rows)
    print(f"Grouped into {len(grouped)} issue/file pairs.")

    metadata: dict[str, Any] = {}
    csv_count = 0
    skipped_missing = 0
    skipped_inconsistent = 0
    skipped_no_valid_bugs = 0

    for (issue_id, positive_path), group_rows in grouped.items():
        sample = group_rows[0]
        required_fields = ["repo_name", "before_fix_sha", "positive", "positive_path"]
        if any(not str(sample.get(field, "")).strip() for field in required_fields):
            skipped_missing += 1
            continue

        repo_name = str(sample["repo_name"]).strip()
        base_commit = str(sample["before_fix_sha"]).strip()
        source = str(sample["positive"])
        rel_path = str(sample["positive_path"]).strip()

        consistent = True
        for row in group_rows[1:]:
            if (
                str(row.get("repo_name", "")).strip() != repo_name
                or str(row.get("before_fix_sha", "")).strip() != base_commit
                or str(row.get("positive_path", "")).strip() != rel_path
                or str(row.get("positive", "")) != source
            ):
                consistent = False
                break
        if not consistent:
            print(f"[warn] inconsistent rows for issue/file {issue_id}::{positive_path} - skipping")
            skipped_inconsistent += 1
            continue

        total_lines = len(source.splitlines())
        observed_splits: set[str] = set()
        raw_bug_lines: list[int] = []
        for row in group_rows:
            split = str(row.get("split", "")).strip()
            if split:
                observed_splits.add(split)
            try:
                raw_bug_lines.append(int(row["buggy_line_number"]) - 1)
            except (KeyError, TypeError, ValueError):
                continue

        valid_bug_lines = sorted({line for line in raw_bug_lines if 0 <= line < total_lines})
        if not valid_bug_lines:
            skipped_no_valid_bugs += 1
            continue

        chunks = chunk_file(source, valid_bug_lines)
        if not chunks:
            skipped_no_valid_bugs += 1
            continue

        issue_meta = metadata.setdefault(
            issue_id,
            {
                "repo": repo_name,
                "base_commit": base_commit,
                "splits": [],
                "files": {},
            },
        )
        if issue_meta["repo"] != repo_name or issue_meta["base_commit"] != base_commit:
            print(f"[warn] inconsistent issue-level metadata for {issue_id} - skipping {rel_path}")
            skipped_inconsistent += 1
            continue
        issue_meta["splits"] = sorted(set(issue_meta["splits"]) | observed_splits)

        safe_path = _sanitize_path(rel_path)
        file_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_start = int(chunk["chunk_start"])
            csv_name = f"{issue_id}__{safe_path}__{chunk_start}.csv"
            csv_path = out_dir / csv_name

            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_ALL, escapechar="\\")
                writer.writerow([chunk["source"], json.dumps(chunk["buggy_lines"])])

            file_chunks.append(
                {
                    "chunk_start": chunk_start,
                    "csv": str(csv_path),
                    "buggy_lines": chunk["buggy_lines"],
                }
            )
            csv_count += 1

        issue_meta["files"][rel_path] = {
            "full_source": source,
            "global_buggy_lines": valid_bug_lines,
            "csv_row_count": len(group_rows),
            "observed_splits": sorted(observed_splits),
            "total_lines": total_lines,
            "chunks": file_chunks,
        }

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    parent = out_dir.parent
    data_name = out_dir.name
    data_path = str(parent).replace("\\", "/")

    print("\nElasticsearch preprocessing complete.")
    print(f"  Issues in metadata          : {len(metadata)}")
    print(f"  CSV files written           : {csv_count}")
    print(f"  Skipped missing rows        : {skipped_missing}")
    print(f"  Skipped inconsistent groups : {skipped_inconsistent}")
    print(f"  Skipped without valid bugs  : {skipped_no_valid_bugs}")
    print(f"  Metadata saved to           : {meta_path}")
    print("\nNext step - extract CodeGen tensors and train:")
    print(
        f'  python custom/elasticsearch_pipeline.py run --data-path "{data_path}" '
        f'--data-name "{data_name}" --pretrain-type 350M --execute'
    )


def stage_run(args: argparse.Namespace) -> None:
    biggest = "1" if args.pretrain_type == "16B" else "0"
    data_path = args.data_path.replace("\\", "/")
    data_name = args.data_name
    root = pathlib.Path(__file__).parent.parent
    codegen_loading = root / "codegen_loading.py"
    training = root / "training.py"

    commands = [
        [sys.executable, str(codegen_loading), data_path, data_name, biggest],
        [sys.executable, str(training), data_path, data_name, args.pretrain_type, "1"],
    ]

    print("Commands to run (in order):\n", flush=True)
    for command in commands:
        print("  " + subprocess.list2cmdline(command), flush=True)

    if args.execute:
        for command in commands:
            print(f"\nExecuting: {subprocess.list2cmdline(command)}", flush=True)
            result = subprocess.run(command)
            if result.returncode != 0:
                sys.exit(
                    f"Command failed with exit code {result.returncode}: "
                    f"{subprocess.list2cmdline(command)}"
                )
    else:
        print("\nAdd --execute to run these commands directly, or copy-paste them.", flush=True)


def _run_command(command: list[str]) -> None:
    print(f"\nExecuting: {subprocess.list2cmdline(command)}", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}: {subprocess.list2cmdline(command)}")


def stage_evaluate(args: argparse.Namespace) -> None:
    root = pathlib.Path(__file__).parent.parent
    processed_dir = pathlib.Path(args.processed_dir)
    filtered_dir = pathlib.Path(args.filtered_dir)
    combined_dir = pathlib.Path(args.combined_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            sys.executable,
            str(root / "custom" / "process_csvs.py"),
            "--validation-logs-dir",
            args.validation_logs_dir,
            "--csv-data-dir",
            args.csv_data_dir,
            "--output-dir",
            str(processed_dir),
        ],
        [
            sys.executable,
            str(root / "custom" / "filter_csvs.py"),
            "--merged-csv-dir",
            str(processed_dir),
            "--validation-logs-dir",
            args.validation_logs_dir,
            "--output-dir",
            str(filtered_dir),
        ],
        [
            sys.executable,
            str(root / "custom" / "combined_csvs.py"),
            "--csv-dir",
            str(filtered_dir),
            "--json-logs-dir",
            args.validation_logs_dir,
            "--output-dir",
            str(combined_dir),
        ],
        [
            sys.executable,
            str(root / "top_k_per_pt.py"),
            "--csv-dir",
            str(combined_dir),
            "--language",
            "java",
            "--metadata",
            args.metadata,
            "--original-data-dir",
            args.original_data_dir,
            "--csv-data-dir",
            args.csv_data_dir,
            "--k-values",
            *[str(k) for k in args.k_values],
        ],
    ]

    print("Evaluation commands to run (in order):\n", flush=True)
    for command in commands:
        print("  " + subprocess.list2cmdline(command), flush=True)

    for command in commands:
        _run_command(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser(
        "preprocess",
        help="Read the Elasticsearch triplet JSON and write LLMAO-format CSVs.",
    )
    p_pre.add_argument(
        "--input-json",
        default="custom/data/triplet_dataset.json",
        help="Path to the triplet JSON. (default: custom/data/triplet_dataset.json)",
    )
    p_pre.add_argument(
        "--out-dir",
        default="data/elasticsearch_pipeline",
        help="Directory to write LLMAO CSVs and metadata.json. (default: data/elasticsearch_pipeline)",
    )
    p_pre.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of JSON rows to process before grouping (useful for testing).",
    )

    p_run = sub.add_parser(
        "run",
        help="Print (or execute) codegen_loading + training for one data_path/data_name.",
    )
    p_run.add_argument(
        "--data-path",
        default="data",
        help="Parent folder passed to codegen_loading / training (default: data).",
    )
    p_run.add_argument(
        "--data-name",
        default="elasticsearch_pipeline",
        help="Subfolder with CSVs under --data-path (default: elasticsearch_pipeline).",
    )
    p_run.add_argument(
        "--pretrain-type",
        default="350M",
        choices=["350M", "2B", "6B", "16B"],
        help="CodeGen model size. (default: 350M)",
    )
    p_run.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the commands instead of just printing them.",
    )

    p_eval = sub.add_parser(
        "evaluate",
        help="Run the CSV evaluation flow for the Elasticsearch dataset.",
    )
    p_eval.add_argument(
        "--validation-logs-dir",
        default="model_logs/elasticsearch_pipeline",
        help="Directory containing training fold subdirectories. (default: model_logs/elasticsearch_pipeline)",
    )
    p_eval.add_argument(
        "--metadata",
        default="data/elasticsearch_pipeline/metadata.json",
        help="Metadata file used to recover full-file function context.",
    )
    p_eval.add_argument(
        "--original-data-dir",
        default="data/elasticsearch_pipeline",
        help="Original preprocess CSV directory used to map PT numbers back to chunks.",
    )
    p_eval.add_argument(
        "--csv-data-dir",
        default=None,
        help="Directory containing mirror CSVs from codegen_loading.py. Defaults to data/codegen_instances_csv/elasticsearch_pipeline_<pretrain_type>.",
    )
    p_eval.add_argument(
        "--processed-dir",
        default="data/processed_csv/elasticsearch_pipeline",
        help="Directory for process_csvs.py output.",
    )
    p_eval.add_argument(
        "--filtered-dir",
        default="data/filtered_csv/elasticsearch_pipeline",
        help="Directory for filter_csvs.py output.",
    )
    p_eval.add_argument(
        "--combined-dir",
        default="data/combined_csvs/elasticsearch_pipeline",
        help="Directory for combined_csvs.py output.",
    )
    p_eval.add_argument(
        "--pretrain-type",
        default="350M",
        choices=["350M", "2B", "6B", "16B"],
        help="CodeGen model size used in training. (default: 350M)",
    )
    p_eval.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="K values to evaluate (default: 1 3 5 10).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        stage_preprocess(args)
    elif args.command == "run":
        stage_run(args)
    elif args.command == "evaluate":
        if args.csv_data_dir is None:
            args.csv_data_dir = f"data/codegen_instances_csv/elasticsearch_pipeline_{args.pretrain_type}"
        stage_evaluate(args)


if __name__ == "__main__":
    main()
