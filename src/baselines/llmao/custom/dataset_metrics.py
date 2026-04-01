#!/usr/bin/env python3
"""
dataset_metrics.py
==================
Download SWE-Bench Verified and compute characteristic metrics for buggy
and non-buggy Python files across all bug-report instances.

Metrics reported
----------------
  Bug Reports          : unique SWE-Bench instances processed
  Total Files          : distinct Python files (buggy / non-buggy)
  Buggy Functions      : functions that contain at least one buggy line
  Total Functions      : all functions found by AST parsing
  Median Functions/File: median number of functions per Python file
  Median Lines/File    : median total lines per Python file
  Median Lines/Function: median function length in lines

Usage
-----
  python custom/dataset_metrics.py --repo-cache ./repos

  # Limit to first 50 instances for a quick test:
  python custom/dataset_metrics.py --repo-cache ./repos --max-instances 50

  # Also write raw per-instance data to JSON:
  python custom/dataset_metrics.py --repo-cache ./repos --output metrics.json

Dependencies
------------
  pip install datasets gitpython
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import statistics
import subprocess
import sys
import tarfile
import textwrap
from typing import Any

# Ensure repo root is on sys.path so sibling imports work regardless of CWD.
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from swebench_pipeline import (
    _ensure_repo,
    _read_file_at_commit,
    get_python_function_ranges,
    map_line_to_function,
    parse_patch,
)


# ---------------------------------------------------------------------------
# Fast bulk file extraction via git archive
# ---------------------------------------------------------------------------

def _read_all_py_files_at_commit(repo, sha: str) -> dict[str, str]:
    """
    Extract ALL .py files from the repo at the given commit in a single
    `git archive` call (vastly faster than one `git show` per file).

    Returns a dict mapping relative path → file source text.
    Silently skips files that cannot be decoded as UTF-8.
    """
    repo_path = repo.working_dir
    try:
        proc = subprocess.run(
            ["git", "archive", "--format=tar", sha, "--", "*.py"],
            cwd=repo_path,
            capture_output=True,
            timeout=300,
        )
    except Exception as exc:
        print(f"  warning: git archive failed for {sha}: {exc}")
        return {}

    if proc.returncode != 0:
        # git archive doesn't support glob on all versions; fall back to full tree
        try:
            proc = subprocess.run(
                ["git", "archive", "--format=tar", sha],
                cwd=repo_path,
                capture_output=True,
                timeout=300,
            )
        except Exception as exc:
            print(f"  warning: git archive (full) failed for {sha}: {exc}")
            return {}
        if proc.returncode != 0:
            print(f"  warning: git archive returned {proc.returncode} for {sha}")
            return {}

    files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".py"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    files[member.name] = f.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
    except Exception as exc:
        print(f"  warning: tar extraction failed for {sha}: {exc}")

    return files


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def _file_metrics(source: str) -> dict[str, Any]:
    """
    Compute per-file metrics from raw source text.

    Returns a dict with keys:
      n_lines      : total physical lines
      functions    : list of (name, start_0based, end_0based)
      n_funcs      : number of functions/methods found
      func_lengths : list of (end - start + 1) for each function
    """
    funcs = get_python_function_ranges(source)
    func_lengths = [end - start + 1 for _, start, end in funcs]
    return {
        "n_lines": len(source.splitlines()),
        "functions": funcs,
        "n_funcs": len(funcs),
        "func_lengths": func_lengths,
    }


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_instance(
    instance: dict[str, Any],
    repo_cache: pathlib.Path,
) -> dict[str, Any] | None:
    """
    Process one SWE-Bench instance.

    Returns a dict with:
      instance_id   : the SWE-Bench instance ID
      buggy_files   : list of per-file metric dicts (files touched by the patch)
      nonbuggy_files: list of per-file metric dicts (all other .py files)
      buggy_funcs   : total functions containing at least one buggy line

    Returns None if the instance should be skipped (no patch, non-Python, clone error).
    """
    iid = instance["instance_id"]
    repo_slug = instance["repo"]
    base_commit = instance["base_commit"]
    patch = instance["patch"]

    if not patch.strip():
        return None

    file_buglines = parse_patch(patch)
    file_buglines = {fp: bl for fp, bl in file_buglines.items() if fp.endswith(".py")}
    if not file_buglines:
        return None

    repo_url = f"https://github.com/{repo_slug}.git"
    try:
        git_repo = _ensure_repo(repo_url, repo_cache, base_commit)
    except Exception as exc:
        print(f"  could not clone/fetch repo: {exc} — skipping.")
        return None

    # Extract all .py files in one shot via git archive.
    print(f"  extracting .py files via git archive...", end=" ", flush=True)
    all_py_sources = _read_all_py_files_at_commit(git_repo, base_commit)
    print(f"{len(all_py_sources)} files found.")

    buggy_paths = set(file_buglines.keys())

    # --- Buggy files ---
    buggy_file_metrics: list[dict[str, Any]] = []
    total_buggy_funcs = 0

    for rel_path, buggy_lines in file_buglines.items():
        source = all_py_sources.get(rel_path)
        if source is None:
            # git archive glob may have missed it; fall back to git show
            source = _read_file_at_commit(git_repo, base_commit, rel_path)
        if source is None:
            continue

        fm = _file_metrics(source)

        buggy_func_names: set[str] = set()
        for bl in buggy_lines:
            fn = map_line_to_function(bl, fm["functions"])
            if fn:
                buggy_func_names.add(fn)

        fm["buggy_func_count"] = len(buggy_func_names)
        total_buggy_funcs += len(buggy_func_names)
        buggy_file_metrics.append(fm)

    # --- Non-buggy files: every .py not touched by the patch ---
    nonbuggy_file_metrics: list[dict[str, Any]] = []
    for rel_path, source in all_py_sources.items():
        if rel_path in buggy_paths:
            continue
        nonbuggy_file_metrics.append(_file_metrics(source))

    print(f"  buggy={len(buggy_file_metrics)} files, "
          f"non-buggy={len(nonbuggy_file_metrics)} files, "
          f"buggy-funcs={total_buggy_funcs}")

    return {
        "instance_id": iid,
        "buggy_files": buggy_file_metrics,
        "nonbuggy_files": nonbuggy_file_metrics,
        "buggy_funcs": total_buggy_funcs,
    }


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _safe_median(values: list[float | int]) -> float:
    return statistics.median(values) if values else 0.0


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute the final summary table values from per-instance result dicts.
    """
    bug_reports = len(results)

    # Flatten across all instances
    buggy_file_list: list[dict[str, Any]] = []
    nonbuggy_file_list: list[dict[str, Any]] = []
    total_buggy_funcs = 0

    for r in results:
        buggy_file_list.extend(r["buggy_files"])
        nonbuggy_file_list.extend(r["nonbuggy_files"])
        total_buggy_funcs += r["buggy_funcs"]

    # Buggy-file stats
    buggy_n_files = len(buggy_file_list)
    buggy_n_funcs = sum(f["n_funcs"] for f in buggy_file_list)
    buggy_funcs_per_file = [f["n_funcs"] for f in buggy_file_list]
    buggy_lines_per_file = [f["n_lines"] for f in buggy_file_list]
    buggy_func_lengths: list[int] = []
    for f in buggy_file_list:
        buggy_func_lengths.extend(f["func_lengths"])

    # Non-buggy-file stats
    nonbuggy_n_files = len(nonbuggy_file_list)
    nonbuggy_n_funcs = sum(f["n_funcs"] for f in nonbuggy_file_list)
    nonbuggy_funcs_per_file = [f["n_funcs"] for f in nonbuggy_file_list]
    nonbuggy_lines_per_file = [f["n_lines"] for f in nonbuggy_file_list]
    nonbuggy_func_lengths: list[int] = []
    for f in nonbuggy_file_list:
        nonbuggy_func_lengths.extend(f["func_lengths"])

    return {
        "bug_reports": bug_reports,
        "buggy": {
            "total_files": buggy_n_files,
            "buggy_functions": total_buggy_funcs,
            "total_functions": buggy_n_funcs,
            "median_funcs_per_file": _safe_median(buggy_funcs_per_file),
            "median_lines_per_file": _safe_median(buggy_lines_per_file),
            "median_lines_per_func": _safe_median(buggy_func_lengths),
        },
        "nonbuggy": {
            "total_files": nonbuggy_n_files,
            "total_functions": nonbuggy_n_funcs,
            "median_funcs_per_file": _safe_median(nonbuggy_funcs_per_file),
            "median_lines_per_file": _safe_median(nonbuggy_lines_per_file),
            "median_lines_per_func": _safe_median(nonbuggy_func_lengths),
        },
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_table(agg: dict[str, Any]) -> None:
    b = agg["buggy"]
    n = agg["nonbuggy"]

    def fmt_int(v: int) -> str:
        return f"{v:,}"

    def fmt_float(v: float) -> str:
        return f"{v:.2f}"

    rows = [
        ("Characteristic",         "Buggy",                                  "Non-buggy"),
        ("-" * 55,                  "",                                       ""),
        ("Bug Reports",             fmt_int(agg["bug_reports"]),              "---"),
        ("Total Files",             fmt_int(b["total_files"]),                fmt_int(n["total_files"])),
        ("Buggy Functions",         fmt_int(b["buggy_functions"]),            "---"),
        ("Total Functions",         fmt_int(b["total_functions"]),            fmt_int(n["total_functions"])),
        ("Median Functions/File",   fmt_float(b["median_funcs_per_file"]),    fmt_float(n["median_funcs_per_file"])),
        ("Median Lines/File",       fmt_float(b["median_lines_per_file"]),    fmt_float(n["median_lines_per_file"])),
        ("Median Lines/Function",   fmt_float(b["median_lines_per_func"]),    fmt_float(n["median_lines_per_func"])),
    ]

    col_w = [35, 14, 14]
    for label, buggy_val, nonbuggy_val in rows:
        if label.startswith("-"):
            print("-" * (col_w[0] + col_w[1] + col_w[2] + 2))
        else:
            print(f"{label:<{col_w[0]}}{buggy_val:>{col_w[1]}}{nonbuggy_val:>{col_w[2]}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-cache",
        default="./repos",
        help="Directory to clone/cache GitHub repositories. (default: ./repos)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit the number of SWE-Bench instances to process (useful for testing).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write raw per-instance JSON results.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("HuggingFace datasets is required: pip install datasets")

    repo_cache = pathlib.Path(args.repo_cache)
    repo_cache.mkdir(parents=True, exist_ok=True)

    print("Loading SWE-Bench Verified from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"  {len(dataset)} instances loaded.")

    if args.max_instances:
        dataset = dataset.select(range(min(args.max_instances, len(dataset))))
        print(f"  Limiting to {len(dataset)} instances (--max-instances).")

    results: list[dict[str, Any]] = []
    skipped = 0

    for idx, instance in enumerate(dataset):
        iid = instance["instance_id"]
        print(f"[{idx + 1}/{len(dataset)}] {iid}")
        result = process_instance(instance, repo_cache)
        if result is None:
            skipped += 1
        else:
            results.append(result)

    print(f"\nProcessed {len(results)} instances ({skipped} skipped).\n")

    if not results:
        print("No instances were processed — cannot compute metrics.")
        sys.exit(1)

    agg = aggregate_metrics(results)

    print_table(agg)

    if args.output:
        out_path = pathlib.Path(args.output)
        payload = {
            "summary": agg,
            "per_instance": [
                {
                    "instance_id": r["instance_id"],
                    "n_buggy_files": len(r["buggy_files"]),
                    "n_nonbuggy_files": len(r["nonbuggy_files"]),
                    "buggy_funcs": r["buggy_funcs"],
                    "buggy_total_funcs": sum(f["n_funcs"] for f in r["buggy_files"]),
                    "nonbuggy_total_funcs": sum(f["n_funcs"] for f in r["nonbuggy_files"]),
                }
                for r in results
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nRaw data written to: {out_path}")


if __name__ == "__main__":
    main()
