"""
top_k_per_pt.py
===============
Step 3d of the LLMAO evaluation pipeline.

Reads the combined CSVs produced by ``combined_csvs.py`` (columns:
pt_number, line_number, line, is_buggy, probability) and computes:

  Line-level     Hit@K : the argmax-probability line is within ±K physical
                         lines of any ground-truth buggy line.
                 MRR   : mean reciprocal rank of the first buggy line in the
                         probability-sorted ordering, averaged over all PTs.

  Function-level Hit@K : any ground-truth buggy function appears in the top-K
                         functions ranked by max-pooled line probability.
                         Functions are discovered by parsing the reconstructed
                         source with Python's built-in ``ast`` module.
                 MRR   : mean reciprocal rank of the first buggy function,
                         averaged over all PTs.

Source reconstruction
---------------------
The ``line`` column contains individual source lines.  For each pt_number the
lines are sorted by ``line_number`` and joined with ``\\n`` to rebuild the
original chunk before AST parsing.  This avoids storing multi-line text in a
CSV cell (which breaks ``pandas.read_csv``).

Usage
-----
    python top_k_per_pt.py --csv-dir data/combined_csvs --k-values 1 3 5 10
"""
from __future__ import annotations

import argparse
import ast
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Python AST helpers (mirrors swebench_pipeline.py)
# ---------------------------------------------------------------------------

def _get_function_ranges(source: str) -> list[tuple[str, int, int]]:
    """Return (name, start_0based, end_0based) for every function/method."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno) - 1
            ranges.append((node.name, start, end))
    return ranges


def _map_line_to_function(
    line_idx: int,
    func_ranges: list[tuple[str, int, int]],
) -> str | None:
    """Return the name of the innermost function containing line_idx."""
    best: tuple[str, int, int] | None = None
    for name, start, end in func_ranges:
        if start <= line_idx <= end:
            if best is None or (end - start) < (best[2] - best[1]):
                best = (name, start, end)
    return best[0] if best else None


def _aggregate_function_scores(
    line_numbers: list[int],
    probabilities: list[float],
    func_ranges: list[tuple[str, int, int]],
) -> dict[str, float]:
    """Max-pool line probabilities into per-function scores."""
    prob_by_line = dict(zip(line_numbers, probabilities))
    scores: dict[str, float] = {}
    for name, start, end in func_ranges:
        window = [prob_by_line[i] for i in range(start, end + 1) if i in prob_by_line]
        if window:
            scores[name] = max(window)
    return scores


# ---------------------------------------------------------------------------
# Per-PT evaluation
# ---------------------------------------------------------------------------

def _evaluate_pt(group: pd.DataFrame, k_values: list[int]) -> dict | None:
    """
    Evaluate one PT (one source chunk).

    Returns a result dict, or None if the PT has no buggy lines (unevaluable).
    The group rows must already be sorted by line_number (guaranteed by the
    groupby + sort_values in evaluate_file).
    """
    if group["is_buggy"].sum() == 0:
        return None

    line_numbers  = group["line_number"].tolist()
    probabilities = group["probability"].tolist()
    is_buggy_col  = group["is_buggy"].tolist()
    buggy_lines   = [ln for ln, b in zip(line_numbers, is_buggy_col) if b == 1]
    buggy_set     = set(buggy_lines)

    result: dict = {}

    # ---- Line-level ----
    pred_idx  = int(np.argmax(probabilities))
    pred_line = line_numbers[pred_idx]
    min_dist  = min(abs(pred_line - bl) for bl in buggy_lines)
    for k in k_values:
        result[f"line_hit_{k}"] = int(min_dist <= k)

    sorted_by_prob = sorted(
        range(len(line_numbers)),
        key=lambda i: probabilities[i],
        reverse=True,
    )
    rr_line = 0.0
    for rank, idx in enumerate(sorted_by_prob, start=1):
        if line_numbers[idx] in buggy_set:
            rr_line = 1.0 / rank
            break
    result["line_mrr"] = rr_line

    # ---- Function-level ----
    # Reconstruct the source chunk from the sorted line texts.
    # line_numbers are 0-based; they are already sorted because the caller
    # sorts the group by line_number before passing it here.
    source = "\n".join(group["line"].tolist())
    func_ranges = _get_function_ranges(source)

    if not func_ranges:
        for k in k_values:
            result[f"func_hit_{k}"] = 0
        result["func_mrr"] = 0.0
        result["has_functions"] = False
        return result

    result["has_functions"] = True
    func_scores  = _aggregate_function_scores(line_numbers, probabilities, func_ranges)
    sorted_funcs = sorted(func_scores, key=func_scores.__getitem__, reverse=True)

    buggy_funcs: set[str] = set()
    for bl in buggy_lines:
        fn = _map_line_to_function(bl, func_ranges)
        if fn:
            buggy_funcs.add(fn)

    for k in k_values:
        top_k = sorted_funcs[:k]
        result[f"func_hit_{k}"] = int(
            bool(buggy_funcs) and any(f in buggy_funcs for f in top_k)
        )

    rr_func = 0.0
    for rank, fn in enumerate(sorted_funcs, start=1):
        if fn in buggy_funcs:
            rr_func = 1.0 / rank
            break
    result["func_mrr"] = rr_func

    return result


# ---------------------------------------------------------------------------
# File-level aggregation
# ---------------------------------------------------------------------------

def evaluate_file(df: pd.DataFrame, k_values: list[int]) -> dict:
    """Aggregate line- and function-level metrics across all PTs in a file."""
    line_hits      = {k: 0 for k in k_values}
    func_hits      = {k: 0 for k in k_values}
    line_rrs:  list[float] = []
    func_rrs:  list[float] = []
    total_pts      = 0
    pts_with_funcs = 0

    # Sort within each group so line reconstruction is in the correct order.
    for _name, group in df.sort_values("line_number").groupby("pt_number", sort=False):
        pt_result = _evaluate_pt(group, k_values)
        if pt_result is None:
            continue
        total_pts += 1
        for k in k_values:
            line_hits[k] += pt_result[f"line_hit_{k}"]
            func_hits[k] += pt_result[f"func_hit_{k}"]
        line_rrs.append(pt_result["line_mrr"])
        func_rrs.append(pt_result["func_mrr"])
        if pt_result.get("has_functions"):
            pts_with_funcs += 1

    return {
        "total_pts":      total_pts,
        "pts_with_funcs": pts_with_funcs,
        "line_hits":      line_hits,
        "func_hits":      func_hits,
        # Collect the raw RR lists so the caller can compute a true global MRR.
        "line_rrs":       line_rrs,
        "func_rrs":       func_rrs,
        "line_mrr":       float(np.mean(line_rrs)) if line_rrs else 0.0,
        "func_mrr":       float(np.mean(func_rrs)) if func_rrs else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_file_results(name: str, res: dict, k_values: list[int]) -> None:
    n = res["total_pts"]
    if n == 0:
        print("  No evaluable PTs.")
        return

    print(f"  PTs evaluated      : {n}  (with Python functions: {res['pts_with_funcs']})")
    print()
    print("  Line-level (Hit@K = argmax line within ±K of any buggy line):")
    for k in k_values:
        h = res["line_hits"][k]
        print(f"    Hit@{k:<2}  {h:>4}/{n}  ({100 * h / n:5.1f}%)")
    print(f"    MRR     {res['line_mrr']:.4f}")
    print()
    print("  Function-level (Hit@K = buggy function in top-K by max-pooled score):")
    for k in k_values:
        h = res["func_hits"][k]
        print(f"    Hit@{k:<2}  {h:>4}/{n}  ({100 * h / n:5.1f}%)")
    print(f"    MRR     {res['func_mrr']:.4f}")


def _print_overall(all_results: list[dict], k_values: list[int]) -> None:
    total_pts = sum(r["total_pts"] for r in all_results)
    if total_pts == 0:
        print("No PTs found to evaluate.")
        return

    line_hits_total = {k: sum(r["line_hits"][k] for r in all_results) for k in k_values}
    func_hits_total = {k: sum(r["func_hits"][k] for r in all_results) for k in k_values}
    pts_with_funcs  = sum(r["pts_with_funcs"] for r in all_results)

    # True global MRR: average over every individual PT's reciprocal rank,
    # not over per-file averages (which would weight files unequally).
    all_line_rrs = [rr for r in all_results for rr in r["line_rrs"]]
    all_func_rrs = [rr for r in all_results for rr in r["func_rrs"]]
    global_line_mrr = float(np.mean(all_line_rrs)) if all_line_rrs else 0.0
    global_func_mrr = float(np.mean(all_func_rrs)) if all_func_rrs else 0.0

    print(f"  Total PTs          : {total_pts}  (with Python functions: {pts_with_funcs})")
    print()
    print("  Line-level (Hit@K = argmax line within ±K of any buggy line):")
    for k in k_values:
        h = line_hits_total[k]
        print(f"    Hit@{k:<2}  {h:>4}/{total_pts}  ({100 * h / total_pts:5.1f}%)")
    print(f"    MRR     {global_line_mrr:.4f}")
    print()
    print("  Function-level (Hit@K = buggy function in top-K by max-pooled score):")
    for k in k_values:
        h = func_hits_total[k]
        print(f"    Hit@{k:<2}  {h:>4}/{total_pts}  ({100 * h / total_pts:5.1f}%)")
    print(f"    MRR     {global_func_mrr:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-dir",
        default="data/combined_csvs",
        help="Directory containing combined CSV files with probability column.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="K values to evaluate (default: 1 3 5 10).",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir
    if not os.path.isdir(csv_dir):
        print(f"Error: directory '{csv_dir}' not found.")
        return

    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    if not csv_files:
        print(f"No CSV files found in '{csv_dir}'.")
        return

    k_values = sorted(set(args.k_values))
    all_results: list[dict] = []

    for csv_file in csv_files:
        print(f"\n--- {csv_file} ---")
        file_path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print("  [warn] empty file, skipping.")
            continue
        except Exception as exc:
            print(f"  [error] {exc}")
            continue

        required = {"pt_number", "line_number", "line", "is_buggy", "probability"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            print(f"  [error] missing columns: {missing}")
            continue

        df["line"] = df["line"].fillna("").astype(str)
        res = evaluate_file(df, k_values)
        _print_file_results(csv_file, res, k_values)
        all_results.append(res)

    print("\n\n=== Overall Summary ===")
    _print_overall(all_results, k_values)


if __name__ == "__main__":
    main()
