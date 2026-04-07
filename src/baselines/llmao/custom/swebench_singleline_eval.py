#!/usr/bin/env python3
"""
swebench_singleline_eval.py
===========================
Tertiary (third) training/evaluation mechanism for LLMAO on SWE-Bench Verified.

Motivation
----------
The standard SWE-Bench pipeline (``swebench_pipeline.py``) treats multi-line
faults as a single instance with multiple ground-truth buggy lines. When a
fault spans 5 lines, the model only needs to rank *any one of those 5 lines*
highly to score a hit.

This tertiary pipeline targets **single-line fault localization**:

    Every multi-line fault is **split into independent single-line instances**.

A fault touching lines [10, 11, 12] in a file becomes three separate instances,
each with exactly one ground-truth line.  The model must localise each
individual line independently.  This is a stricter, more fine-grained benchmark
that aligns with how many competing fault-localization tools report results
(one prediction per fault location).

Key differences from the standard SWE-Bench pipeline
----------------------------------------------------
* **No line-count filter** — the full SWE-Bench Verified dataset is used,
  including files longer than 128 lines (which are chunked as in the primary
  pipeline).
* **One CSV row per buggy line** — a file with N buggy lines produces N
  separate CSV rows (and N separate ``.pt`` files after CodeGen extraction),
  each labelled with exactly one buggy line.
* **Evaluation unit is a single line** — Hit@K and MRR are computed per
  single-line instance.  Hit@1 (exact match) is the headline metric.

Subcommands
-----------
  preprocess   Download SWE-Bench Verified, split multi-line faults into
               single-line instances, write LLMAO-format CSVs and
               ``metadata_singleline.json``.

  run          Print (or execute) codegen_loading + training commands.

  evaluate     Evaluate a checkpoint on single-line instances using
               exact-match line Hit@K plus function Hit@K and MRR.

  stats        Print dataset statistics vs the primary pipeline.

Usage examples
--------------
  # Step 1 — preprocess (split multi-line faults into single-line instances)
  python custom/swebench_singleline_eval.py preprocess \\
      --repo-cache ./repos \\
      --out-dir    data/swebench_singleline \\
      --max-instances 500

  # Step 2 — extract CodeGen states + train
  python custom/swebench_singleline_eval.py run \\
      --data-path data --data-name swebench_singleline --pretrain-type 350M

  # Step 3 — evaluate with a trained checkpoint
  python custom/swebench_singleline_eval.py evaluate \\
      --checkpoint    model_checkpoints/swebench_singleline_350M \\
      --pretrain-type 350M \\
      --metadata      data/swebench_singleline/metadata_singleline.json \\
      --repo-cache    ./repos \\
      --k-values      0 1 3 5 10

  # Optional — compare dataset sizes vs primary
  python custom/swebench_singleline_eval.py stats \\
      --metadata-primary    data/swebench/metadata.json \\
      --metadata-singleline data/swebench_singleline/metadata_singleline.json

Dependencies
------------
  pip install datasets gitpython torch transformers accelerate
"""

from __future__ import annotations

import argparse
import sys as _sys
import os as _os

# Ensure the repo root (parent of this custom/ directory) is on sys.path so
# that imports of transformer, codegen, etc. work regardless of CWD.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import ast
import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys
import textwrap
from typing import Any

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Re-use helpers from swebench_pipeline (same repo, same custom/ dir)
# ---------------------------------------------------------------------------
from swebench_pipeline import (
    _ensure_repo,
    _read_file_at_commit,
    parse_patch,
    chunk_file,
    get_python_function_ranges,
    map_line_to_function,
    aggregate_function_scores,
    infer_file,
    MAX_LINES,
)

# ---------------------------------------------------------------------------
# Single-line instance splitting
# ---------------------------------------------------------------------------

def split_to_single_line_instances(
    source: str,
    buggy_lines: list[int],
) -> list[dict]:
    """
    Split a multi-line fault into independent single-line instances.

    Each returned dict represents one instance and has the same structure as
    a chunk from ``chunk_file()``, but with exactly one entry in
    ``buggy_lines``.  The chunk boundaries are determined by the same
    128-line windowing used in the primary pipeline, so CodeGen extraction
    sees identical inputs — only the label differs.

    For a file with buggy lines [10, 11, 50] this produces three dicts:
      - chunk containing line 10, labelled [local_index_of_10]
      - chunk containing line 11, labelled [local_index_of_11]
      - chunk containing line 50, labelled [local_index_of_50]

    If two buggy lines fall in the same 128-line chunk they still become two
    separate instances (same source text, different single-line label each).
    """
    all_lines = source.splitlines(keepends=True)
    instances = []

    for buggy_line in sorted(set(buggy_lines)):
        if not (0 <= buggy_line < len(all_lines)):
            continue

        # Determine which 128-line chunk this line belongs to
        chunk_start = (buggy_line // MAX_LINES) * MAX_LINES
        chunk_end = min(chunk_start + MAX_LINES, len(all_lines))
        chunk_source = "".join(all_lines[chunk_start:chunk_end])
        local_bug = buggy_line - chunk_start  # 0-based within chunk

        instances.append({
            "source": chunk_source,
            "buggy_lines": [local_bug],   # exactly one line
            "chunk_start": chunk_start,
            "global_buggy_line": buggy_line,
        })

    return instances


# ---------------------------------------------------------------------------
# Single-line evaluation
# ---------------------------------------------------------------------------

def evaluate_single_line_instance(
    source: str,
    global_buggy_line: int,
    chunk_start: int,
    per_line_probs: list[float],
    k_values: list[int],
) -> dict[str, Any]:
    """
    Evaluate one single-line instance.

    The ground truth is exactly one line (``global_buggy_line``). Metrics:

    Line-level
      Hit@K  : the target line appears exactly within the top-K ranked line
               predictions. K=0 is kept as a legacy alias for exact top-1 match.
      MRR    : reciprocal rank of the target line in descending-probability
               order.

    Function-level
      Hit@K  : the function containing the target line appears in the top-K
               functions ranked by max-pooled line probability.
      MRR    : reciprocal rank of that function.
    """
    func_ranges = get_python_function_ranges(source)
    local_bug = global_buggy_line - chunk_start

    if not (0 <= local_bug < len(per_line_probs)):
        return {"skipped": True}

    result: dict[str, Any] = {
        "skipped": False,
        "global_buggy_line": global_buggy_line,
        "local_buggy_line": local_bug,
    }

    # ---- Line-level ----
    if not per_line_probs:
        for k in k_values:
            result[f"line_hit_{k}"] = False
        result["top_line"] = None
        result["reciprocal_rank_line"] = 0.0
    else:
        sorted_lines = sorted(
            range(len(per_line_probs)),
            key=lambda i: per_line_probs[i],
            reverse=True,
        )
        top_line = sorted_lines[0]
        result["top_line"] = top_line + chunk_start

        for k in k_values:
            if k == 0:
                result[f"line_hit_{k}"] = (top_line == local_bug)
                continue
            result[f"line_hit_{k}"] = (local_bug in sorted_lines[:k])
        rr_line = 0.0
        for rank, li in enumerate(sorted_lines, start=1):
            if li == local_bug:
                rr_line = 1.0 / rank
                break
        result["reciprocal_rank_line"] = rr_line

    # ---- Function-level ----
    if not func_ranges:
        for k in k_values:
            result[f"func_hit_{k}"] = False
        result["top_func"] = None
        result["buggy_func"] = None
        result["reciprocal_rank_func"] = 0.0
        return result

    buggy_func = map_line_to_function(local_bug, func_ranges)
    result["buggy_func"] = buggy_func

    func_scores = aggregate_function_scores(per_line_probs, func_ranges)
    sorted_funcs = sorted(func_scores, key=func_scores.__getitem__, reverse=True)
    result["top_func"] = sorted_funcs[0] if sorted_funcs else None

    for k in k_values:
        top_k_funcs = sorted_funcs[:k] if k > 0 else []
        result[f"func_hit_{k}"] = (
            buggy_func is not None and buggy_func in top_k_funcs
        )

    # func_hit_1 is the canonical function-level exact-match metric
    result[f"func_hit_1"] = (
        buggy_func is not None
        and bool(sorted_funcs)
        and sorted_funcs[0] == buggy_func
    )

    rr_func = 0.0
    for rank, fn in enumerate(sorted_funcs, start=1):
        if fn == buggy_func:
            rr_func = 1.0 / rank
            break
    result["reciprocal_rank_func"] = rr_func

    return result


# ---------------------------------------------------------------------------
# Stage 1: preprocess
# ---------------------------------------------------------------------------

def stage_preprocess(args: argparse.Namespace) -> None:
    """
    Download SWE-Bench Verified, split every multi-line fault into
    independent single-line instances, and write LLMAO-format CSVs.

    Each CSV row contains:
      col 0 — the 128-line chunk source (same windowing as primary pipeline)
      col 1 — JSON list with exactly one element: the local (chunk-relative)
               0-based index of the single target buggy line

    Files longer than 128 lines are chunked (not filtered out), matching the
    primary pipeline's behaviour.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("HuggingFace datasets is required: pip install datasets")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_cache = pathlib.Path(args.repo_cache)

    print("Loading SWE-Bench Verified from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"  {len(dataset)} instances loaded.")

    if args.max_instances:
        dataset = dataset.select(range(min(args.max_instances, len(dataset))))
        print(f"  Limiting to {len(dataset)} instances (--max-instances).")

    metadata: dict[str, Any] = {}

    csv_count = 0
    single_line_count = 0   # total single-line instances written
    multi_line_faults = 0   # faults that were split (had >1 buggy line)
    skipped_no_patch = 0
    skipped_no_py = 0
    skipped_repo_err = 0

    for instance in dataset:
        iid = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        patch = instance["patch"]

        if not patch.strip():
            skipped_no_patch += 1
            continue

        file_buglines = parse_patch(patch)
        if not file_buglines:
            skipped_no_patch += 1
            continue

        file_buglines = {
            fp: bl for fp, bl in file_buglines.items() if fp.endswith(".py")
        }
        if not file_buglines:
            skipped_no_py += 1
            continue

        repo_url = f"https://github.com/{repo}.git"
        print(f"[{iid}] Checking out {repo}@{base_commit[:8]}...")

        try:
            git_repo = _ensure_repo(repo_url, repo_cache, base_commit)
        except Exception as exc:
            print(f"  error cloning/fetching: {exc} — skipping.")
            skipped_repo_err += 1
            continue

        instance_meta: dict[str, Any] = {
            "repo": repo,
            "base_commit": base_commit,
            "files": {},
        }

        for rel_path, buggy_lines in file_buglines.items():
            source = _read_file_at_commit(git_repo, base_commit, rel_path)
            if source is None:
                continue

            total_lines = len(source.splitlines())
            valid_bugs = [b for b in buggy_lines if 0 <= b < total_lines]
            if not valid_bugs:
                continue

            if len(valid_bugs) > 1:
                multi_line_faults += 1

            instances = split_to_single_line_instances(source, valid_bugs)
            if not instances:
                continue

            safe_path = rel_path.replace("/", "__").replace("\\", "__")
            file_instance_meta: list[dict[str, Any]] = []

            for inst in instances:
                global_bug = inst["global_buggy_line"]
                csv_name = f"{iid}__{safe_path}__line{global_bug}.csv"
                csv_path = out_dir / csv_name

                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\")
                    writer.writerow([inst["source"], json.dumps(inst["buggy_lines"])])

                file_instance_meta.append({
                    "csv": str(csv_path),
                    "chunk_start": inst["chunk_start"],
                    "global_buggy_line": global_bug,
                    "local_buggy_line": inst["buggy_lines"][0],
                })
                csv_count += 1
                single_line_count += 1

            instance_meta["files"][rel_path] = {
                "total_lines": total_lines,
                "all_global_buggy_lines": valid_bugs,
                "full_source": source,
                "single_line_instances": file_instance_meta,
            }

        if instance_meta["files"]:
            metadata[iid] = instance_meta

    meta_path = out_dir / "metadata_singleline.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSingle-line preprocessing complete.")
    print(f"  Instances processed          : {len(metadata)}")
    print(f"  Single-line CSV files written: {csv_count}")
    print(f"  Total single-line instances  : {single_line_count}")
    print(f"  Multi-line faults split      : {multi_line_faults}")
    print(f"  Instances skipped (no patch) : {skipped_no_patch}")
    print(f"  Instances skipped (no Python): {skipped_no_py}")
    print(f"  Instances skipped (repo err) : {skipped_repo_err}")
    print(f"  Metadata saved to            : {meta_path}")

    parent = out_dir.parent
    data_name = out_dir.name
    dp = str(parent).replace("\\", "/")
    print(f"\nNext step — extract CodeGen tensors and train:")
    print(
        f'  python custom/swebench_singleline_eval.py run '
        f'--data-path "{dp}" --data-name "{data_name}" --pretrain-type 350M --execute'
    )


# ---------------------------------------------------------------------------
# Stage 2: run
# ---------------------------------------------------------------------------

def stage_run(args: argparse.Namespace) -> None:
    """Print or execute codegen_loading + training for the single-line dataset."""
    biggest = "1" if args.pretrain_type == "16B" else "0"
    dp = args.data_path.replace("\\", "/")
    dn = args.data_name
    _root = pathlib.Path(__file__).parent.parent
    _codegen_loading = _root / "codegen_loading.py"
    _training = _root / "training.py"
    commands = [
        f'python "{_codegen_loading}" "{dp}" "{dn}" {biggest}',
        f'python "{_training}" "{dp}" "{dn}" {args.pretrain_type} 1',
    ]

    print("Commands to run (in order):\n")
    for cmd in commands:
        print(f"  {cmd}")

    if args.execute:
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                sys.exit(f"Command failed with exit code {result.returncode}: {cmd}")
    else:
        print(
            "\nAdd --execute to run these commands directly, or copy-paste them."
        )


# ---------------------------------------------------------------------------
# Stage 3: evaluate
# ---------------------------------------------------------------------------

def run_singleline_evaluation(
    metadata: dict[str, Any],
    checkpoint_path: str | pathlib.Path,
    pretrain_type: str,
    repo_cache: pathlib.Path,
    k_values: list[int],
    *,
    quiet: bool = False,
    results_output: pathlib.Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate a checkpoint on single-line instances.

    For each instance the model sees the same 128-line chunk as during
    training.  The ground truth is exactly one line.  Metrics are computed
    per instance and then averaged.

    The headline metric is ``line_hit_0`` (exact match: argmax line ==
    target line).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for evaluation.")

    import torch

    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not quiet:
        print(f"Loading LLMAO checkpoint: {checkpoint_path}")

    try:
        from transformer import VoltronTransformerPretrained, TokenizeMask, get_model_config
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformer.py. Run from the LLMAO repository root."
        ) from exc

    cfg = get_model_config(pretrain_type)
    model = VoltronTransformerPretrained(
        num_layer=cfg["num_layer"],
        dim_model=cfg["dim_model"],
        num_head=cfg["num_head"],
        target_dim=cfg["target_dim"],
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu"),
        strict=False,
    )
    model.eval()

    if not quiet:
        print(f"Loading TokenizeMask for {pretrain_type}...")
    tokenize_mask = TokenizeMask(pretrain_type)

    k_values = sorted(set(k_values))
    func_hits = {k: 0 for k in k_values}
    line_hits = {k: 0 for k in k_values}
    rr_funcs: list[float] = []
    rr_lines: list[float] = []
    total_evaluated = 0
    results_per_instance: dict[str, Any] = {}

    # Cache inferred chunks: (repo_url, base_commit, rel_path, chunk_start)
    # so we don't re-run the model on the same chunk for each buggy line.
    _chunk_cache: dict[tuple, list[float]] = {}

    for iid, inst_meta in metadata.items():
        repo_url = f"https://github.com/{inst_meta['repo']}.git"
        base_commit = inst_meta["base_commit"]

        try:
            git_repo = _ensure_repo(repo_url, repo_cache, base_commit)
        except Exception as exc:
            if not quiet:
                print(f"[{iid}] Could not access repo: {exc} — skipping.")
            continue

        instance_results: dict[str, Any] = {}

        for rel_path, file_meta in inst_meta["files"].items():
            source = _read_file_at_commit(git_repo, base_commit, rel_path)
            if source is None:
                continue

            all_lines = source.splitlines(keepends=True)

            for sl_inst in file_meta["single_line_instances"]:
                global_bug = sl_inst["global_buggy_line"]
                chunk_start = sl_inst["chunk_start"]

                # Re-use cached inference for this chunk if available
                cache_key = (repo_url, base_commit, rel_path, chunk_start)
                if cache_key not in _chunk_cache:
                    chunk_end = min(chunk_start + MAX_LINES, len(all_lines))
                    chunk_source = "".join(all_lines[chunk_start:chunk_end])
                    _chunk_cache[cache_key] = infer_file(
                        chunk_source, model, tokenize_mask
                    )

                per_line_probs = _chunk_cache[cache_key]

                chunk_end = min(chunk_start + MAX_LINES, len(all_lines))
                chunk_source = "".join(all_lines[chunk_start:chunk_end])

                eval_result = evaluate_single_line_instance(
                    source=chunk_source,
                    global_buggy_line=global_bug,
                    chunk_start=chunk_start,
                    per_line_probs=per_line_probs,
                    k_values=k_values,
                )

                if eval_result.get("skipped"):
                    continue

                total_evaluated += 1
                for k in k_values:
                    if eval_result.get(f"func_hit_{k}"):
                        func_hits[k] += 1
                    if eval_result.get(f"line_hit_{k}"):
                        line_hits[k] += 1
                rr_funcs.append(eval_result["reciprocal_rank_func"])
                rr_lines.append(eval_result["reciprocal_rank_line"])

                instance_results[f"{rel_path}::line{global_bug}"] = eval_result

        if instance_results:
            results_per_instance[iid] = instance_results

    mrr_func = statistics.mean(rr_funcs) if rr_funcs else 0.0
    mrr_line = statistics.mean(rr_lines) if rr_lines else 0.0

    if results_output is not None:
        results_output.write_text(
            json.dumps(results_per_instance, indent=2), encoding="utf-8"
        )

    if not quiet:
        print(f"\n{'=' * 60}")
        print("SWE-Bench Verified LLMAO — Single-Line Evaluation")
        print("(full dataset; multi-line faults split into single-line instances)")
        print(f"{'=' * 60}")
        print(f"Instances in metadata        : {len(metadata)}")
        print(f"Single-line instances eval'd : {total_evaluated}")
        print()

        if total_evaluated == 0:
            print("No instances were evaluated. Check the metadata and repo cache.")
        else:
            print("Line-level results:")
            print("  (Hit@0 = exact top-1 match; Hit@K = exact target line in top-K ranked lines)")
            for k in k_values:
                h = line_hits[k]
                label = "exact match" if k == 0 else f"in top-{k}"
                print(
                    f"  Hit@{k:<2} ({label:<16}): "
                    f"{h}/{total_evaluated} ({100 * h / total_evaluated:.1f}%)"
                )
            print(f"  MRR (line)                  : {mrr_line:.4f}")
            print()
            print("Function-level results (Hit@K = buggy function in top-K ranked):")
            for k in k_values:
                if k == 0:
                    continue  # function Hit@0 is always 0 (can't be in top-0)
                h = func_hits[k]
                print(
                    f"  Hit@{k:<2}                      : "
                    f"{h}/{total_evaluated} ({100 * h / total_evaluated:.1f}%)"
                )
            print(f"  MRR (function)              : {mrr_func:.4f}")
        if results_output is not None:
            print(f"\nDetailed results saved to: {results_output}")

    return {
        "n_metadata_instances": len(metadata),
        "instances_evaluated": total_evaluated,
        "func_hits": func_hits,
        "line_hits": line_hits,
        "mrr_func": mrr_func,
        "mrr_line": mrr_line,
        "rr_funcs": rr_funcs,
        "rr_lines": rr_lines,
    }


def stage_evaluate(args: argparse.Namespace) -> None:
    """Load a trained checkpoint and run the single-line evaluation protocol."""
    if not _TORCH_AVAILABLE:
        sys.exit("torch is required for evaluation: pip install torch")

    meta_path = pathlib.Path(args.metadata)
    if not meta_path.exists():
        sys.exit(f"Metadata file not found: {meta_path}. Run preprocess first.")

    metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Loaded single-line metadata for {len(metadata)} instances.")

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    out_path = pathlib.Path(args.output) if args.output else None
    try:
        run_singleline_evaluation(
            metadata,
            args.checkpoint,
            args.pretrain_type,
            pathlib.Path(args.repo_cache),
            args.k_values,
            quiet=False,
            results_output=out_path,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(str(exc))


# ---------------------------------------------------------------------------
# Stage 4: stats
# ---------------------------------------------------------------------------

def stage_stats(args: argparse.Namespace) -> None:
    """
    Print a comparison of the primary and single-line datasets.

    Shows how splitting multi-line faults changes instance counts and
    what fraction of faults are single-line vs multi-line in the original data.
    """
    primary_path = pathlib.Path(args.metadata_primary)
    singleline_path = pathlib.Path(args.metadata_singleline)

    primary: dict[str, Any] = {}
    singleline: dict[str, Any] = {}

    if primary_path.exists():
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
    else:
        print(f"[warn] primary metadata not found: {primary_path}")

    if singleline_path.exists():
        singleline = json.loads(singleline_path.read_text(encoding="utf-8"))
    else:
        print(f"[warn] single-line metadata not found: {singleline_path}")

    # Primary stats
    p_instances = len(primary)
    p_files = sum(len(v["files"]) for v in primary.values())
    p_chunks = sum(
        len(f["chunks"])
        for v in primary.values()
        for f in v["files"].values()
        if "chunks" in f
    )

    # Single-line stats
    s_instances = len(singleline)
    s_files = sum(len(v["files"]) for v in singleline.values())
    s_total_sl = sum(
        len(f["single_line_instances"])
        for v in singleline.values()
        for f in v["files"].values()
    )

    # Fault-size distribution from single-line metadata
    fault_sizes: list[int] = []
    for v in singleline.values():
        for f in v["files"].values():
            fault_sizes.append(len(f.get("all_global_buggy_lines", [])))

    single_line_faults = sum(1 for s in fault_sizes if s == 1)
    multi_line_faults = sum(1 for s in fault_sizes if s > 1)

    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:.1f}%" if d else "N/A"

    print(f"\n{'=' * 62}")
    print("Dataset comparison: primary vs single-line split")
    print("  Single-line: multi-line faults split into individual instances")
    print(f"{'=' * 62}")
    print(f"{'Metric':<34} {'Primary':>10} {'Single-line':>12}")
    print(f"{'-' * 62}")
    print(f"{'Instances (SWE-bench IDs)':<34} {p_instances:>10} {s_instances:>12}")
    print(f"{'Files (buggy, Python)':<34} {p_files:>10} {s_files:>12}")
    print(f"{'Training samples (CSV rows)':<34} {p_chunks:>10} {s_total_sl:>12}")
    print(f"{'=' * 62}")

    if fault_sizes:
        print(f"\nFault-size distribution (from single-line metadata):")
        print(f"  Single-line faults (1 buggy line) : {single_line_faults}  ({pct(single_line_faults, len(fault_sizes))})")
        print(f"  Multi-line faults  (>1 buggy line): {multi_line_faults}  ({pct(multi_line_faults, len(fault_sizes))})")
        if fault_sizes:
            print(f"  Mean buggy lines per fault        : {statistics.mean(fault_sizes):.2f}")
            print(f"  Max buggy lines in one fault      : {max(fault_sizes)}")

        # Size histogram
        buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, None)]
        print(f"\n  Buggy-lines-per-fault histogram:")
        for lo, hi in buckets:
            if hi is None:
                count = sum(1 for s in fault_sizes if s >= lo)
                label = f"{lo}+"
            else:
                count = sum(1 for s in fault_sizes if lo <= s <= hi)
                label = f"{lo}" if lo == hi else f"{lo}–{hi}"
            print(f"    {label:<6} lines : {count:>4}  ({pct(count, len(fault_sizes))})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- preprocess ---
    p_pre = sub.add_parser(
        "preprocess",
        help=(
            "Split multi-line faults into single-line instances and write "
            "LLMAO-format CSVs (full dataset, no line-count filter)."
        ),
    )
    p_pre.add_argument(
        "--repo-cache",
        default="./repos",
        help="Directory to clone/cache GitHub repositories. (default: ./repos)",
    )
    p_pre.add_argument(
        "--out-dir",
        default="data/swebench_singleline",
        help=(
            "Directory to write LLMAO CSVs and metadata_singleline.json. "
            "(default: data/swebench_singleline)"
        ),
    )
    p_pre.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit the number of SWE-Bench instances processed (for testing).",
    )

    # --- run ---
    p_run = sub.add_parser(
        "run",
        help="Print (or execute) codegen_loading + training for the single-line dataset.",
    )
    p_run.add_argument(
        "--data-path",
        default="data",
        help="Parent folder passed to codegen_loading / training. (default: data)",
    )
    p_run.add_argument(
        "--data-name",
        default="swebench_singleline",
        help="Subfolder with CSVs under --data-path. (default: swebench_singleline)",
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

    # --- evaluate ---
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate a trained LLMAO checkpoint on single-line instances.",
    )
    p_eval.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the LLMAO checkpoint (model_checkpoints/<name>).",
    )
    p_eval.add_argument(
        "--pretrain-type",
        default="350M",
        choices=["350M", "2B", "6B", "16B"],
        help="CodeGen model size matching the checkpoint. (default: 350M)",
    )
    p_eval.add_argument(
        "--metadata",
        default="data/swebench_singleline/metadata_singleline.json",
        help=(
            "Path to metadata_singleline.json written by preprocess. "
            "(default: data/swebench_singleline/metadata_singleline.json)"
        ),
    )
    p_eval.add_argument(
        "--repo-cache",
        default="./repos",
        help="Directory containing cloned repositories. (default: ./repos)",
    )
    p_eval.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[0, 1, 3, 5, 10],
        help=(
            "K values for Hit@K metrics. K=0 means exact match. "
            "(default: 0 1 3 5 10)"
        ),
    )
    p_eval.add_argument(
        "--output",
        default=None,
        help="Optional path to write detailed per-instance JSON results.",
    )

    # --- stats ---
    p_stats = sub.add_parser(
        "stats",
        help="Compare primary vs single-line dataset sizes and fault-size distribution.",
    )
    p_stats.add_argument(
        "--metadata-primary",
        default="data/swebench/metadata.json",
        help="Path to primary metadata.json. (default: data/swebench/metadata.json)",
    )
    p_stats.add_argument(
        "--metadata-singleline",
        default="data/swebench_singleline/metadata_singleline.json",
        help=(
            "Path to metadata_singleline.json. "
            "(default: data/swebench_singleline/metadata_singleline.json)"
        ),
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
        stage_evaluate(args)
    elif args.command == "stats":
        stage_stats(args)


if __name__ == "__main__":
    main()
