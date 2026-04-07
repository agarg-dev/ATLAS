#!/usr/bin/env python3
"""
swebench_pipeline.py
====================
End-to-end pipeline for running LLMAO on SWE-Bench Verified.

Subcommands
-----------
  preprocess   Download SWE-Bench Verified, parse gold patches, checkout repos,
               write LLMAO-format CSVs and ``metadata.json`` (single combined dataset).

  run          Print (or execute) codegen_loading + training for one
               ``data_path`` / ``data_name`` (default: ``data`` / ``swebench``).

  evaluate     Evaluate one checkpoint + ``metadata.json`` (paper protocol).

Paper evaluation protocol (from the paper):
    For each issue ID, the ground-truth file is provided directly to the model.
    The model ranks (1) functions within that file, then (2) lines within those
    functions. The final rank is the highest-ranking path (function + line) that
    matches the ground truth.

Usage examples
--------------
  # Step 1 — preprocess all 500 SWE-Bench Verified instances
  python swebench_pipeline.py preprocess \\
      --repo-cache ./repos \\
      --out-dir    data/swebench \\
      --max-instances 500

  # Step 2 — show commands needed to extract + train
  python swebench_pipeline.py run --pretrain-type 350M

  # Step 3 — evaluate with a trained checkpoint
  python swebench_pipeline.py evaluate \\
      --checkpoint    model_checkpoints/swebench_350M \\
      --pretrain-type 350M \\
      --metadata      data/swebench/metadata.json \\
      --repo-cache    ./repos \\
      --k-values      1 3 5 10

Dependencies
------------
  pip install datasets gitpython torch transformers accelerate pandas
"""

from __future__ import annotations

import argparse
# Ensure the repo root (parent of this custom/ directory) is on sys.path so
# that imports of transformer, codegen, etc. work regardless of CWD.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import ast
import csv
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# Optional heavy imports — only needed for evaluate subcommand
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Git helpers (adapted from data_processing/preprocess_llmao.py)
# ---------------------------------------------------------------------------

def _ensure_repo(repo_url: str, cache_dir: pathlib.Path, *shas: str):
    """Clone repo_url into cache_dir if absent, then fetch the given SHAs."""
    try:
        import git
        from git.exc import GitCommandError
    except ImportError:
        sys.exit("GitPython is required: pip install gitpython")

    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = cache_dir / repo_name

    if not repo_path.exists():
        print(f"  cloning {repo_url} → {repo_path}")
        git.Repo.clone_from(
            repo_url,
            repo_path,
            depth=1,
            no_single_branch=True,
            multi_options=["--filter=blob:none"],
        )

    repo = git.Repo(repo_path)

    for sha in shas:
        try:
            repo.commit(sha)
        except (ValueError, GitCommandError):
            try:
                repo.git.fetch("origin", sha, "--depth", "1")
            except GitCommandError as exc:
                print(f"  warning: could not fetch {sha}: {exc}")

    return repo


def _read_file_at_commit(repo, sha: str, rel_path: str) -> str | None:
    """Return file content at the given commit, or None on failure."""
    try:
        import git
        return repo.git.show(f"{sha}:{rel_path}")
    except Exception as exc:
        print(f"  warning: could not read {rel_path}@{sha}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Unified diff parsing
# ---------------------------------------------------------------------------

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_FILE_OLD = re.compile(r"^--- a/(.+)$")
_DIFF_FILE_NEW = re.compile(r"^\+\+\+ b/(.+)$")


def parse_patch(patch: str) -> dict[str, list[int]]:
    """
    Parse a unified diff patch and return a mapping of
    ``{file_path: [0-based buggy line numbers in the pre-fix file]}``.

    "Buggy lines" are lines that the patch removes or modifies — i.e. lines
    present in the ``-`` (before-fix) version of each hunk.
    """
    result: dict[str, list[int]] = {}
    current_file: str | None = None
    old_line: int = 0  # 1-based current position in the old file

    for raw_line in patch.splitlines():
        m_old = _DIFF_FILE_OLD.match(raw_line)
        if m_old:
            # /dev/null means new file — no pre-fix lines to record
            path = m_old.group(1)
            current_file = None if path == "/dev/null" else path
            continue

        if _DIFF_FILE_NEW.match(raw_line):
            if current_file and current_file not in result:
                result[current_file] = []
            continue

        m_hunk = _HUNK_HEADER.match(raw_line)
        if m_hunk:
            old_line = int(m_hunk.group(1))
            continue

        if current_file is None:
            continue

        if raw_line.startswith("-"):
            # This line exists in the pre-fix file (potentially buggy)
            result.setdefault(current_file, []).append(old_line - 1)  # 0-based
            old_line += 1
        elif raw_line.startswith("+"):
            # Pure addition — no old-file line number consumed
            pass
        else:
            # Context line — present in both versions
            old_line += 1

    # Deduplicate and sort
    return {f: sorted(set(lines)) for f, lines in result.items()}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

MAX_LINES = 128  # matches codegen_loading.py MAX_LEN


def chunk_file(source: str, buggy_lines: list[int]) -> list[dict]:
    """
    Split a source file into non-overlapping 128-line chunks.
    Returns only chunks that contain at least one buggy line, with line
    indices re-based to be 0-relative within each chunk.

    Each returned dict has keys: ``source`` (str), ``buggy_lines`` (list[int]),
    ``chunk_start`` (int, global 0-based start of this chunk).
    """
    all_lines = source.splitlines(keepends=True)
    buggy_set = set(buggy_lines)
    chunks = []

    for start in range(0, len(all_lines), MAX_LINES):
        end = min(start + MAX_LINES, len(all_lines))
        chunk_lines = all_lines[start:end]
        local_bugs = sorted(b - start for b in buggy_set if start <= b < end)
        if local_bugs:
            chunks.append({
                "source": "".join(chunk_lines),
                "buggy_lines": local_bugs,
                "chunk_start": start,
            })

    return chunks


# ---------------------------------------------------------------------------
# Stage 1: preprocess
# ---------------------------------------------------------------------------

def stage_preprocess(args: argparse.Namespace) -> None:
    """
    Download SWE-Bench Verified, parse each gold patch, checkout the
    base_commit, extract the buggy file(s), and write LLMAO-format CSVs.
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
    skipped = 0

    for instance in dataset:
        iid = instance["instance_id"]
        repo = instance["repo"]  # e.g. "django/django"
        base_commit = instance["base_commit"]
        patch = instance["patch"]

        if not patch.strip():
            print(f"[{iid}] No patch — skipping.")
            skipped += 1
            continue

        # Parse the diff to find changed files and buggy lines
        file_buglines = parse_patch(patch)
        if not file_buglines:
            print(f"[{iid}] Patch produced no parseable file/line info — skipping.")
            skipped += 1
            continue

        # Filter to Python files only (SWE-Bench Verified is all Python)
        file_buglines = {
            fp: bl for fp, bl in file_buglines.items() if fp.endswith(".py")
        }
        if not file_buglines:
            print(f"[{iid}] No Python files changed — skipping.")
            skipped += 1
            continue

        repo_url = f"https://github.com/{repo}.git"
        print(f"[{iid}] Checking out {repo}@{base_commit[:8]}...")

        try:
            git_repo = _ensure_repo(repo_url, repo_cache, base_commit)
        except Exception as exc:
            print(f"  error cloning/fetching: {exc} — skipping.")
            skipped += 1
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

            chunks = chunk_file(source, buggy_lines)
            if not chunks:
                # File shorter than MAX_LINES and no buggy lines — shouldn't
                # happen given parse_patch, but handle defensively
                print(f"  {rel_path}: no buggy chunks found — skipping file.")
                continue

            safe_path = rel_path.replace("/", "__").replace("\\", "__")
            file_csv_paths = []

            for chunk_idx, chunk in enumerate(chunks):
                csv_name = f"{iid}__{safe_path}__{chunk_idx}.csv"
                csv_path = out_dir / csv_name

                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_ALL,
                                        escapechar="\\")
                    writer.writerow([chunk["source"],
                                     json.dumps(chunk["buggy_lines"])])

                file_csv_paths.append({
                    "csv": str(csv_path),
                    "chunk_start": chunk["chunk_start"],
                    "local_buggy_lines": chunk["buggy_lines"],
                    "global_buggy_lines": buggy_lines,
                })
                csv_count += 1

            instance_meta["files"][rel_path] = {
                "global_buggy_lines": buggy_lines,
                "chunks": file_csv_paths,
                "full_source_lines": len(source.splitlines()),
                "full_source": source,
            }

        if instance_meta["files"]:
            metadata[iid] = instance_meta

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nPreprocessing complete.")
    print(f"  Instances processed : {len(metadata)}")
    print(f"  CSV files written   : {csv_count}")
    print(f"  Instances skipped   : {skipped}")
    print(f"  Metadata saved to   : {meta_path}")
    parent = out_dir.parent
    data_name = out_dir.name
    dp = str(parent).replace("\\", "/")
    print(f"\nNext step — codegen tensors + train one LLMAO model on all SWE-Bench CSVs:")
    print(
        f'  python custom/swebench_pipeline.py run --data-path "{dp}" '
        f'--data-name "{data_name}" --pretrain-type 350M --execute'
    )


# ---------------------------------------------------------------------------
# Stage 2: run — print (and optionally execute) pipeline commands
# ---------------------------------------------------------------------------

def stage_run(args: argparse.Namespace) -> None:
    """Print or run codegen_loading + training for a single ``data_path`` / ``data_name``."""
    biggest = "1" if args.pretrain_type == "16B" else "0"
    dp = args.data_path.replace("\\", "/")
    dn = args.data_name
    # The upstream scripts live one directory above this file (the repo root).
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
# Python AST-based function extraction
# ---------------------------------------------------------------------------

def get_python_function_ranges(source_code: str) -> list[tuple[str, int, int]]:
    """
    Parse Python source and return a list of
    ``(function_name, start_line_0based, end_line_0based)`` for every
    top-level function and method.

    Uses Python's built-in ``ast`` module (no third-party parser needed).
    Line numbers from the AST are 1-based; this function converts to 0-based.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    ranges: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1          # AST is 1-based → 0-based
            end = (node.end_lineno or node.lineno) - 1
            ranges.append((node.name, start, end))

    return ranges


def map_line_to_function(
    line_idx: int,
    func_ranges: list[tuple[str, int, int]],
) -> str | None:
    """Return the name of the innermost function containing line_idx."""
    best: tuple[str, int, int] | None = None
    for name, start, end in func_ranges:
        if start <= line_idx <= end:
            # Prefer the narrowest (innermost) enclosing function
            if best is None or (end - start) < (best[2] - best[1]):
                best = (name, start, end)
    return best[0] if best else None


def aggregate_function_scores(
    per_line_probs: list[float],
    func_ranges: list[tuple[str, int, int]],
) -> dict[str, float]:
    """
    For each function, aggregate line probabilities via max pooling.
    Returns ``{function_name: max_probability}``.
    """
    scores: dict[str, float] = {}
    for name, start, end in func_ranges:
        window = [
            per_line_probs[i]
            for i in range(start, min(end + 1, len(per_line_probs)))
        ]
        if window:
            scores[name] = max(window)
    return scores


# ---------------------------------------------------------------------------
# LLMAO inference on a single file (mirrors demo.py / TokenizeMask logic)
# ---------------------------------------------------------------------------

def infer_file(source: str, model, tokenize_mask) -> list[float]:
    """
    Run LLMAO inference on a single source string.
    Returns a list of per-line suspiciousness scores (probabilities),
    one value per line in ``source.splitlines()``.

    The scores are aligned to the original physical lines.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for inference.")

    import torch

    # Filter comment lines (Python: #) and empty/trivial lines — mirrors
    # demo.py but adapted for Python (no C-style // or * comment removal).
    lines = source.splitlines(keepends=True)
    filtered_lines = []
    original_indices: list[int] = []  # map filtered → original line index

    for orig_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped in ("{", "}"):
            continue
        filtered_lines.append(line)
        original_indices.append(orig_idx)

    if not filtered_lines:
        return [0.0] * len(lines)

    code_str = "".join(filtered_lines)

    try:
        inp, mask, input_size, _ = tokenize_mask.generate_token_mask(code_str)
    except Exception as exc:
        print(f"  warning: tokenisation failed: {exc}")
        return [0.0] * len(lines)

    inp = inp[None, :]
    mask = mask[None, :]

    with torch.no_grad():
        predictions = model(inp, mask)

    probs = torch.flatten(torch.sigmoid(predictions))
    real_indices = torch.flatten(mask == 1)
    probs = probs[real_indices].tolist()
    probs = probs[:input_size + 1]

    # Map filtered-line probabilities back to original line indices
    result = [0.0] * len(lines)
    for filt_idx, orig_idx in enumerate(original_indices):
        if filt_idx < len(probs):
            result[orig_idx] = probs[filt_idx]

    return result


# ---------------------------------------------------------------------------
# Paper evaluation protocol
# ---------------------------------------------------------------------------

def evaluate_instance(
    source: str,
    global_buggy_lines: list[int],
    chunk_start: int,
    per_line_probs: list[float],
    k_values: list[int],
) -> dict[str, Any]:
    """
    Evaluate one chunk using the paper protocol.

    Paper protocol:
    1. Rank functions by their max-probability line.
    2. Within the top-ranked function, find the highest-probability line.
    3. Report whether the top-ranked function / top-K ranked lines contain a bug.

    Returns a dict with keys:
      ``func_hit`` (top-1 function hit), ``func_hit_{k}`` and ``line_hit_{k}``
      for each ``k`` in ``k_values``,
      ``top_func``, ``top_line``, ``buggy_funcs``,
      ``reciprocal_rank_func``, ``reciprocal_rank_line``.

    ``func_hit_k``: true if some ground-truth buggy function appears in the top-K
    ranked functions. ``line_hit_k``: true if some ground-truth buggy line appears
    in the top-K ranked lines in the chunk.
    """
    func_ranges = get_python_function_ranges(source)

    # Global buggy lines → local (chunk-relative) coordinates
    local_buggy = set(b - chunk_start for b in global_buggy_lines
                      if 0 <= (b - chunk_start) < len(per_line_probs))

    if not local_buggy:
        return {"skipped": True}

    result: dict[str, Any] = {"skipped": False}

    # --- Function-level evaluation ---
    if func_ranges:
        func_scores = aggregate_function_scores(per_line_probs, func_ranges)
        sorted_funcs = sorted(func_scores, key=func_scores.get, reverse=True)

        buggy_funcs = set()
        for bl in local_buggy:
            fn = map_line_to_function(bl, func_ranges)
            if fn:
                buggy_funcs.add(fn)

        top_func = sorted_funcs[0] if sorted_funcs else None
        result["top_func"] = top_func
        result["buggy_funcs"] = sorted(buggy_funcs)

        for k in k_values:
            depth = min(k, len(sorted_funcs))
            topk = sorted_funcs[:depth] if depth > 0 else []
            result[f"func_hit_{k}"] = (
                bool(buggy_funcs) and any(fn in buggy_funcs for fn in topk)
            )
        result["func_hit"] = top_func is not None and top_func in buggy_funcs

        # MRR — rank of first buggy function
        rr_func = 0.0
        for rank, fn in enumerate(sorted_funcs, start=1):
            if fn in buggy_funcs:
                rr_func = 1.0 / rank
                break
        result["reciprocal_rank_func"] = rr_func
    else:
        result["top_func"] = None
        result["buggy_funcs"] = []
        result["reciprocal_rank_func"] = 0.0
        for k in k_values:
            result[f"func_hit_{k}"] = False
        result["func_hit"] = False

    # --- Line-level evaluation (across the entire chunk) ---
    if not per_line_probs:
        for k in k_values:
            result[f"line_hit_{k}"] = False
        result["top_line"] = None
        result["reciprocal_rank_line"] = 0.0
        return result

    sorted_lines = sorted(
        range(len(per_line_probs)),
        key=lambda i: per_line_probs[i],
        reverse=True,
    )
    top_line = sorted_lines[0]
    result["top_line"] = top_line + chunk_start  # global coordinate

    for k in k_values:
        topk = sorted_lines[:k] if k > 0 else []
        result[f"line_hit_{k}"] = any(li in local_buggy for li in topk)

    # Line MRR — rank of first buggy line in probability-sorted order
    rr_line = 0.0
    for rank, li in enumerate(sorted_lines, start=1):
        if li in local_buggy:
            rr_line = 1.0 / rank
            break
    result["reciprocal_rank_line"] = rr_line

    return result


# ---------------------------------------------------------------------------
# Paper evaluation (used by evaluate subcommand)
# ---------------------------------------------------------------------------

def run_paper_evaluation(
    metadata: dict[str, Any],
    checkpoint_path: str | pathlib.Path,
    pretrain_type: str,
    repo_cache: pathlib.Path,
    k_values: list[int],
    *,
    label: str = "",
    quiet: bool = False,
    results_output: pathlib.Path | None = None,
) -> dict[str, Any]:
    """
    Run paper-protocol evaluation; returns aggregate stats and optionally
    writes per-instance JSON to ``results_output``.
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

            global_buggy = file_meta["global_buggy_lines"]

            for chunk_info in file_meta["chunks"]:
                chunk_start = chunk_info["chunk_start"]
                all_lines = source.splitlines(keepends=True)
                chunk_lines = all_lines[chunk_start: chunk_start + MAX_LINES]
                chunk_source = "".join(chunk_lines)

                per_line_probs = infer_file(chunk_source, model, tokenize_mask)

                eval_result = evaluate_instance(
                    source=chunk_source,
                    global_buggy_lines=global_buggy,
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

                instance_results[f"{rel_path}::chunk{chunk_start}"] = eval_result

        if instance_results:
            results_per_instance[iid] = instance_results

    mrr_func = statistics.mean(rr_funcs) if rr_funcs else 0.0
    mrr_line = statistics.mean(rr_lines) if rr_lines else 0.0

    if results_output is not None:
        results_output.write_text(
            json.dumps(results_per_instance, indent=2), encoding="utf-8"
        )

    if not quiet:
        title = label or "SWE-Bench Verified LLMAO Evaluation — Paper Protocol"
        print(f"\n{'=' * 60}")
        print(title)
        print(f"{'=' * 60}")
        print(f"Instances in metadata   : {len(metadata)}")
        print(f"Chunks evaluated        : {total_evaluated}")
        print()

        if total_evaluated == 0:
            print("No chunks were evaluated. Check the metadata and repo cache.")
        else:
            print("Function-level results (Hit@K = any buggy function in top-K ranked):")
            for k in k_values:
                h = func_hits[k]
                print(
                    f"  Hit@{k:<2}               : {h}/{total_evaluated} "
                    f"({100*h/total_evaluated:.1f}%)"
                )
            print(f"  MRR (function)        : {mrr_func:.4f}")
            print()
            print(
                "Line-level results (Hit@K = exact buggy line in top-K ranked lines):"
            )
            for k in k_values:
                h = line_hits[k]
                print(
                    f"  Hit@{k:<2}               : {h}/{total_evaluated} "
                    f"({100*h/total_evaluated:.1f}%)"
                )
            print(f"  MRR (line)            : {mrr_line:.4f}")
        if results_output is not None:
            print(f"\nDetailed results saved to: {results_output}")

    return {
        "label": label,
        "n_metadata_instances": len(metadata),
        "chunks_evaluated": total_evaluated,
        "func_hits": func_hits,
        "line_hits": line_hits,
        "mrr_func": mrr_func,
        "mrr_line": mrr_line,
        "rr_funcs": rr_funcs,
        "rr_lines": rr_lines,
    }


# ---------------------------------------------------------------------------
# Stage 3: evaluate
# ---------------------------------------------------------------------------

def stage_evaluate(args: argparse.Namespace) -> None:
    """
    Load a trained LLMAO checkpoint and evaluate using the paper protocol.
    Reads the metadata.json written by the preprocess stage.
    """
    if not _TORCH_AVAILABLE:
        sys.exit("torch is required for evaluation: pip install torch")

    meta_path = pathlib.Path(args.metadata)
    if not meta_path.exists():
        sys.exit(f"Metadata file not found: {meta_path}. Run preprocess first.")

    metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Loaded metadata for {len(metadata)} instances.")

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    out_path = pathlib.Path(args.output) if args.output else None
    try:
        run_paper_evaluation(
            metadata,
            checkpoint_path,
            args.pretrain_type,
            pathlib.Path(args.repo_cache),
            args.k_values,
            quiet=False,
            results_output=out_path,
        )
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    except RuntimeError as exc:
        sys.exit(str(exc))


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
        help="Download SWE-Bench Verified and write LLMAO-format CSVs.",
    )
    p_pre.add_argument(
        "--repo-cache",
        default="./repos",
        help="Directory to clone/cache GitHub repositories into. (default: ./repos)",
    )
    p_pre.add_argument(
        "--out-dir",
        default="data/swebench",
        help="Directory to write LLMAO CSVs and metadata.json. (default: data/swebench)",
    )
    p_pre.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit the number of SWE-Bench instances to process (useful for testing).",
    )

    # --- run ---
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
        default="swebench",
        help="Subfolder with CSVs under --data-path (default: swebench).",
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
        help="Evaluate a trained LLMAO checkpoint using the paper protocol.",
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
        default="data/swebench/metadata.json",
        help="Path to metadata.json written by preprocess. (default: data/swebench/metadata.json)",
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
        default=[1, 3, 5, 10],
        help=(
            "K list for both metrics: function Hit@K (rank depth in sorted-by-score "
            "function list) and line Hit@K (max line distance from argmax to any bug). "
            "MRR for function and line is always reported. (default: 1 3 5 10)"
        ),
    )
    p_eval.add_argument(
        "--output",
        default=None,
        help="Optional path to write detailed per-instance JSON results.",
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


if __name__ == "__main__":
    main()
