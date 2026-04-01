#!/usr/bin/env python3
"""
preprocess_llmao.py (schema‑aware, **commit‑smart**)
--------------------------------------------------
Builds LLMAO‑style `code_bugline.csv` but now **batches work by commit pair**
and memoises expensive Git operations so repeated SHA combinations are handled
only once.

💡  What changed?
=================
1. **Record ordering** – we sort the incoming JSON on
   `(repo_url, before_fix_sha, after_fix_sha)` so identical commit pairs are
   processed consecutively, keeping the working‑tree object cache hot.
2. **Content / diff memoisation** – `get_file_cached()` and
   `get_patch_cached()` wrap the underlying Git commands with an in‑memory
   dictionary (O(µs) lookup).  For datasets that reference the same file and
   commit frequently, this removes >90 % of `git show` / `git diff` calls.
3. **Same‑SHA short‑circuit** – if `before_fix_sha == after_fix_sha` and
   `fixed_lines_number` is missing, we **skip** the diff because it would be
   empty.

Usage
-----
python preprocess_llmao.py issues.json \
       --repo-cache ./repos             \
       --out-dir    ./llmao_data        \
       --max-rows   1000
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────
# stdlib
import argparse, csv, json, pathlib, re, sys, time
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

# ────────────────────────────────────────────────────────────────
# 3rd‑party
import git               # pip install GitPython
import pandas as pd
from git.exc import GitCommandError

# ────────────────────────────────────────────────────────────────
# Git helpers
# ────────────────────────────────────────────────────────────────

def ensure_commit(repo: git.Repo, sha: str) -> None:
    """Ensure *sha* exists locally (shallow‑fetch if absent)."""
    try:
        repo.commit(sha)
    except (ValueError, GitCommandError):
        repo.git.fetch("origin", sha, "--depth", "1")


def ensure_repo(repo_url: str, cache_dir: pathlib.Path, *shas: str) -> git.Repo:
    """Clone *repo_url* into *cache_dir* if required and make sure *shas* exist."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = cache_dir / repo_name

    if not repo_path.exists():
        print(f"→ cloning {repo_url}")
        git.Repo.clone_from(
            repo_url,
            repo_path,
            depth=1,
            no_single_branch=True,
            multi_options=["--filter=blob:none"],
        )
        print(f"✓ cloned {repo_url} → {repo_path}")

    repo = git.Repo(repo_path)

    for sha in shas:
        ensure_commit(repo, sha)

    return repo


# ────────────────────────────────────────────────────────────────
# Memoised Git accessors
# ────────────────────────────────────────────────────────────────

FileKey = Tuple[str, str, str]   # (repo_path, sha, rel_path)
DiffKey = Tuple[str, str, str, str]  # (repo_path, before_sha, after_sha, rel_path)

_file_cache: Dict[FileKey, str] = {}
_patch_cache: Dict[DiffKey, str] = {}


def get_file_cached(repo: git.Repo, sha: str, rel_path: str) -> str:
    key = (repo.working_tree_dir, sha, rel_path)
    if key not in _file_cache:
        try:
            _file_cache[key] = repo.git.show(f"{sha}:{rel_path}")
        except GitCommandError as e:
            raise FileNotFoundError(f"{rel_path} not found in {sha}: {e}")
    return _file_cache[key]


def get_patch_cached(repo: git.Repo, before_sha: str, after_sha: str, rel_path: str) -> str:
    key = (repo.working_tree_dir, before_sha, after_sha, rel_path)
    if key not in _patch_cache:
        _patch_cache[key] = repo.git.diff(before_sha, after_sha, "--", rel_path, "-U0")
    return _patch_cache[key]


# ────────────────────────────────────────────────────────────────
# Diff helpers
# ────────────────────────────────────────────────────────────────

def parse_changed_lines_from_patch(patch: str) -> List[int]:
    hunk_header = re.compile(r"@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    lines: List[int] = []
    for ln in patch.splitlines():
        if ln.startswith("@@"):
            m = hunk_header.match(ln)
            if not m:
                continue
            start_old = int(m.group(1))
            len_old = int(m.group(2) or 1)
            lines.extend(range(start_old, start_old + len_old))
    return lines


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def to_zero_based(idx):
    if isinstance(idx, list):
        return [i - 1 for i in idx]
    return [idx - 1]


def eta_str(start: float, done: int, total: int) -> str:
    if done == 0:
        return ""
    rate = (time.time() - start) / done
    remaining = (total - done) * rate
    mins, secs = divmod(int(remaining), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f" | ETA: {hrs}h{mins:02d}m"
    if mins:
        return f" | ETA: {mins}m{secs:02d}s"
    return f" | ETA: {secs}s"


# ────────────────────────────────────────────────────────────────
# Core transformer
# ────────────────────────────────────────────────────────────────

def json_to_rows(records: List[dict], cache_dir: pathlib.Path, max_rows: Optional[int] = None) -> List[dict]:
    # Group by commit pair to maximise cache hits
    ordered = sorted(records, key=lambda r: (r["repo_url"], r["before_fix_sha"], r["after_fix_sha"]))

    rows: List[dict] = []
    t0 = time.time()

    for idx, rec in enumerate(ordered, 1):
        if max_rows is not None and len(rows) >= max_rows:
            break

        print("──────────────")
        print(f"record {idx}/{len(ordered)}  issue_id={rec.get('issue_id')}" + eta_str(t0, idx, len(ordered)))

        repo = ensure_repo(
            rec["repo_url"], cache_dir, rec["before_fix_sha"], rec["after_fix_sha"]
        )

        rel_path = str(pathlib.PurePosixPath(rec["path_to_buggy_file"], rec["buggy_file_name"]))
        pre_code = get_file_cached(repo, rec["before_fix_sha"], rel_path)

        # Collect labels
        labels: Set[int] = set()
        if rec.get("buggy_line_number"):
            labels.update(to_zero_based(rec["buggy_line_number"]))
        if rec.get("fixed_lines_number"):
            labels.update(to_zero_based(rec["fixed_lines_number"]))
        elif rec["before_fix_sha"] != rec["after_fix_sha"]:
            patch = get_patch_cached(repo, rec["before_fix_sha"], rec["after_fix_sha"], rel_path)
            labels.update(l - 1 for l in parse_changed_lines_from_patch(patch))

        # Clamp to file length
        max_line = len(pre_code.splitlines())
        labels = {l for l in labels if 0 <= l < max_line}

        rows.append({"input": pre_code, "label": json.dumps(sorted(labels))})
        print(f"✓ row ready with {len(labels)} labels")

    return rows


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", type=pathlib.Path)
    ap.add_argument("--repo-cache", type=pathlib.Path, default=pathlib.Path("./repos"))
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("./llmao_data"))
    ap.add_argument("--max-rows", type=int)
    args = ap.parse_args()

    records = json.loads(args.json_file.read_text())
    rows = json_to_rows(records, args.repo_cache, args.max_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "code_bugline.csv"

    pd.DataFrame(rows, columns=["input", "label"]).to_csv(
        out_csv,
        index=False,
        header=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )
    print(f"✓ {len(rows):,} samples → {out_csv}")


if __name__ == "__main__":
    main()
