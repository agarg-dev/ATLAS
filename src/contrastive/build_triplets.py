import ast
import os
import json
import re
import random
import argparse
import shutil
import subprocess
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def clone_repo(repo_url, dest, cache_dir):
    """Clone repo from local cache if available, otherwise from remote.

    If cache_dir/<repo_name> exists, clones from there (fast, no network).
    Otherwise clones from repo_url and saves a copy to the cache for future runs.
    Returns True on success, False on failure.
    """
    repo_name = repo_url.split('/')[-1]
    cached = Path(cache_dir) / repo_name

    if cached.exists():
        print(f"  Cloning {repo_name} from local cache...")
        result = subprocess.run(
            ["git", "clone", str(cached), str(dest)],
            capture_output=True,
        )
        if result.returncode == 0:
            return True
        print(f"  Local clone failed ({result.stderr.decode().strip()}), trying remote...")

    print(f"  Cloning {repo_name} from {repo_url}...")
    result = subprocess.run(
        ["git", "clone", repo_url, str(dest)],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  Remote clone failed: {result.stderr.decode().strip()}")
        return False

    if not cached.exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Caching {repo_name} to {cached}...")
        subprocess.run(["git", "clone", repo_url, str(cached)], capture_output=True)

    return True


def checkout(repo_dir, sha):
    """Checkout a specific commit in the repo. Returns True on success."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", sha, "-q"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  Checkout {sha[:7]} failed: {result.stderr.decode().strip()}")
        return False
    return True


def list_head_files(repo_dir, extensions):
    """List all files at HEAD as (filepath, blob_hash) tuples using git ls-tree.

    No checkout required — reads directly from the git object store.
    Returns an empty list on failure.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-tree", "-r", "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    files = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        filepath = parts[1]
        if not any(filepath.endswith(ext) for ext in extensions):
            continue
        blob_hash = parts[0].split()[2]
        files.append((filepath, blob_hash))
    return files


def read_blob(repo_dir, blob_hash):
    """Read a file's content directly from git's object store."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-p", blob_hash],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return result.stdout.decode("utf-8", errors="replace")


def _extract_java_functions(content):
    """Extract Java method spans from file content using regex + brace counting.

    Uses the same regex as the original extract_buggy_function_spans.py.
    Returns list of {name, start_line, end_line}.
    """
    functions = []
    method_pattern = r'(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *\{?'

    for match in re.finditer(method_pattern, content):
        method_start_line = content[:match.start()].count('\n') + 1
        method_name = match.group(2)
        open_braces = 0
        close_pos = match.end()

        for i in range(match.end(), len(content)):
            if content[i] == '{':
                open_braces += 1
            elif content[i] == '}':
                if open_braces == 0:
                    close_pos = i
                    break
                open_braces -= 1

        method_end_line = content[:close_pos].count('\n') + 1
        functions.append({
            'name': method_name,
            'start_line': method_start_line,
            'end_line': method_end_line,
        })

    return functions


def _extract_python_functions(content):
    """Extract Python function/method spans using the stdlib ast module (Python 3.8+).

    Returns list of {name, start_line, end_line}.
    Falls back to an empty list on SyntaxError.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                'name':       node.name,
                'start_line': node.lineno,
                'end_line':   node.end_lineno,
            })
    return functions


def extract_functions(content, ext='.java'):
    """Extract function/method spans from file content.

    Dispatches to the language-specific extractor based on file extension.
    Returns list of {name, start_line, end_line}.
    """
    if ext == '.py':
        return _extract_python_functions(content)
    return _extract_java_functions(content)


def find_function_for_line(content, line_num, ext='.java'):
    """Return (func_name, start_line, end_line) for the span containing line_num.

    Consolidates analyze_file_structure() + update_issue_with_file_info() from the
    original extract_buggy_function_spans.py into one function.

    Returns:
      (None, 0, 0)              — line_num out of range (caller skips record)
      (func_name, start, end)   — line is inside a named function
      ("global", start, end)    — line is in global/top-level scope
    """
    lines = content.splitlines()
    if line_num > len(lines):
        return (None, 0, 0)

    functions = extract_functions(content, ext)

    # Check named functions first
    for func in functions:
        if func['start_line'] <= line_num <= func['end_line']:
            return (func['name'], func['start_line'], func['end_line'])

    # Line is in global scope — compute contiguous global spans
    all_lines = set(range(1, len(lines) + 1))
    for func in functions:
        all_lines -= set(range(func['start_line'], func['end_line'] + 1))

    global_lines = sorted(all_lines)
    global_spans = []
    if global_lines:
        span_start = global_lines[0]
        prev = global_lines[0]
        for ln in global_lines[1:]:
            if ln > prev + 1:
                global_spans.append((span_start, prev))
                span_start = ln
            prev = ln
        global_spans.append((span_start, prev))

    for span_start, span_end in global_spans:
        if span_start <= line_num <= span_end:
            return ("global", span_start, span_end)

    return ("global", line_num, line_num)


def main():
    parser = argparse.ArgumentParser(
        description="Build triplet dataset from bug localization records in one git pass."
    )
    parser.add_argument(
        "--input_path", default="../../data/bug_localization_dataset.json",
        help="Path to bug_localization_dataset.json produced by parse_elasticsearch.py.",
    )
    parser.add_argument(
        "--output_path", default="../../data/triplet_dataset.json",
        help="Where to write the output triplet_dataset.json.",
    )
    parser.add_argument(
        "--repos_cache_dir", default="../../data/repos",
        help="Permanent local mirror directory for cloned repos.",
    )
    parser.add_argument(
        "--temp_dir", default=None,
        help="Working directory for temporary checkouts. "
             "Use a fast local disk path for speed. "
             "Defaults to ../../data/temp_repos if not set.",
    )
    parser.add_argument(
        "--extensions", nargs="+", default=[".java"],
        help="File extensions to consider for negative sampling.",
    )
    args = parser.parse_args()

    temp_root = Path(args.temp_dir) if args.temp_dir else Path("../../data/temp_repos")
    cache_dir = Path(args.repos_cache_dir)
    extensions = args.extensions

    print(f"Loading {args.input_path}...")
    with open(args.input_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} records.")

    # Pre-git validation — skip records missing required fields before any I/O.
    passing = []
    pre_git_skipped = 0
    for bug in dataset:
        if not bug.get('path_to_buggy_file'):
            pre_git_skipped += 1; continue
        if not bug.get('buggy_file_name'):
            pre_git_skipped += 1; continue
        if not bug.get('buggy_line_number'):
            pre_git_skipped += 1; continue
        if not bug.get('stack_trace', '').strip():
            pre_git_skipped += 1; continue
        if not bug.get('fixed_lines_number'):
            pre_git_skipped += 1; continue
        if not bug.get('fixed_lines_content'):
            pre_git_skipped += 1; continue
        if bug.get('fixed_lines_span', [0, 0]) == [0, 0]:
            pre_git_skipped += 1; continue
        passing.append(bug)

    print(f"Pre-git filter: {pre_git_skipped} skipped, {len(passing)} passing.")

    # Group by (repo_url, before_fix_sha) — one clone + one checkout per SHA.
    groups = defaultdict(lambda: defaultdict(list))
    for bug in passing:
        groups[bug['repo_url']][bug['before_fix_sha']].append(bug)

    triplets = []
    post_git_skipped = 0

    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        for repo_url, sha_map in tqdm(groups.items(), desc="Repos"):
            repo_name = repo_url.split('/')[-1]
            temp_repo = temp_root / repo_name

            if temp_repo.exists():
                shutil.rmtree(temp_repo)

            if not clone_repo(repo_url, temp_repo, cache_dir):
                n = sum(len(bugs) for bugs in sha_map.values())
                print(f"  Skipping repo {repo_name} ({n} bugs).")
                post_git_skipped += n
                continue

            # Build HEAD file list once per repo for negative sampling.
            # Using HEAD keeps negatives consistent regardless of which
            # before_fix_sha is currently checked out.
            head_files = list_head_files(temp_repo, extensions)
            if not head_files:
                n = sum(len(bugs) for bugs in sha_map.values())
                print(f"  WARNING: No HEAD files found for {repo_name}, skipping ({n} bugs).")
                post_git_skipped += n
                continue

            for sha, bugs in sha_map.items():
                if not checkout(temp_repo, sha):
                    post_git_skipped += len(bugs)
                    continue

                # Share file content across bugs at the same SHA
                file_cache = {}

                for bug in bugs:
                    rel_path = os.path.join(bug['path_to_buggy_file'], bug['buggy_file_name'])

                    if rel_path not in file_cache:
                        try:
                            file_cache[rel_path] = (temp_repo / rel_path).read_text(
                                encoding='utf-8', errors='replace'
                            )
                        except OSError:
                            file_cache[rel_path] = None

                    content = file_cache[rel_path]
                    if not content or not content.strip():
                        post_git_skipped += 1
                        continue

                    ext = os.path.splitext(bug['buggy_file_name'])[1].lower()
                    func_name, span_start, span_end = find_function_for_line(
                        content, bug['buggy_line_number'], ext
                    )
                    if func_name is None or (span_start == 0 and span_end == 0):
                        post_git_skipped += 1
                        continue

                    # Sample negative from HEAD via git object store.
                    # Per-bug RNG keeps results reproducible regardless of
                    # processing order.
                    rng = random.Random(hash((bug['issue_id'], bug['buggy_file_name'], bug['buggy_line_number'])) % (2 ** 32))
                    neg_pool = [(fp, bh) for fp, bh in head_files if fp != rel_path]
                    rng.shuffle(neg_pool)

                    negative = None
                    candidate = None
                    for neg_path, blob_hash in neg_pool:
                        text = read_blob(temp_repo, blob_hash)
                        if text and text.strip():
                            negative = text
                            candidate = neg_path
                            break

                    if negative is None:
                        post_git_skipped += 1
                        continue

                    triplets.append({
                        "issue_id":            bug['issue_id'],
                        "repo_name":           bug['repo_name'].split('/')[-1],
                        "buggy_file_name":     bug['buggy_file_name'],
                        "buggy_function_name": func_name,
                        "buggy_line_number":   bug['buggy_line_number'],
                        "path_to_buggy_file":  bug['path_to_buggy_file'],
                        "before_fix_sha":      bug['before_fix_sha'],
                        "anchor":              bug['stack_trace'],
                        "positive":            content,
                        "positive_path":       rel_path,
                        "negative":            negative,
                        "negative_path":       candidate,
                    })
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"\nTotal input:      {len(dataset)}")
    print(f"Pre-git skipped:  {pre_git_skipped}")
    print(f"Post-git skipped: {post_git_skipped}")
    print(f"Triplets written: {len(triplets)}")

    with open(args.output_path, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Saved to {args.output_path}")

    # Dataset statistics (for paper Table 2)
    if triplets:
        unique_issues = set(t['issue_id'] for t in triplets)
        unique_repos = set(t['repo_name'] for t in triplets)
        unique_files = set((t['issue_id'], t['buggy_file_name']) for t in triplets)
        unique_negatives = set(hash(t['negative']) for t in triplets)
        lines_per_issue = Counter(t['issue_id'] for t in triplets)

        print(f"\nDataset Statistics (Table 2):")
        print(f"  Unique issues:     {len(unique_issues)}")
        print(f"  Unique repos:      {len(unique_repos)}")
        print(f"  Total triplets:    {len(triplets)}")
        print(f"  Unique buggy files:{len(unique_files)}")
        print(f"  Unique negatives:  {len(unique_negatives)}")
        print(f"  Avg lines/issue:   {len(triplets)/len(unique_issues):.1f}")
        print(f"  Repos: {sorted(unique_repos)}")

        stats = {
            "unique_issues": len(unique_issues),
            "unique_repos": len(unique_repos),
            "total_triplets": len(triplets),
            "unique_buggy_files": len(unique_files),
            "unique_negatives": len(unique_negatives),
            "avg_lines_per_issue": round(len(triplets) / len(unique_issues), 1),
            "repos": sorted(unique_repos),
        }
        stats_path = os.path.join(os.path.dirname(args.output_path), "dataset_statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")

        # Issue-level stratified split (80/10/10), stratified by repo
        # Build issue -> repo mapping (use first repo seen for each issue)
        issue_repo = {}
        for t in triplets:
            if t['issue_id'] not in issue_repo:
                issue_repo[t['issue_id']] = t['repo_name']

        sorted_issues = sorted(unique_issues)
        repo_issue_count = Counter(issue_repo[iid] for iid in sorted_issues)

        # Repos with only 1 issue can't be stratified (sklearn requires >= 2 per class).
        # Assign singleton-repo issues directly to train; stratify the rest.
        singleton_issues = [iid for iid in sorted_issues if repo_issue_count[issue_repo[iid]] == 1]
        multi_issues = [iid for iid in sorted_issues if repo_issue_count[issue_repo[iid]] > 1]
        multi_labels = [issue_repo[iid] for iid in multi_issues]

        train_multi, temp_issues = train_test_split(
            multi_issues, test_size=0.2, random_state=42, stratify=multi_labels)
        train_issues = list(train_multi) + singleton_issues

        # Don't stratify val/test: temp is small and individual repos may have only 1 entry
        val_issues, test_issues = train_test_split(
            temp_issues, test_size=0.5, random_state=42)

        train_set = set(train_issues)
        val_set = set(val_issues)
        test_set = set(test_issues)

        for t in triplets:
            iid = t['issue_id']
            if iid in train_set:
                t['split'] = 'train'
            elif iid in val_set:
                t['split'] = 'val'
            else:
                t['split'] = 'test'

        # Re-save with split field
        with open(args.output_path, "w") as f:
            json.dump(triplets, f, indent=2)

        split_counts = Counter(t['split'] for t in triplets)
        print(f"\nIssue-level stratified split (by repo):")
        print(f"  Train: {len(train_issues)} issues, {split_counts['train']} triplets"
              f"  ({len(singleton_issues)} from singleton repos, forced to train)")
        print(f"  Val:   {len(val_issues)} issues, {split_counts['val']} triplets")
        print(f"  Test:  {len(test_issues)} issues, {split_counts['test']} triplets")


if __name__ == "__main__":
    main()
