"""Parse SWE-bench Verified instances into bug_localization_dataset format.

Reads from a local JSON export or the HuggingFace Hub (princeton-nlp/SWE-bench_Verified).
The fix is encoded as a unified diff in the `patch` field. Output schema is identical
to parse_elasticsearch.py so all downstream steps work unchanged.
"""

import os
import re
import json
import argparse

try:
    import unidiff
    USE_UNIDIFF = True
except ImportError:
    USE_UNIDIFF = False

# ---------------------------------------------------------------------------
# Unified diff parser (regex-based fallback)
# ---------------------------------------------------------------------------

DIFF_FILE_RE = re.compile(r'^diff --git a/.+ b/(.+)$')
HUNK_RE      = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')


def _flush_hunk(current_file, hunk_removed, hunk_added, records):
    """Emit one record per removed line in the current hunk."""
    if not current_file or not hunk_removed:
        return
    fixed_nums    = [ln for ln, _ in hunk_added]
    fixed_content = [c  for _, c  in hunk_added]
    for buggy_ln, buggy_content in hunk_removed:
        records.append({
            "path_to_buggy_file":  os.path.dirname(current_file),
            "buggy_file_name":     os.path.basename(current_file),
            "buggy_line_number":   buggy_ln,
            "buggy_line_content":  buggy_content,
            "fixed_lines_number":  list(fixed_nums),
            "fixed_lines_content": list(fixed_content),
            "fixed_lines_span":    [min(fixed_nums), max(fixed_nums)] if fixed_nums else [0, 0],
        })


def _parse_patch_regex(patch_text):
    """Manual regex-based unified diff parser.

    Key insight: accumulate all removed/added lines per hunk, then flush at
    hunk or file boundaries. This correctly pairs all removed lines with all
    added lines in a hunk regardless of intervening context lines.
    """
    records = []
    current_file = None
    hunk_removed, hunk_added = [], []
    old_line = new_line = 0

    for line in patch_text.splitlines():
        m = DIFF_FILE_RE.match(line)
        if m:
            _flush_hunk(current_file, hunk_removed, hunk_added, records)
            current_file = m.group(1)
            hunk_removed, hunk_added = [], []
            continue

        if line.startswith(('--- ', '+++ ')):
            continue

        m = HUNK_RE.match(line)
        if m:
            _flush_hunk(current_file, hunk_removed, hunk_added, records)
            hunk_removed, hunk_added = [], []
            old_line, new_line = int(m.group(1)), int(m.group(2))
            continue

        if current_file is None:
            continue

        if line.startswith('-') and not line.startswith('---'):
            hunk_removed.append((old_line, line[1:]))
            old_line += 1
        elif line.startswith('+') and not line.startswith('+++'):
            hunk_added.append((new_line, line[1:]))
            new_line += 1
        else:
            # Context line — advance counters only
            old_line += 1
            new_line += 1

    _flush_hunk(current_file, hunk_removed, hunk_added, records)
    return records


def _parse_patch_unidiff(patch_text):
    """Parse unified diff using the `unidiff` library (more robust)."""
    records = []
    try:
        patchset = unidiff.PatchSet(patch_text)
    except Exception:
        # Fall back to regex parser on malformed input
        return _parse_patch_regex(patch_text)

    for patched_file in patchset:
        path = patched_file.path
        for hunk in patched_file:
            removed = [(line.source_line_no, line.value.rstrip('\n'))
                       for line in hunk if line.is_removed and line.source_line_no is not None]
            added   = [(line.target_line_no, line.value.rstrip('\n'))
                       for line in hunk if line.is_added and line.target_line_no is not None]
            if not removed:
                continue
            fixed_nums    = [ln for ln, _ in added]
            fixed_content = [c  for _, c  in added]
            for buggy_ln, buggy_content in removed:
                records.append({
                    "path_to_buggy_file":  os.path.dirname(path),
                    "buggy_file_name":     os.path.basename(path),
                    "buggy_line_number":   buggy_ln,
                    "buggy_line_content":  buggy_content,
                    "fixed_lines_number":  list(fixed_nums),
                    "fixed_lines_content": list(fixed_content),
                    "fixed_lines_span":    [min(fixed_nums), max(fixed_nums)] if fixed_nums else [0, 0],
                })
    return records


def parse_patch(patch_text):
    """Parse a unified diff string into per-removed-line records.

    Uses `unidiff` if available, otherwise falls back to the manual regex parser.
    """
    if USE_UNIDIFF:
        return _parse_patch_unidiff(patch_text)
    return _parse_patch_regex(patch_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench Verified instances to bug_localization_dataset format."
    )
    parser.add_argument(
        "--input_path", default="../../data/swebench_verified.json",
        help="Path to local SWE-bench JSON export, or 'huggingface' to load live from HF Hub.",
    )
    parser.add_argument(
        "--output_path", default="../../data/swebench/bug_localization_dataset.json",
        help="Where to write the output bug_localization_dataset JSON.",
    )
    parser.add_argument(
        "--include_hints", action="store_true",
        help="Append hints_text to the anchor (problem_statement) when non-empty.",
    )
    args = parser.parse_args()

    # Load instances
    if args.input_path == "huggingface":
        from datasets import load_dataset
        instances = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
        print(f"Loaded {len(instances)} instances from HuggingFace Hub.")
    else:
        with open(args.input_path) as f:
            instances = json.load(f)
        print(f"Loaded {len(instances)} instances from {args.input_path}.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    records = []
    skipped_test = 0
    skipped_non_py = 0
    skipped_pure_deletion = 0

    for inst in instances:
        anchor = inst['problem_statement']
        if args.include_hints and inst.get('hints_text', '').strip():
            anchor = anchor + "\n\n" + inst['hints_text']

        for rec in parse_patch(inst.get('patch', '')):
            fname = rec['buggy_file_name']
            fpath = rec['path_to_buggy_file']

            # Skip test files — not the bug's location
            if fname.startswith('test_') or '/test' in fpath or fname.startswith('conftest'):
                skipped_test += 1
                continue

            # Skip non-Python files (docs, configs, etc.)
            if not fname.endswith('.py'):
                skipped_non_py += 1
                continue

            # Skip pure deletions (no fixed lines)
            if not rec['fixed_lines_number']:
                skipped_pure_deletion += 1
                continue

            records.append({
                "issue_id":            inst['instance_id'],
                "repo_name":           inst['repo'],
                "repo_url":            "https://github.com/" + inst['repo'],
                "before_fix_sha":      inst['base_commit'],
                "after_fix_sha":       "",
                "stack_trace":         anchor,
                "buggy_function_name": "",    # filled downstream by build_triplets.py
                "buggy_function_span": [0, 0],
                **rec,
            })

    print(f"\nSWE-bench parse complete:")
    print(f"  Instances processed:     {len(instances)}")
    print(f"  Records produced:        {len(records)}")
    print(f"  Unique instance IDs:     {len({r['issue_id'] for r in records})}")
    print(f"  Skipped (test files):    {skipped_test}")
    print(f"  Skipped (non-.py):       {skipped_non_py}")
    print(f"  Skipped (pure deletion): {skipped_pure_deletion}")
    print(f"  diff parser backend:     {'unidiff' if USE_UNIDIFF else 'regex'}")

    with open(args.output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
