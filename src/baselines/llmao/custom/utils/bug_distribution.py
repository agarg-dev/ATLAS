#!/usr/bin/env python3
"""
bug_distribution.py
===================
Analyse the distribution of single-line vs multi-line bugs and their
*dispersion* (how spread out the buggy lines are within each chunk) across
the LLMAO-format CSV dataset.

Definitions
-----------
Single-line bug   : exactly 1 buggy line in the chunk.
Multi-line bug    : 2+ buggy lines in the chunk.

Dispersion (multi-line only)
  contiguous  : all buggy lines form one unbroken block
                (max_gap between consecutive sorted lines == 1).
  clustered   : all buggy lines fit within a window of ≤ CLUSTER_THRESHOLD
                lines but are not all contiguous (some small internal gaps).
  sparse      : buggy lines are spread across a range wider than
                CLUSTER_THRESHOLD lines.

Metrics reported
  - count and % for single / multi
  - multi-line breakdown: contiguous / clustered / sparse counts and %
  - span  : max_line − min_line + 1  (width of the buggy region)
  - density : n_buggy_lines / span   (1.0 = fully contiguous block)
  - gap statistics : gaps between consecutive buggy lines

Usage
-----
    python custom/utils/bug_distribution.py \\
        --csv-dir data/codegen_instances_csv/swebench_350M

    # Show per-sample detail as well:
    python custom/utils/bug_distribution.py \\
        --csv-dir data/codegen_instances_csv/swebench_350M --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

csv.field_size_limit(2 ** 31 - 1)

# Lines within this span are considered "clustered" rather than "sparse".
CLUSTER_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SampleStats:
    csv_file:    str
    pt_number:   int          # 0-based index within file (always 0 for single-row CSVs)
    n_lines:     int          # total source lines in chunk
    n_buggy:     int          # number of buggy lines
    buggy_lines: list[int]    # sorted 0-based indices

    # Derived (filled by analyse())
    kind:        str = ""     # "single" | "contiguous" | "clustered" | "sparse"
    span:        int = 0      # max_line - min_line + 1  (1 for single)
    density:     float = 0.0  # n_buggy / span
    max_gap:     int = 0      # largest gap between consecutive buggy lines
    mean_gap:    float = 0.0  # average gap (0 for single)
    gaps:        list[int] = field(default_factory=list)

    def analyse(self, cluster_threshold: int = CLUSTER_THRESHOLD) -> None:
        lines = sorted(self.buggy_lines)
        self.n_buggy = len(lines)

        if self.n_buggy == 0:
            self.kind = "no_bugs"
            return

        if self.n_buggy == 1:
            self.kind    = "single"
            self.span    = 1
            self.density = 1.0
            return

        self.span    = lines[-1] - lines[0] + 1
        self.density = self.n_buggy / self.span
        self.gaps    = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
        self.max_gap  = max(self.gaps)
        self.mean_gap = float(np.mean(self.gaps))

        if self.max_gap == 1:
            self.kind = "contiguous"
        elif self.span <= cluster_threshold:
            self.kind = "clustered"
        else:
            self.kind = "sparse"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_csv_dir(csv_dir: pathlib.Path, cluster_threshold: int = CLUSTER_THRESHOLD) -> list[SampleStats]:
    samples: list[SampleStats] = []
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files found in {csv_dir}")

    for csv_path in csv_files:
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            for pt_idx, row in enumerate(csv.reader(fh)):
                if len(row) < 2:
                    continue
                source = row[0]
                try:
                    buggy_lines: list[int] = json.loads(row[1])
                except Exception:
                    buggy_lines = []

                n_lines = len(source.splitlines())
                s = SampleStats(
                    csv_file=csv_path.name,
                    pt_number=pt_idx,
                    n_lines=n_lines,
                    n_buggy=len(buggy_lines),
                    buggy_lines=sorted(buggy_lines),
                )
                s.analyse(cluster_threshold)
                samples.append(s)

    return samples


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:5.1f}%" if total else "  N/A "


def _stats_block(values: list[float], label: str) -> str:
    if not values:
        return f"  {label}: N/A"
    a = np.array(values)
    return (
        f"  {label}:\n"
        f"    mean={a.mean():.2f}  median={np.median(a):.2f}  "
        f"std={a.std():.2f}  min={a.min():.0f}  max={a.max():.0f}"
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(samples: list[SampleStats], verbose: bool) -> None:
    evaluable = [s for s in samples if s.kind != "no_bugs"]
    total = len(evaluable)
    if total == 0:
        print("No samples with buggy lines found.")
        return

    singles   = [s for s in evaluable if s.kind == "single"]
    multis    = [s for s in evaluable if s.kind != "single"]
    contigs   = [s for s in multis if s.kind == "contiguous"]
    clustered = [s for s in multis if s.kind == "clustered"]
    sparse    = [s for s in multis if s.kind == "sparse"]

    print("=" * 60)
    print("Bug Distribution Analysis")
    print("=" * 60)
    print(f"Total chunks with bugs : {total}")
    print(f"  Single-line          : {len(singles):>5}  ({_pct(len(singles), total)})")
    print(f"  Multi-line           : {len(multis):>5}  ({_pct(len(multis), total)})")
    print()

    if multis:
        print(f"Multi-line breakdown  (cluster threshold = {CLUSTER_THRESHOLD} lines)")
        print(f"  Contiguous           : {len(contigs):>5}  ({_pct(len(contigs), len(multis))} of multi)")
        print(f"  Clustered            : {len(clustered):>5}  ({_pct(len(clustered), len(multis))} of multi)")
        print(f"  Sparse               : {len(sparse):>5}  ({_pct(len(sparse), len(multis))} of multi)")
        print()

    # Bug count distribution
    bug_counts = Counter(s.n_buggy for s in evaluable)
    print("Buggy line count distribution:")
    for n in sorted(bug_counts):
        bar = "#" * min(bug_counts[n], 40)
        print(f"  {n:>3} lines : {bug_counts[n]:>5}  {bar}")
    print()

    # Span distribution (multi only)
    if multis:
        spans = [s.span for s in multis]
        print(_stats_block(spans, "Span (max_line - min_line + 1, multi only)"))
        print()

        densities = [s.density for s in multis]
        print(_stats_block(densities, "Density (n_buggy / span, multi only)"))
        print()

        all_gaps = [g for s in multis for g in s.gaps]
        if all_gaps:
            print(_stats_block(all_gaps, "Inter-bug gaps (all consecutive pairs, multi only)"))
            gap_counts = Counter(all_gaps)
            print("  Gap frequency (top 10):")
            for g, c in sorted(gap_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"    gap={g:>3}: {c}")
        print()

    # Span histogram buckets (multi only)
    if multis:
        buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 128)]
        print("Span histogram (multi only):")
        for lo, hi in buckets:
            count = sum(1 for s in multis if lo <= s.span <= hi)
            print(f"  {lo:>3}-{hi:<3} lines : {count:>5}  ({_pct(count, len(multis))} of multi)")
        print()

    # Verbose: per-sample detail for multi-line sparse only
    if verbose:
        print("-" * 60)
        print("Per-sample detail (multi-line, sorted by span desc):")
        print(f"  {'File':<30} {'pt':>3} {'kind':>10} {'n':>3} {'span':>5} {'density':>7} {'max_gap':>8}")
        for s in sorted(multis, key=lambda x: -x.span):
            print(
                f"  {s.csv_file:<30} {s.pt_number:>3} {s.kind:>10} "
                f"{s.n_buggy:>3} {s.span:>5} {s.density:>7.3f} {s.max_gap:>8}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--csv-dir",
        default="data/codegen_instances_csv/swebench_350M",
        help="Directory of LLMAO mirror CSVs (default: data/codegen_instances_csv/swebench_350M).",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample detail for multi-line bugs, sorted by span.",
    )
    ap.add_argument(
        "--cluster-threshold",
        type=int,
        default=CLUSTER_THRESHOLD,
        metavar="N",
        help=f"Max span to call 'clustered' rather than 'sparse' (default: {CLUSTER_THRESHOLD}).",
    )
    args = ap.parse_args()

    csv_dir = pathlib.Path(args.csv_dir)
    if not csv_dir.is_dir():
        sys.exit(f"ERROR: --csv-dir not found: {csv_dir}")

    print(f"Scanning {csv_dir} ...\n")
    samples = parse_csv_dir(csv_dir, cluster_threshold=args.cluster_threshold)
    print(f"Loaded {len(samples)} chunk(s).\n")
    print_report(samples, verbose=args.verbose)


if __name__ == "__main__":
    main()
