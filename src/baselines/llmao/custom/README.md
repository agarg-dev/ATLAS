# custom/ — Extended Pipelines for LLMAO

This directory contains everything that is **not part of the upstream
[squaresLab/LLMAO](https://github.com/squaresLab/LLMAO)** repository.
It adds SWE-Bench Verified support, alternative CodeGen hidden-state
extractors, a CSV-based evaluation chain, and dataset analysis utilities.

All scripts must be run from the **repository root** (the directory that
contains `codegen_loading.py`, `training.py`, etc.) so that `sys.path`
resolves correctly.

---

## Table of contents

1. [File overview](#file-overview)
2. [SWE-Bench Verified pipeline](#swe-bench-verified-pipeline)
3. [Elasticsearch triplet pipeline](#elasticsearch-triplet-pipeline)
4. [CSV evaluation pipeline (generic datasets)](#csv-evaluation-pipeline-generic-datasets)
5. [Alternative CodeGen loaders](#alternative-codegen-loaders)
6. [Analysis utilities](#analysis-utilities)
7. [Debug utilities](#debug-utilities)
8. [Relation to the rest of the repo](#relation-to-the-rest-of-the-repo)
9. [Dependencies](#dependencies)

---

## File overview

| File | Role |
|------|------|
| [`swebench_pipeline.py`](#swebench_pipelinepy) | End-to-end SWE-Bench Verified: preprocess → run → evaluate |
| [`swebench_singleline_eval.py`](#swe-bench-single-line-evaluation) | SWE-Bench single-line benchmark: split each buggy line into its own instance |
| [`elasticsearch_pipeline.py`](#elasticsearch-triplet-pipeline) | End-to-end Elasticsearch triplet pipeline: preprocess -> run -> CSV evaluate |
| [`codegen_loading_changed.py`](#codegen_loading_changedpy) | Alternative extractor: batched FP16 GPU inference with progress ETA |
| [`process_csvs.py`](#process_csvspy) | CSV pipeline step 1: expand `codegen_instances_csv/` into per-fold line-level CSVs |
| [`filter_csvs.py`](#filter_csvspy) | CSV pipeline step 2: keep only validation-set samples |
| [`combined_csvs.py`](#combined_csvspy) | CSV pipeline step 3: join with model probabilities from `step_*.json` |
| [`dataset_metrics.py`](#dataset_metricspy) | SWE-Bench dataset characterisation (file/function counts, medians) |
| [`utils/bug_distribution.py`](#utilsbug_distributionpy) | Classify bug-line distributions in LLMAO-format CSVs |
| [`observe_pt.py`](#observe_ptpy) | Print keys / shapes of a `.pt` tensor file |
| [`TOP_K_RESULTS.md`](TOP_K_RESULTS.md) | Saved `top_k_per_pt.py` commands and current Elasticsearch / SWE-Bench single-line outputs |
| [`CUSTOM_DATA.md`](CUSTOM_DATA.md) | Full documentation for running LLMAO on any custom dataset |
| [`__init__.py`](__init__.py) | Package marker |

---

## SWE-Bench Verified pipeline

**File:** `custom/swebench_pipeline.py`

End-to-end pipeline that downloads
[princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
converts gold patches into LLMAO-format CSVs, prints (or runs) the
extraction + training commands, and evaluates with paper-style metrics.

This is the supported multi-line SWE-Bench evaluation path. The older
whole-file `<=128` variant is retained only as a historical script and is
not part of the active documentation anymore.

### Step 1 — Preprocess

Downloads the dataset, clones/fetches each repo, parses unified diffs,
chunks files to ≤128 lines (keeping chunks that contain a buggy line),
and writes:

- `data/swebench/*.csv` — one CSV per chunk (two-column LLMAO format)
- `data/swebench/metadata.json` — mapping: instance → files → chunk indices

```bash
python custom/swebench_pipeline.py preprocess \
    --repo-cache ./repos \
    --out-dir    data/swebench \
    --max-instances 500      # omit to process all ~500 verified instances
```

After this step the script prints the exact `run` command to use next.

### Step 2 — Extract hidden states and train

Prints the `codegen_loading.py` and `training.py` commands; add
`--execute` to run them immediately.

```bash
python custom/swebench_pipeline.py run \
    --data-path   data \
    --data-name   swebench \
    --pretrain-type 350M \
    --execute
```

> Without `--execute` the commands are only printed, which is useful for
> inspecting or scheduling them manually.

### Step 3 — Evaluate

Loads the trained checkpoint, runs live inference on each chunk using the
pre-fix source, and reports function-level and line-level Hit@K / MRR.

```bash
python custom/swebench_pipeline.py evaluate \
    --checkpoint    model_checkpoints/swebench_350M \
    --pretrain-type 350M \
    --metadata      data/swebench/metadata.json \
    --repo-cache    ./repos \
    --k-values      1 3 5 10 \
    --output        results/swebench_eval.json   # optional
```

**Metrics reported** (from a 660-sample run):

```
=== Overall Summary ===
  Total PTs          : 660  (with Python functions: 110)

  Line-level (Hit@K = exact gold line appears in the top-K ranked lines):
    Hit@1     55/660  (  8.3%)
    Hit@3     83/660  ( 12.6%)
    Hit@5    115/660  ( 17.4%)
    Hit@10   183/660  ( 27.7%)
    MRR     0.1149

  Function-level (Hit@K = buggy function in top-K by max-pooled score):
    Hit@1     29/660  (  4.4%)
    Hit@3     62/660  (  9.4%)
    Hit@5     74/660  ( 11.2%)
    Hit@10    81/660  ( 12.3%)
    MRR     0.0719
```

---

## SWE-Bench Single-Line Evaluation

**File:** `custom/swebench_singleline_eval.py`

Variant of the SWE-Bench pipeline that splits every buggy line into its own
training and evaluation instance. Long files are still chunked to 128 lines as
in the standard pipeline; the difference is that each chunk carries exactly one
gold line.

```bash
python custom/swebench_singleline_eval.py preprocess \
    --repo-cache ./repos \
    --out-dir    data/swebench_singleline

python custom/swebench_singleline_eval.py run \
    --data-path data \
    --data-name swebench_singleline \
    --pretrain-type 350M \
    --execute
```

For CSV-based `top_k_per_pt.py` evaluation with full-file Python function
context:

```bash
python top_k_per_pt.py \
    --csv-dir data/combined_swebench_singleline \
    --language python \
    --metadata data/swebench_singleline/metadata_singleline.json \
    --original-data-dir data/swebench_singleline \
    --csv-data-dir data/codegen_instances_csv/swebench_singleline_350M \
    --repo-cache repos
```

The current saved output for this command lives in
`custom/SWEBENCH_SINGLELINE_TOPK_OUTPUT.txt`, and a short summary is recorded
in [`TOP_K_RESULTS.md`](TOP_K_RESULTS.md).

---

## Elasticsearch Triplet Pipeline

**File:** `custom/elasticsearch_pipeline.py`

End-to-end pipeline for the bundled Elasticsearch triplet JSON in
`custom/data/triplet_dataset.json`.

### Dataset semantics

- Flattens the JSON `train` / `val` / `test` split labels into one LLMAO dataset.
- Uses the `positive` source and `positive_path` as the sample to score.
- Merges duplicate rows for the same `(issue_id, positive_path)` by unioning all
  buggy lines after converting the 1-based JSON line numbers to LLMAO's 0-based
  line labels.
- Applies the same 128-line bug-window chunking policy as `swebench_pipeline.py`.

### Step 1 - Preprocess

```bash
python custom/elasticsearch_pipeline.py preprocess \
    --input-json custom/data/triplet_dataset.json \
    --out-dir    data/elasticsearch_pipeline
```

Writes:

- `data/elasticsearch_pipeline/*.csv` - one LLMAO CSV per retained 128-line bug window
- `data/elasticsearch_pipeline/metadata.json` - merged issue/file/chunk metadata

### Step 2 - Extract hidden states and train

```bash
python custom/elasticsearch_pipeline.py run \
    --data-path data \
    --data-name elasticsearch_pipeline \
    --pretrain-type 350M \
    --execute
```

This uses the same `codegen_loading.py` and `training.py` flow as the other
dataset pipelines, so LLMAO still performs its standard 10-fold validation over
the full flattened dataset.

### Step 3 - Evaluate

```bash
python custom/elasticsearch_pipeline.py evaluate --pretrain-type 350M
```

This wraps the generic CSV evaluation chain:

`process_csvs.py -> filter_csvs.py -> combined_csvs.py -> top_k_per_pt.py`

and automatically passes `--language java` to `top_k_per_pt.py` so the final
report includes Java function-level metrics as well as line-level Hit@K / MRR.

> Java function extraction requires `javalang`. Install dependencies from
> `requirements.txt` before running Java-aware evaluation.

---

## CSV evaluation pipeline (generic datasets)

Use this four-step chain when you have trained LLMAO on **any** custom
dataset and want line-level Hit@K / MRR without modifying `top_scores.py`.
It is also the recommended evaluation path for the beetlebox dataset.

```
process_csvs.py → filter_csvs.py → combined_csvs.py → top_k_per_pt.py (repo root)
```

### Step 1 — Build per-fold master CSVs

**File:** `custom/process_csvs.py`

Reads `model_logs/<dataset>/*/validation_set_files.json` to discover which
`.pt` indices belong to each validation fold, then maps each index to its
mirror CSV in `codegen_instances_csv/` and expands it to one row per
physical line.

```bash
python custom/process_csvs.py \
    --validation-logs-dir model_logs/<dataset_name> \
    --csv-data-dir        data/codegen_instances_csv/<dataset_name>_<pretrain_type> \
    --output-dir          data/processed_csv
```

**Output columns:** `pt_number`, `line_number`, `line`, `is_buggy`

### Step 2 — Filter to validation samples only

**File:** `custom/filter_csvs.py`

Keeps only rows whose `pt_number` appears in the validation set for the
corresponding fold. If no fold directory is found, copies the merged CSV
as-is.

```bash
python custom/filter_csvs.py \
    --merged-csv-dir      data/processed_csv \
    --validation-logs-dir model_logs/<dataset_name> \
    --output-dir          data/filtered_csv
```

### Step 3 — Join with model probabilities

**File:** `custom/combined_csvs.py`

Merges each filtered CSV with the `prob` array from the matching
`step_<iter>.json` training log, adding a `probability` column. The fold
is matched by the stem of the CSV filename to the subdirectory in
`json_logs/`.

```bash
python custom/combined_csvs.py \
    --csv-dir       data/filtered_csv \
    --json-logs-dir model_logs/<dataset_name> \
    --output-dir    data/combined_csvs
```

> **Known issue:** `training.py` previously created its validation
> DataLoader with `shuffle=True`, which can misalign the flat `prob`
> array in `step_*.json` with the CSV rows. This has been fixed in-repo
> (validation loader now uses `shuffle=False`). If you used an older
> checkout, re-run training before evaluating.

### Step 4 — Compute Top-K and MRR

**File:** `top_k_per_pt.py` (repo root, not inside `custom/`)

```bash
python top_k_per_pt.py --csv-dir data/combined_csvs --k-values 1 3 5 10
python top_k_per_pt.py --csv-dir data/combined_csvs/elasticsearch_pipeline --language java
```

Reports line-level and language-aware function-level Hit@K and MRR across all
validation folds.

---

## Alternative CodeGen loaders

The upstream `codegen_loading.py` is the default extraction script.
`codegen_loading_changed.py` is an alternative for when you want faster
throughput on a single GPU.

### `codegen_loading_changed.py`

Improvements over the upstream loader:

- **Batched GPU inference** (`BATCH = 6`, tunable) — processes six samples
  at once via `get_hidden_states_batch`, significantly faster than the
  one-sample-at-a-time upstream path.
- **FP16** — casts the model to half-precision on GPU (`.half()`) to halve
  VRAM use and accelerate matrix ops.
- **Progress ETA** — prints `[rows_done/total] Elapsed / ETA` after each
  batch so you can estimate remaining time.
- **Hard GPU lock** — sets `CUDA_VISIBLE_DEVICES=0` at startup; to use a
  different GPU, change that line or unset it and use the env var in the
  shell instead.

```bash
python custom/codegen_loading_changed.py <data_path> <data_name> <biggest_model>
# biggest_model: 1 = 16B, 0 = 350M (loop currently defaults to 350M)
```

Output layout matches `codegen_loading.py`:

```
data/codegen_states/<dataset_name>_<pretrain_type>/*.pt
```

---

## Analysis utilities

### `dataset_metrics.py`

Characterises SWE-Bench Verified at the file and function level before
running the full pipeline. Useful for deciding chunking strategy or
understanding dataset balance.

**Metrics reported:** unique bug reports, total Python files
(buggy / non-buggy), buggy function count, total function count, and
median functions/file, lines/file, lines/function.

```bash
# Full dataset
python custom/dataset_metrics.py --repo-cache ./repos

# Quick test on 50 instances
python custom/dataset_metrics.py --repo-cache ./repos --max-instances 50

# Also save raw per-instance data
python custom/dataset_metrics.py --repo-cache ./repos --output metrics.json
```

> **Import note:** `dataset_metrics.py` imports helpers from
> `swebench_pipeline`. Run it from the repo root so that `custom/` is on
> `sys.path`, or set `PYTHONPATH=custom` before running.

### `utils/bug_distribution.py`

Reads a directory of LLMAO-format CSVs and classifies each sample by its
bug-line distribution:

- **Single-line** — exactly one buggy line
- **Contiguous** — all buggy lines in one unbroken run
- **Clustered** — multiple groups closer than `--cluster-threshold` lines
  apart (default 10)
- **Sparse** — multiple groups farther apart than the threshold

```bash
python custom/utils/bug_distribution.py \
    --csv-dir          data/swebench \
    --cluster-threshold 10 \
    --verbose
```

Output is printed to stdout; redirect to a file if needed.

---

## Debug utilities

### `observe_pt.py`

Loads a single `.pt` tensor file and prints its dict keys, tensor shapes,
and basic statistics. Useful for verifying that `codegen_loading.py` (or
an alternative) wrote the expected data.

```bash
python custom/observe_pt.py data/codegen_states/swebench_350M/0.pt
```

---

## Relation to the rest of the repo

```
Upstream LLMAO (squaresLab/LLMAO)          custom/ additions
──────────────────────────────────         ──────────────────────────────────
codegen_loading.py  ──extract──►  codegen_loading_changed.py (batched alternative)
training.py
top_scores.py                     ◄── CSV pipeline (process → filter → combine)
top_k_per_pt.py                   ◄── combined CSVs from custom/ pipeline
demo.py                           ◄── swebench_pipeline.py evaluate (same checkpoint format)
```

The checkpoint format is identical across all paths: a plain PyTorch
`state_dict` saved by `training.py` and read with
`get_model_config(pretrain_type)` from `transformer.py`.

For full documentation on running LLMAO on any custom dataset — including
the data contract, directory layout, GPU setup, and known issues — see
[`CUSTOM_DATA.md`](CUSTOM_DATA.md).

---

## Dependencies

In addition to the base LLMAO requirements (see the top-level
[`README.md`](../README.md)):

```bash
# Required for SWE-Bench pipeline and dataset_metrics.py
pip install datasets gitpython
```

`datasets` and `gitpython` are the only additions needed for the SWE-Bench
pipeline. Everything else (`torch`, `transformers`, `accelerate`) is
already required by the upstream repo.
