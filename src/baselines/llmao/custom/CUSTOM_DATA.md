# Running LLMAO on a Custom Dataset

This document explains the LLMAO data contract, directory layout, preprocessing options, commands
to run, and evaluation caveats — everything you need to reproduce results on a new dataset.

---

## Quickstart checklist

1. Produce per-sample CSVs in the two-column LLMAO format (see [Data contract](#data-contract))
2. Place them under `data/<dataset_name>/`
3. Run CodeGen hidden-state extraction (`codegen_loading.py`)
4. Run training (`training.py`)
5. Run evaluation via the CSV pipeline (`process_csvs` → `combined_csvs` → `top_k_per_pt`)

---

## Data contract

Every sample must be represented as a **headerless, two-column CSV row**:

| Column | Content |
|--------|---------|
| 0 | Full source code of the file/snippet as a plain string |
| 1 | JSON array of **0-based** physical line indices that are buggy, e.g. `[4, 17, 18]` |

Constraints:

- **Line indexing is 0-based** and corresponds to `source.splitlines()`. This is enforced by
  `CSVDataLoader.row_processer` in `codegen_loading.py` and by `process_csvs.py`.
- **Maximum 128 lines** per sample for the pre-extraction pipeline (`codegen_loading.py`,
  `MAX_LEN = 128`). Samples with more lines should be chunked. `NaiveDataset` in
  `transformer.py` has a separate 2 048-token limit but is less commonly used.
- Lines must contain **newline characters** that the CodeGen tokeniser recognises (token ids 198 /
  628). Stripping all trailing newlines will cause extraction to fail silently.

Example valid row (abbreviated):

```
"def foo(x):\n    if x > 0:\n        return x\n    return -x\n","[1]"
```

---

## Directory layout

All paths are **relative to the working directory from which you run the scripts** (they use
`os.getcwd()`). Keep the repo root as your working directory.

```
data/
  <dataset_name>/
    *.csv                              # one CSV per sample (codegen_loading.py reads all)
    code_bugline.csv                   # single aggregate CSV (NaiveDataset path only)
  codegen_states/
    <dataset_name>_<pretrain_type>/
      0.pt  1.pt  2.pt  …             # pre-extracted hidden-state tensors (one per sample)
  codegen_instances_csv/
    <dataset_name>_<pretrain_type>/
      0.csv  1.csv  2.csv  …          # mirror CSVs written alongside the tensors

model_logs/
  <dataset_name>/
    <run_subdir>/                      # one subdir per 10-fold split
      validation_set_files.json        # list of .pt paths in the validation fold
      step_<iter>.json                 # best-validation prob+label snapshot

model_checkpoints/
  <dataset_name>_<pretrain_type>       # PyTorch state_dict saved during training
```

The `<run_subdir>` names are generated automatically by `training.py` using the pattern
`{dataset_name}_{pretrain_type}_{target_dim}_{fold_index}`.

---

## Preprocessing options

### Option A — From Git history + a JSON issues file

Use [`data_processing/preprocess_llmao.py`](data_processing/preprocess_llmao.py) if your dataset
is described as a collection of bug-fixing commits:

```bash
python data_processing/preprocess_llmao.py issues.json \
    --repo-cache ./repos \
    --out-dir    ./llmao_data \
    --max-rows   1000
```

**Input JSON schema** (one object per bug):

```json
[
  {
    "repo_url":           "https://github.com/owner/repo",
    "before_fix_sha":     "abc123",
    "after_fix_sha":      "def456",
    "path_to_buggy_file": "src/",
    "buggy_file_name":    "foo.py",
    "buggy_line_number":  [5, 6],      // optional — 1-based; converted to 0-based
    "fixed_lines_number": [5, 6]       // optional — 1-based; converted to 0-based
  }
]
```

**Output:** `llmao_data/code_bugline.csv` — two-column, headerless, labels are **0-based**.

> **Warning:** `preprocess_llmao_small.py` (also in `data_processing/`) uses a different
> line-index convention (1-based for some fields). Do not mix output from both scripts without
> converting indices.

### Option B — Manual / external ETL

Write CSVs directly in the two-column format. You can split the data into as many individual files
as you like and place them all in `data/<dataset_name>/`. `codegen_loading.py` will process every
`.csv` it finds in that directory.

Utility scripts for inspecting existing CSVs:

- [`data_processing/check_datasets.py`](data_processing/check_datasets.py) — reports token and
  line-count statistics across a data folder.
- [`data_processing/filter_long_rows.py`](data_processing/filter_long_rows.py) — removes samples
  that exceed a line-count threshold (paths are hardcoded; edit before use).
- [`data_processing/trim_dataset.py`](data_processing/trim_dataset.py) — trims
  `./llmao_data/code_bugline.csv` to a smaller output file.

---

## Step-by-step pipeline

### 1. Extract CodeGen hidden states

`codegen_loading.py` reads every `.csv` in `data/<dataset_name>/`, runs each source file through
the CodeGen language model, and saves the per-line hidden-state tensors as `.pt` files.

```bash
python codegen_loading.py <data_path> <dataset_name> <biggest_model>
```

| Argument | Values | Meaning |
|----------|--------|---------|
| `data_path` | e.g. `data` | Parent directory containing `<dataset_name>/` |
| `dataset_name` | e.g. `swebench` | Subfolder name |
| `biggest_model` | `1` | Use 16B checkpoint (requires ~38 GB VRAM across 2-3 GPUs) |
| | `0` | Use 350M checkpoint (requires ~2.6 GB VRAM) |

Shell wrapper for quick edits: [`codegen_loading.sh`](codegen_loading.sh) — set `data_name` and
`biggest_model`, then `bash codegen_loading.sh`.

**Outputs:**
- `data/codegen_states/<dataset_name>_<pretrain_type>/0.pt`, `1.pt`, …
- `data/codegen_instances_csv/<dataset_name>_<pretrain_type>/0.csv`, `1.csv`, …

### 2. Train the LLMAO model

```bash
python training.py <data_path> <dataset_name> <pretrain_type> <pretraining>
```

| Argument | Values | Meaning |
|----------|--------|---------|
| `data_path` | e.g. `data` | Same as above |
| `dataset_name` | e.g. `swebench` | Must match what was used in Step 1 |
| `pretrain_type` | `350M`, `2B`, `6B`, `16B` | CodeGen checkpoint size |
| `pretraining` | `1` | Load pre-extracted `.pt` tensors (`PreloadedDataset`) |
| | `0` | Load raw `code_bugline.csv` on-the-fly (`NaiveDataset`, slower) |

Shell wrapper: [`fault_localizer.sh`](fault_localizer.sh).

**Outputs:**
- `model_checkpoints/<dataset_name>_<pretrain_type>` — best checkpoint (state_dict)
- `model_logs/<dataset_name>/<run_subdir>/step_<iter>.json` — per-fold validation snapshots
- `model_logs/<dataset_name>/<run_subdir>/validation_set_files.json` — which `.pt` files were
  held out in this fold

#### Making GPU training work

[`training.py`](training.py) always puts the Voltron model and every batch on **`cuda:0`**
(`model.to("cuda:0")`; [`PreloadedDataset`](transformer.py) / [`NaiveDataset`](transformer.py) use
`.to("cuda:0")` in `__getitem__`). There is **no** `DataParallel` / second-GPU training path in
this script—only **one** CUDA device is used for training.

**Checklist:**

1. **PyTorch can execute kernels on your GPU** — run the verification snippet in
   [Ensuring CUDA works for your GPU](#ensuring-cuda-works-for-your-gpu-blackwell--sm_120-two-gpus)
   (`matmul ok` on `cuda:0`). If you see **“no kernel image…”** or **sm_120 not compatible**,
   install a **newer PyTorch + CUDA** build from [pytorch.org](https://pytorch.org/get-started/locally/)
   until that test passes. Training will not work on GPU until this succeeds.

2. **Pick which physical card is `cuda:0`** — if you have two GPUs and want training on the
   *second* one, set before running Python, e.g. PowerShell:
   `$env:CUDA_VISIBLE_DEVICES="1"`  
   Then `cuda:0` inside the process is your second physical GPU. (Do not set `LLMAO_DEVICE=cpu`.)

3. **VRAM** — default `batch_size` in `training.py` depends on `data_name` (often `8`). If you
   hit OOM, lower `batch_size` in `model_pipe` / `driver` after reading the surrounding logic.

4. **Pre-extracted `.pt` files** — if Step 1 ran on **CPU**, tensors are stored on CPU; during
   training, `PreloadedDataset` moves each batch to **`cuda:0`**. That is fine as long as step 1
   completed; you do not need to re-run extraction on GPU for training to use the GPU.

5. **Run training** — same command as above, e.g.
   `python training.py data swebench 350M 1`.

With two GPUs, the **second** is unused by `training.py` unless you run another process on it
(e.g. leave CodeGen extraction on GPU 0 and training on GPU 1 via `CUDA_VISIBLE_DEVICES` in
separate terminals).

### 3. Demo / single-file inference

Run LLMAO on a single source file using a trained checkpoint:

```bash
python demo.py <demo_type> <pretrain_type> <code_file_path>
```

`demo_type` controls which checkpoint to load (`model_checkpoints/<demo_type>_<pretrain_type>`).
Use your `dataset_name` as `demo_type` after training.

> **Note:** `demo.py` filters out lines starting with `/`, `*`, `#`, and bare `{`/`}` before
> scoring. This filter was designed for C/Java files. For Python files, only the `#` filter
> applies; the `/` and `*` checks are harmless but irrelevant.

---

## Evaluation

### Option A — Line-level CSV pipeline (recommended for custom data)

This pipeline is more flexible than `top_scores.py` and does not depend on dataset-specific
hardcoded constants.

```
process_csvs.py  →  filter_csvs.py  →  combined_csvs.py  →  top_k_per_pt.py
```

**Step 3a — Build per-fold master CSVs**

```bash
python process_csvs.py \
    --validation-logs-dir model_logs/<dataset_name> \
    --csv-data-dir        data/codegen_instances_csv/<dataset_name>_<pretrain_type> \
    --output-dir          data/processed_csv
```

Produces one master CSV per fold with columns: `pt_number`, `line_number`, `line`, `is_buggy`.

**Step 3b — Filter to validation samples only**

```bash
python filter_csvs.py \
    --merged-csv-dir      data/processed_csv \
    --validation-logs-dir model_logs/<dataset_name> \
    --output-dir          data/filtered_csv
```

**Step 3c — Add model probabilities**

```bash
python combined_csvs.py \
    --csv-dir        data/filtered_csv \
    --json-logs-dir  model_logs/<dataset_name> \
    --output-dir     data/combined_csvs
```

Adds a `probability` column by joining against `step_<iter>.json`.

> **Shuffle caveat:** `training.py` creates the validation DataLoader with `shuffle=True`
> (line 219). This means the flat `prob`/`label` arrays in `step_*.json` are **not** in a
> stable sample order. `combined_csvs.py` assumes they are ordered identically to the CSV rows,
> which may cause silent misalignment. See [Known issues](#known-issues) below.

**Step 3d — Compute Top-K and MRR**

```bash
python top_k_per_pt.py --csv-dir data/combined_csvs
```

Reports Top-0 through Top-10 accuracy and MRR across all validation folds.

### Option B — Paper-style top-K from step_*.json

```bash
python top_scores.py <log_path> <pretrain_type>
# Example:
python top_scores.py model_logs 350M
```

> **Warning:** `top_scores.py` hardcodes `data_name = "beetlebox"` in its `__main__` block
> (line 275) and contains dataset-specific `data_split`, `window_split`, and `total_bugs`
> constants in `calculate_top_k_hits` and `results`. Running this on a new dataset will raise
> `ValueError: Unknown dataset in label_name`. You must add a branch for your dataset name or
> pass `data_name` from the CLI (requires a small code change).

---

## Known issues

| Issue | Location | Recommended fix |
|-------|----------|-----------------|
| Validation DataLoader used to shuffle samples, misaligning `step_*.json` with CSV rows | `training.py` | Fixed in-repo: validation loader uses `shuffle=False` |
| `top_scores.py` hardcodes `data_name = "beetlebox"` | `top_scores.py` line 275 | Accept `data_name` as a CLI argument and add a branch for your dataset |
| `demo.py` comment filter includes `/` and `*` prefixes (C/Java convention) | `demo.py` line 38 | Only the `#` check matters for Python |

---

## Ensuring CUDA works for your GPU (Blackwell / sm_120, two GPUs)

LLMAO needs a PyTorch build whose **precompiled CUDA kernels** include your GPU’s **compute
capability**. If PyTorch only lists up to `sm_90` but your card is **sm_120** (e.g. GeForce RTX 5090,
5070 Ti), any real CUDA op can fail with **“no kernel image is available for execution on the
device”** even though `torch.cuda.is_available()` is true.

Do **not** set `LLMAO_DEVICE=cpu` if you want the GPU path; that forces CPU on purpose.

### 1. Driver and runtime

- Install the **latest NVIDIA driver** for your OS from [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx).
- Confirm the driver sees both cards:

  ```powershell
  nvidia-smi
  ```

  You should see each GPU, memory, and a recent driver version.

### 2. Install a PyTorch build that supports your architecture

Use the official selector and pick the **newest PyTorch** and **highest CUDA** variant offered for
your OS (for Blackwell, that often means **CUDA 12.8+** or a **nightly** build, not an old `cu121`
wheel):

- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Typical pattern (adjust index to what the site shows for your platform):

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If stable builds still lag on **Windows** for sm_120, check PyTorch’s forums and GitHub issues for
**Windows + sm_120**; some users use **Linux** or **WSL2** with a newer wheel while native Windows
binaries catch up.

**Also required for current `transformers`:** PyTorch **≥ 2.6** when loading CodeGen’s
`pytorch_model.bin` (see CVE-2025-32434 in the troubleshooting section below).

### 3. Verify CUDA in the same environment you use for LLMAO

Activate your conda/venv, then run:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cap', torch.cuda.get_device_capability(0)); print('name', torch.cuda.get_device_name(0)); x=torch.zeros(256,256,device='cuda',dtype=torch.float16); torch.mm(x,x); torch.cuda.synchronize(); print('matmul ok')"
```

- **`matmul ok`** means a kernel actually ran on GPU 0.
- If this raises **`RuntimeError: no kernel image...`**, your installed PyTorch **does not**
  include kernels for that GPU; go back to step 2 (different CUDA package or nightly).

Optional: see which SMs the wheel claims to support:

```powershell
python -c "import torch; print(torch.cuda.get_arch_list())"
```

You want your GPU’s architecture to be covered (for sm_120, the list must include something
compatible—exact string varies by PyTorch version).

### 4. Using two GPUs

**Physical numbering vs `cuda:0`:** PyTorch only sees GPUs in the order exposed by CUDA. You can
**remap** devices without editing code:

| Goal | Typical approach |
|------|------------------|
| Use **only** the second physical GPU as the one visible process | `set CUDA_VISIBLE_DEVICES=1` then run Python (that card becomes `cuda:0`). |
| Expose **both** GPUs to PyTorch | `set CUDA_VISIBLE_DEVICES=0,1` (PowerShell: `$env:CUDA_VISIBLE_DEVICES="0,1"`). |

**This repository:**

- [`codegen_loading.py`](codegen_loading.py) sets `CUDA_VISIBLE_DEVICES="0"` at the top of the file,
  which **forces only one GPU** (the first *visible* to the driver before any parent env—actually it
  overwrites to physical GPU 0). To use your **second** GPU for extraction, either:
  - change that line to `"1"`, or
  - remove the line and set `CUDA_VISIBLE_DEVICES` in the shell before launching Python.
- [`codegen.py`](codegen.py) uses `device_map="balanced"` when CUDA works, so **Hugging Face can
  shard the CodeGen model across multiple GPUs** if more than one device is visible and the model
  is large enough to benefit.
- [`training.py`](training.py) and [`transformer.py`](transformer.py) use **`cuda:0`** in many places
  for the Voltron head and tensor batches; the **small** LLMAO head usually sits on one GPU while
  CodeGen may use `balanced` across two during **`codegen_loading.py`**.

**Practical split:** Many people run **`codegen_loading.py`** with both GPUs visible (remove or
adjust `CUDA_VISIBLE_DEVICES` in `codegen_loading.py`) so CodeGen can use `balanced`, then run
**training** with a single full-size GPU if the second card is weaker or you want deterministic
device 0 behavior.

### 5. If verification still fails

- Confirm you are not mixing a **CPU-only** `torch` with a CUDA driver.
- Re-run the **matmul** test on **`cuda:1`** if you use two GPUs:

  ```powershell
  python -c "import torch; d=torch.device('cuda:1'); x=torch.zeros(256,256,device=d,dtype=torch.float16); torch.mm(x,x); torch.cuda.synchronize(); print('cuda:1 ok')"
  ```

- Search PyTorch issues for your exact card (e.g. “RTX 5070 Ti sm_120 Windows”).

**CPU fallback (optional):** [`codegen.py`](codegen.py) falls back to CPU only when the CUDA probe
fails. That is a stopgap, not a substitute for a correct PyTorch+GPU pairing.

---

## Troubleshooting: PyTorch and Transformers

Recent `transformers` releases require **PyTorch 2.4 or newer** for the library to enable the
PyTorch backend at all. Separately, loading **`.bin`** checkpoints (Salesforce CodeGen on the Hub
only ships `pytorch_model.bin`, not `model.safetensors`) triggers a **PyTorch 2.6+** requirement
in current `transformers` (CVE-2025-32434). If you see:

`ValueError: ... upgrade torch to at least v2.6 ...`

install PyTorch **2.6 or newer** (same CUDA channel you already use), for example:

```powershell
python -m pip install --upgrade "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Use the [official install matrix](https://pytorch.org/get-started/locally/) (`cu121`, `cu128`,
`cpu`, etc.) if `cu124` does not match your driver.

For **sm_120 / RTX 50-series** and **multi-GPU** setup, follow [Ensuring CUDA works for your GPU](#ensuring-cuda-works-for-your-gpu-blackwell--sm_120-two-gpus) above.

**Windows — plain PyPI (CPU or default wheel):**

```powershell
python -m pip install --upgrade "torch>=2.6"
```

Do **not** use `torch==2.4.0+cpu` with `-f https://download.pytorch.org/whl/torch_stable.html` on
Windows; that index often has no matching `+cpu` build. If `pip` still fails, use PyTorch’s CPU
wheel index:

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Windows — CUDA (pick the CUDA version that matches your driver; example CUDA 12.1):**

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then reinstall or upgrade `transformers` if needed:

```powershell
python -m pip install --upgrade transformers accelerate
```

`codegen_loading.py` and `training.py` expect a **GPU** for reasonable runtime; CPU-only installs
may work for tiny tests but will be very slow and CodeGen 350M still needs enough RAM.

### `torchdata` (optional)

`codegen_loading.py` no longer uses `torchdata.datapipes` — it streams CSV rows with
`torch.utils.data.IterableDataset`, so **`pip install torchdata` is not required** for extraction.

If you still have an old `torchdata` install that pins `torch==2.0.0`, you can remove it:

```powershell
python -m pip uninstall torchdata -y
```

If you see `ModuleNotFoundError: No module named 'torchdata.datapipes'` after upgrading to
**torchdata 0.11+**, that API was removed upstream; use a current checkout of this repo (which
does not depend on it) or downgrade to `torchdata==0.6.0` **only** together with `torch==2.0.0`
(not recommended if you need a newer PyTorch for `transformers`).

### `ModuleNotFoundError: transformers.utils.model_parallel_utils`

Newer `transformers` releases removed that module. This repo vendors the small helpers in
[`model_parallel_utils.py`](model_parallel_utils.py); [`modeling_codegen.py`](modeling_codegen.py)
imports them automatically when the upstream import fails. Run training from the **repository
root** so `model_parallel_utils.py` is on `sys.path`.

### `pynvml` deprecation warning

PyTorch may print a `FutureWarning` about `pynvml`; it is harmless. To silence it you can install
`nvidia-ml-py` (`pip install nvidia-ml-py`) as the message suggests.

---

## Extending `top_scores.py` for a new dataset

Add a branch to `calculate_top_k_hits` and `results` in `top_scores.py`:

```python
elif "your_dataset_name" in label_name:
    data_split, window_split = <num_projects>, <bugs_per_project>
```

And set `total_bugs = <total_number_of_bugs>` in the `results` function. Then change the
`__main__` block to accept `data_name` from the CLI rather than hardcoding it.

---

## SWE-Bench Verified

See [`swebench_pipeline.py`](swebench_pipeline.py). It downloads SWE-Bench Verified, parses gold
patches, checks out `base_commit`, chunks to 128 lines, and writes **one combined** set of LLMAO
CSVs plus a single `metadata.json` under `--out-dir` (default `data/swebench`). You then train
**one** LLMAO model on all CSVs and evaluate on the full benchmark.

```bash
# 1) Preprocess (all instances → data/swebench/*.csv + metadata.json)
python swebench_pipeline.py preprocess --repo-cache ./repos --out-dir data/swebench

# 2) CodeGen tensor extraction + Voltron training (defaults: data_path=data, data_name=swebench)
python swebench_pipeline.py run --pretrain-type 350M --execute

# 3) Paper-protocol evaluation on the trained checkpoint
python swebench_pipeline.py evaluate \
    --checkpoint  model_checkpoints/swebench_350M \
    --pretrain-type 350M \
    --metadata    data/swebench/metadata.json \
    --repo-cache  ./repos
```

If you use a non-default `--out-dir`, pass matching `--data-path` / `--data-name` to `run` (parent
folder and last path segment). After preprocess, the script prints the exact `run` line.

- **Checkpoint:** `model_checkpoints/swebench_350M` when `data_name` is `swebench`.
- **Tensors:** `data/codegen_states/swebench_350M/` (with default layout).
- **`evaluate` metrics:** `--k-values` (default `1 3 5 10`) sets both **function Hit@K** (any
  ground-truth buggy function appears in the top-K functions ranked by max pooled line score) and
  **line Hit@K** (the single highest-probability line is within K physical lines of any buggy
  line). **MRR** (mean reciprocal rank) is reported separately for function- and line-level ranking.

  
  **Results** (from a 660-sample evaluation):
  
  ```
  === Overall Summary ===
    Total PTs          : 660  (with Python functions: 110)

    Line-level (Hit@K = argmax line within ±K of any buggy line):
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