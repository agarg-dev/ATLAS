# C2C: From Codebase to Culprit

C2C is a two-stage hierarchical system for automated bug localization. Given a bug report (title, body, and stack trace), it identifies the specific file, function, and line responsible for the bug.

**Stage 1 — CodeRetriever:** A CodeBERT encoder fine-tuned with iterative contrastive learning (in-batch InfoNCE + margin loss, hard negative mining) and indexed in per-repository FAISS indices. It retrieves a ranked shortlist of candidate files.

**Stage 2 — HRL Localization:** A hierarchy of three reinforcement learning agents (file → function → line) trained with per-position REINFORCE and teacher forcing. Each agent progressively narrows the search space.

Evaluated on two datasets:
- **Elasticsearch** — single Java repository (326 bug reports)
- **SWE-bench Verified** — 11 Python repositories (393 bug reports)

---

## Requirements

- Python 3.9 or later
- CUDA-capable GPU (recommended; CPU inference is possible but slow)
- ~10 GB disk space for model checkpoints and indices

Install dependencies:

```bash
pip install -r requirements.txt
```

`faiss-cpu` is installed from PyPI. If you are on an HPC cluster that provides a system FAISS module, you can omit `faiss-cpu` from `requirements.txt` and load the system module instead.

Alternatively, use the provided setup script to create an isolated virtual environment:

```bash
bash scripts/setup_env.sh         # creates ./venv/
bash scripts/setup_env.sh /path/to/venv  # custom location
source venv/bin/activate
```

---

## Repository Layout

```
C2C/
├── requirements.txt
├── scripts/
│   ├── setup_env.sh                    # one-time environment setup
│   ├── run_preprocessing.sh            # Elasticsearch preprocessing pipeline (steps 1–9)
│   ├── run_preprocessing_swebench.sh   # SWE-bench preprocessing pipeline (steps 1–9)
│   ├── train_rl.sh                     # RL training + auto-evaluation
│   └── evaluate_rl.sh                  # standalone RL evaluation
└── src/
    ├── contrastive/                     # Stage 1: CodeRetriever
    │   ├── parse_elasticsearch.py       # Step 1 (Elasticsearch)
    │   ├── parse_swebench.py            # Step 1 (SWE-bench)
    │   ├── build_triplets.py            # Step 2: triplet construction
    │   ├── train_bert.py                # Steps 3, 6: CodeBERT fine-tuning
    │   ├── build_faiss_index.py         # Steps 4, 7: per-repo FAISS index
    │   ├── mine_hard_negatives.py       # Step 5: hard negative mining
    │   ├── build_hierarchical_dataset.py# Step 8: RL training data
    │   └── eval_retrieval.py            # Step 9: retrieval evaluation
    └── rl/                              # Stage 2: HRL Localization
        ├── agents.py                    # agent architectures, reward calculator, trainer
        ├── train.py                     # training entry point
        ├── evaluate.py                  # evaluation (4 protocols)
        ├── dataembedder.py              # CodeBERT embedder, function extraction
        └── chart.py                     # visualization utility
```

---

## Data Preparation

### Elasticsearch (Java)

1. Obtain `processed_samples_train_all.json` from the Beetlebox/Elasticsearch bug dataset and place it at:
   ```
   data/processed_samples_train_all.json
   ```

### SWE-bench Verified (Python)

1. Download the dataset from HuggingFace:
   ```bash
   python3 -c "
   from datasets import load_dataset; import json
   ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
   with open('data/swebench_verified.json', 'w') as f:
       json.dump(list(ds), f, indent=2)
   "
   ```

### CodeBERT tokenizer and model

Download the `microsoft/codebert-base` tokenizer once and save it locally:

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('microsoft/codebert-base')
tok.save_pretrained('models/codebert/codebert_tokenizer')
"
```

The fine-tuned CodeBERT checkpoint (`best_codebert_triplet.pt`) is produced by Step 3 (or Step 6 for the hard-negative round) of the preprocessing pipeline below.

### Repository sources

`build_triplets.py` and `build_faiss_index.py` need to clone or access the bug-fix repositories. They accept a `--repos_cache_dir` argument (default: `data/repos/`) where previously cloned repos are cached to avoid repeated network fetches.

---

## Stage 1: CodeRetriever

### Pipeline overview

The preprocessing pipeline has 9 sequential steps, each resumable from any step:

```
Step 1  parse_*.py                  raw JSON → bug_localization_dataset.json
Step 2  build_triplets.py           bug records → triplet_dataset.json
Step 3  train_bert.py               CodeBERT fine-tuning (random negatives)
Step 4  build_faiss_index.py        per-repo FAISS index (initial model)
Step 5  mine_hard_negatives.py      triplet_dataset_hardneg.json
Step 6  train_bert.py               CodeBERT fine-tuning (hard negatives, warm-start)
Step 7  build_faiss_index.py        per-repo FAISS index (hard-neg model)
Step 8  build_hierarchical_dataset  {train,val,test}_filtered.json for RL training
Step 9  eval_retrieval.py           Hit@K, Recall@K, MRR@K on test split
```

### Running the pipeline

**Elasticsearch (Java):**

```bash
# Run all steps
bash scripts/run_preprocessing.sh

# Resume from a specific step
bash scripts/run_preprocessing.sh 5
```

**SWE-bench Verified (Python):**

```bash
bash scripts/run_preprocessing_swebench.sh
bash scripts/run_preprocessing_swebench.sh 5   # resume from step 5
```

### Running individual steps manually

All scripts in `src/contrastive/` can also be run directly. Set `PYTHONPATH` first:

```bash
export PYTHONPATH="${PWD}/src/contrastive:$PYTHONPATH"
cd src/contrastive
```

**Step 2 — Build triplets:**
```bash
python build_triplets.py \
    --input_path  ../../data/elasticsearch/bug_localization_dataset.json \
    --output_path ../../data/elasticsearch/triplet_dataset.json \
    --repos_cache_dir ../../data/repos \
    --extensions .java
```

**Step 3 — Fine-tune CodeBERT (initial):**
```bash
python train_bert.py \
    --triplet_path ../../data/elasticsearch/triplet_dataset.json \
    --output_dir   ../../outputs/checkpoints/contrastive/elasticsearch \
    --use_path \
    --symmetric_loss
```

**Step 6 — Retrain with hard negatives (warm-start):**
```bash
python train_bert.py \
    --triplet_path ../../data/elasticsearch/triplet_dataset_hardneg.json \
    --output_dir   ../../outputs/checkpoints/contrastive/elasticsearch/hardneg \
    --model_path   ../../outputs/checkpoints/contrastive/elasticsearch/best_codebert_triplet.pt \
    --lr 1e-5 \
    --warmup_steps 0 \
    --patience 5 \
    --epochs 15 \
    --batch_size 16 \
    --grad_accum 4 \
    --use_path \
    --symmetric_loss \
    --selection_metric loss
```

**Step 9 — Evaluate retrieval:**
```bash
python eval_retrieval.py \
    --triplet_path ../../data/elasticsearch/triplet_dataset_hardneg.json \
    --index_dir    ../../data/elasticsearch/faiss_index_codebert_hardneg \
    --model_ckpt   ../../outputs/checkpoints/contrastive/elasticsearch/hardneg/best_codebert_triplet.pt \
    --tokenizer_path ../../models/codebert/codebert_tokenizer \
    --split test
```

### Output layout (Stage 1)

```
data/elasticsearch/
├── bug_localization_dataset.json
├── triplet_dataset.json
├── triplet_dataset_hardneg.json
├── faiss_index_codebert/           initial FAISS index
│   └── elasticsearch/              one subdirectory per repository
│       ├── faiss_index.bin
│       ├── faiss_metadata.json
│       └── file_contents.json
├── faiss_index_codebert_hardneg/   hard-neg FAISS index
└── hierarchical_dataset/
    ├── train_filtered.json
    ├── val_filtered.json
    └── test_filtered.json

outputs/checkpoints/contrastive/elasticsearch/
├── best_codebert_triplet.pt        initial fine-tuned model
└── hardneg/
    └── best_codebert_triplet.pt    hard-negative fine-tuned model (used for RL)
```

---

## Stage 2: HRL Localization

### Training strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| **Teacher-Forced REINFORCE** | `twf` | Each agent trained independently with ground-truth upstream inputs (teacher forcing) and per-position REINFORCE. **Primary strategy.** |
| **Top-Down Freezing REINFORCE** | `t1b1rl` | Train file agent → freeze → train function agent on frozen predictions → freeze → train line agent. Per-position REINFORCE at each phase. |
| Vanilla Joint | `vt` | All three agents trained jointly via REINFORCE with a shared critic. |
| Pretrain + Joint | `pt1` | Cross-entropy pretrain per agent, then joint REINFORCE. |
| Sequential CE | `pt2` | Each agent trained with cross-entropy on ground-truth inputs (no RL). |
| Top-Down Freezing CE | `t1b1` | Same freeze structure as `t1b1rl` but cross-entropy only. |

### Running RL training

```bash
# Primary strategy: TWF with intermediate rewards, Elasticsearch
bash scripts/train_rl.sh twf 42 intermediate elasticsearch

# Primary strategy: TWF with intermediate rewards, SWE-bench
bash scripts/train_rl.sh twf 42 intermediate swebench

# Top-down freezing REINFORCE, SWE-bench
bash scripts/train_rl.sh t1b1rl 42 intermediate swebench

# Sparse reward (ablation)
bash scripts/train_rl.sh twf 42 sparse elasticsearch
```

The script trains the agents and then automatically evaluates both the final and best checkpoints.

**Arguments:** `[strategy] [seed] [reward_type] [dataset]`
- `strategy`: `twf` (default), `t1b1rl`, `vt`, `pt1`, `pt2`, `t1b1`
- `seed`: integer, default `42`
- `reward_type`: `intermediate` (default) or `sparse`
- `dataset`: `elasticsearch` (default) or `swebench`

### Running RL training manually

```bash
export PYTHONPATH="${PWD}/src/rl:$PYTHONPATH"

python src/rl/train.py \
    --training_strategy twf \
    --seed 42 \
    --epochs 40 \
    --reward_type intermediate \
    --data_dir data/elasticsearch/hierarchical_dataset \
    --checkpoint_dir outputs/checkpoints/rl/elasticsearch \
    --embedder_ckpt_path outputs/checkpoints/contrastive/elasticsearch/hardneg/best_codebert_triplet.pt \
    --tokenizer_path models/codebert/codebert_tokenizer \
    --learning_rate 1e-4 \
    --use_path
```

### Evaluation

```bash
# Evaluate best checkpoints for TWF intermediate on Elasticsearch
bash scripts/evaluate_rl.sh "" "" "" twf 42 best intermediate elasticsearch

# Evaluate best checkpoints on SWE-bench
bash scripts/evaluate_rl.sh "" "" "" twf 42 best intermediate swebench

# Evaluate specific checkpoint files
bash scripts/evaluate_rl.sh \
    outputs/checkpoints/rl/elasticsearch/file_agent_twf_intermediate_seed42_best_file.pt \
    outputs/checkpoints/rl/elasticsearch/function_agent_twf_intermediate_seed42_best_oracle_func.pt \
    outputs/checkpoints/rl/elasticsearch/line_agent_twf_intermediate_seed42_best_oracle_line.pt \
    twf 42 best intermediate elasticsearch
```

**Arguments:** `[file.pt] [func.pt] [line.pt] [strategy] [seed] [variant] [reward_type] [dataset]`
- Pass `""` for the first three to use automatically derived checkpoint paths.
- `variant`: `final` or `best`

### Evaluation protocols

`evaluate.py` reports four evaluation protocols:

1. **E2E Cascade** — full pipeline: file → function → line. Reports Hit@{1,5,10} and MRR at each level. This is the main paper metric.
2. **LLMAO-style** — ground-truth file given to the function agent; reports function and line metrics only. Matches the LLMAO baseline comparison protocol.
3. **Issue-based per-level** — each level evaluated independently with issue-aware matching (multiple correct answers per issue counted).
4. **Multi-line recall** — fraction of all buggy lines recovered per issue.

Results are written as JSON to the current working directory (the script changes to `outputs/eval/{dataset}/{strategy}_{reward}_{seed}_{variant}/`).

### Output layout (Stage 2)

```
outputs/checkpoints/rl/elasticsearch/
├── file_agent_twf_intermediate_seed42.pt           final
├── file_agent_twf_intermediate_seed42_best_file.pt best file Hit@1
├── function_agent_twf_intermediate_seed42_best_oracle_func.pt
├── line_agent_twf_intermediate_seed42_best_oracle_line.pt
└── ...

outputs/eval/elasticsearch/twf_intermediate_seed42_best/
└── evaluation_results.json
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| CodeBERT backbone | `microsoft/codebert-base` | 768-d encoder |
| Contrastive temperature τ | 0.1 | InfoNCE softmax temperature |
| Margin | 0.2 | Margin-based triplet loss |
| Sliding window | 512 tokens, stride 510, max 50 chunks | File embedding |
| Round 1 fine-tuning | lr=2e-5, batch 64, max 10 epochs, patience 3 | Random negatives |
| Round 2 fine-tuning | lr=1e-5, warm-start, max 15 epochs, patience 5 | Hard negatives |
| File top-K (RL) | 5 | Candidates per bug report |
| Function top-K (RL) | 15 | Candidates per file |
| Line top-K (RL) | 15 | Candidates per function |
| RL learning rate | 1e-4 | Adam optimizer |
| RL batch size | 32 | |
| RL epochs per phase | 40 | TWF/T1B1-RL |
| Epsilon exploration | 0.3 → 0.05 (decay 0.995/step) | Reset each phase |
| Position weights | [1.0, 0.8, 0.6, 0.4, 0.2] | Per-position REINFORCE rewards |
| Wrong-item penalty | 0.1 | Intermediate reward only |

---

## Weights & Biases

WandB logging is disabled by default in the shell scripts (`WANDB_MODE=offline`).

To enable online logging, remove or override that variable:

```bash
unset WANDB_MODE
bash scripts/train_rl.sh twf 42 intermediate elasticsearch
```

The RL training script also accepts `--use_wandb` and `--project_name` arguments.
The contrastive training script accepts `--wandb_project` (default: `c2c-codebert`).

---

## Reproducing Main Results

To reproduce the primary TWF + intermediate results (Table 3 in the paper), run the full pipeline for each dataset at 3 seeds and average:

```bash
# Elasticsearch
for SEED in 42 123 456; do
    bash scripts/run_preprocessing.sh      # only needed once
    bash scripts/train_rl.sh twf ${SEED} intermediate elasticsearch
done

# SWE-bench
for SEED in 42 123 456; do
    bash scripts/run_preprocessing_swebench.sh   # only needed once
    bash scripts/train_rl.sh twf ${SEED} intermediate swebench
done
```

Evaluation results (JSON) will be written to `outputs/eval/{dataset}/twf_intermediate_seed{seed}_best/`.
