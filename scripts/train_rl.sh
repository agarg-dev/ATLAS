#!/bin/bash
# Train hierarchical RL agents for C2C bug localization.
#
# Training strategies:
#   twf     — Teacher-Forced REINFORCE (primary; decoupled, per-position REINFORCE)
#   t1b1rl  — Top-Down Freezing REINFORCE (per-position REINFORCE, frozen upstream)
#   vt      — Vanilla joint REINFORCE
#   pt1     — Pretrain then joint REINFORCE
#   pt2     — Sequential teacher-forced cross-entropy (CE only)
#   t1b1    — Top-down freezing cross-entropy (CE only)
#
# Usage: bash scripts/train_rl.sh [strategy] [seed] [reward_type] [dataset]
#   strategy:    twf (default)
#   seed:        42 (default)
#   reward_type: intermediate (default) | sparse
#   dataset:     elasticsearch (default) | swebench

strategy=${1:-twf}
seed=${2:-42}
reward_type=${3:-intermediate}
dataset=${4:-elasticsearch}

reward_tag="_${reward_type}"

if [ "${strategy}" = "vt" ] || [ "${strategy}" = "pt1" ]; then
    epochs=30
elif [ "${strategy}" = "pt2" ] || [ "${strategy}" = "t1b1" ] || \
     [ "${strategy}" = "twf" ] || [ "${strategy}" = "t1b1rl" ]; then
    epochs=40
else
    epochs=20
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/${dataset}/hierarchical_dataset"
EMBEDDER_CKPT="${PROJECT_DIR}/outputs/checkpoints/contrastive/${dataset}/hardneg/best_codebert_triplet.pt"
TOKENIZER_PATH="${PROJECT_DIR}/models/codebert/codebert_tokenizer"
CKPT_DIR="${PROJECT_DIR}/outputs/checkpoints/rl/${dataset}"

mkdir -p "${PROJECT_DIR}/outputs/logs/rl"
mkdir -p "${CKPT_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/wandb"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/outputs/wandb"
export PYTHONPATH="${PROJECT_DIR}/src/rl:$PYTHONPATH"
export HF_HOME="${PROJECT_DIR}/models/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "=========================================="
echo "Dataset:  ${dataset}"
echo "Strategy: ${strategy}"
echo "Seed:     ${seed}"
echo "Reward:   ${reward_type}"
echo "Epochs:   ${epochs}"
echo "Started:  $(date)"
echo "=========================================="

cd "${PROJECT_DIR}"
python src/rl/train.py \
    --training_strategy ${strategy} \
    --seed ${seed} \
    --epochs ${epochs} \
    --data_dir ${DATA_DIR} \
    --checkpoint_dir ${CKPT_DIR} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --embedder_ckpt_path ${EMBEDDER_CKPT} \
    --reward_type ${reward_type} \
    --learning_rate 1e-4 \
    --use_path

echo "=========================================="
echo "Training finished: $(date)"
echo "Running evaluation on test set..."
echo "=========================================="

# --- Evaluate FINAL checkpoint ---
EVAL_DIR="${PROJECT_DIR}/outputs/eval/${dataset}/${strategy}${reward_tag}_seed${seed}_final"
mkdir -p "${EVAL_DIR}"
cd "${EVAL_DIR}"
echo "--- Evaluating FINAL checkpoint ---"
python "${PROJECT_DIR}/src/rl/evaluate.py" \
    --file_agent_path ${CKPT_DIR}/file_agent_${strategy}${reward_tag}_seed${seed}.pt \
    --func_agent_path ${CKPT_DIR}/function_agent_${strategy}${reward_tag}_seed${seed}.pt \
    --line_agent_path ${CKPT_DIR}/line_agent_${strategy}${reward_tag}_seed${seed}.pt \
    --test_data_path ${DATA_DIR}/test_filtered.json \
    --device cuda \
    --evaluation_type both \
    --embedder_ckpt_path ${EMBEDDER_CKPT} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --use_path

echo "=========================================="
echo "Final checkpoint eval done: $(date)"
echo "Running evaluation on BEST checkpoint..."
echo "=========================================="

# --- Evaluate BEST checkpoint ---
# vt/pt1: all three agents saved together at best_oracle_line epoch.
# pt2/t1b1/twf/t1b1rl: use hybrid best (best_file + best_oracle_func + best_oracle_line).
if [ "${strategy}" = "vt" ] || [ "${strategy}" = "pt1" ]; then
    BEST_FILE="${CKPT_DIR}/file_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
    BEST_FUNC="${CKPT_DIR}/function_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
    BEST_LINE="${CKPT_DIR}/line_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
else
    BEST_FILE="${CKPT_DIR}/file_agent_${strategy}${reward_tag}_seed${seed}_best_file.pt"
    BEST_FUNC="${CKPT_DIR}/function_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_func.pt"
    BEST_LINE="${CKPT_DIR}/line_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
fi

EVAL_DIR="${PROJECT_DIR}/outputs/eval/${dataset}/${strategy}${reward_tag}_seed${seed}_best"
mkdir -p "${EVAL_DIR}"
cd "${EVAL_DIR}"
echo "--- Evaluating BEST checkpoint ---"
python "${PROJECT_DIR}/src/rl/evaluate.py" \
    --file_agent_path ${BEST_FILE} \
    --func_agent_path ${BEST_FUNC} \
    --line_agent_path ${BEST_LINE} \
    --test_data_path ${DATA_DIR}/test_filtered.json \
    --device cuda \
    --evaluation_type both \
    --embedder_ckpt_path ${EMBEDDER_CKPT} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --use_path

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
