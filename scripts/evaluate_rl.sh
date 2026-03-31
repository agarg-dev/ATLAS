#!/bin/bash
# Evaluate trained RL agents for C2C bug localization.
#
# Usage: bash scripts/evaluate_rl.sh [file.pt] [func.pt] [line.pt] [strategy] [seed] [variant] [reward_type] [dataset]
#   variant:     final (default) | best
#   reward_type: sparse (default) | intermediate
#   dataset:     swebench (default) | elasticsearch
#
# Examples:
#   bash scripts/evaluate_rl.sh "" "" "" twf 42 best intermediate elasticsearch
#   bash scripts/evaluate_rl.sh "" "" "" twf 42 final intermediate swebench

strategy=${4:-vt}
seed=${5:-42}
variant=${6:-final}
reward_type=${7:-sparse}
dataset=${8:-swebench}
reward_tag="_${reward_type}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/${dataset}/hierarchical_dataset"
EMBEDDER_CKPT="${PROJECT_DIR}/outputs/checkpoints/contrastive/${dataset}/hardneg/best_codebert_triplet.pt"
TOKENIZER_PATH="${PROJECT_DIR}/models/codebert/codebert_tokenizer"
CKPT_DIR="${PROJECT_DIR}/outputs/checkpoints/rl/${dataset}"

if [ "${variant}" = "best" ]; then
    if [ "${strategy}" = "vt" ] || [ "${strategy}" = "pt1" ]; then
        default_file="${CKPT_DIR}/file_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
        default_func="${CKPT_DIR}/function_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
        default_line="${CKPT_DIR}/line_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
    else
        default_file="${CKPT_DIR}/file_agent_${strategy}${reward_tag}_seed${seed}_best_file.pt"
        default_func="${CKPT_DIR}/function_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_func.pt"
        default_line="${CKPT_DIR}/line_agent_${strategy}${reward_tag}_seed${seed}_best_oracle_line.pt"
    fi
else
    default_file="${CKPT_DIR}/file_agent_${strategy}${reward_tag}_seed${seed}.pt"
    default_func="${CKPT_DIR}/function_agent_${strategy}${reward_tag}_seed${seed}.pt"
    default_line="${CKPT_DIR}/line_agent_${strategy}${reward_tag}_seed${seed}.pt"
fi

file_agent=${1:-${default_file}}
func_agent=${2:-${default_func}}
line_agent=${3:-${default_line}}

mkdir -p "${PROJECT_DIR}/outputs/logs/rl"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src/rl:$PYTHONPATH"
export HF_HOME="${PROJECT_DIR}/models/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "=========================================="
echo "Dataset: ${dataset}  Strategy: ${strategy}  Seed: ${seed}  Variant: ${variant}  Reward: ${reward_type}"
echo "File agent:  ${file_agent}"
echo "Func agent:  ${func_agent}"
echo "Line agent:  ${line_agent}"
echo "Started: $(date)"
echo "=========================================="

EVAL_OUT="${PROJECT_DIR}/outputs/eval/${dataset}/${strategy}${reward_tag}_seed${seed}_${variant}"
mkdir -p "${EVAL_OUT}"
cd "${EVAL_OUT}"
python "${PROJECT_DIR}/src/rl/evaluate.py" \
    --file_agent_path ${file_agent} \
    --func_agent_path ${func_agent} \
    --line_agent_path ${line_agent} \
    --test_data_path ${DATA_DIR}/test_filtered.json \
    --tokenizer_path ${TOKENIZER_PATH} \
    --embedder_ckpt_path ${EMBEDDER_CKPT} \
    --evaluation_type both \
    --use_path

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
