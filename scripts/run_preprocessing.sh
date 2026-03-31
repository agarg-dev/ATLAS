#!/bin/bash
# Full preprocessing pipeline for C2C — Elasticsearch (Java) dataset.
#
# Steps:
#   1. parse_elasticsearch.py         (requires: data/processed_samples_train_all.json)
#   2. build_triplets.py              (requires: data/repos/ with cloned repos)
#   3. train_bert.py                  (initial CodeBERT fine-tuning)
#   4. build_faiss_index.py           (index trained embeddings)
#   5. mine_hard_negatives.py         (find hard negatives via FAISS)
#   6. train_bert.py                  (retrain with hard negatives)
#   7. build_faiss_index.py           (rebuild index with hard-neg model)
#   8. build_hierarchical_dataset.py  (create RL training data)
#   9. eval_retrieval.py              (evaluate retrieval: Hit@K, MRR)
#
# Usage:
#   bash scripts/run_preprocessing.sh            # run all steps
#   bash scripts/run_preprocessing.sh 3          # resume from step 3

set -e

START_STEP=${1:-1}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/elasticsearch"
MODELS_DIR="${PROJECT_DIR}/models/codebert"
CONTRASTIVE_CKPT_DIR="${PROJECT_DIR}/outputs/checkpoints/contrastive/elasticsearch"
FAISS_DIR="${DATA_DIR}/faiss_index_codebert"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/outputs/wandb"
export HF_HOME="${PROJECT_DIR}/models/huggingface_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONPATH="${PROJECT_DIR}/src/contrastive:$PYTHONPATH"

mkdir -p "${PROJECT_DIR}/outputs/logs/contrastive"
mkdir -p "${CONTRASTIVE_CKPT_DIR}"
mkdir -p "${DATA_DIR}/hierarchical_dataset"

echo "=========================================="
echo "Start step: ${START_STEP}"
echo "Started: $(date)"
echo "Pipeline: Elasticsearch (Java)"
echo "=========================================="

cd "${PROJECT_DIR}/src/contrastive"

# ---------------------------------------------------------------------------
# Step 1: Parse raw Elasticsearch dataset
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 1 ]; then
    RAW_INPUT="${PROJECT_DIR}/data/processed_samples_train_all.json"
    PARSED_OUTPUT="${DATA_DIR}/bug_localization_dataset.json"

    if [ ! -f "${RAW_INPUT}" ]; then
        echo "ERROR: ${RAW_INPUT} not found. Obtain this file before running step 1."
        exit 1
    fi

    echo "[Step 1] Parsing raw Elasticsearch dataset..."
    mkdir -p "${DATA_DIR}"
    python parse_elasticsearch.py \
        --input_path  "${RAW_INPUT}" \
        --output_path "${PARSED_OUTPUT}"
    echo "[Step 1] Done."
fi

# ---------------------------------------------------------------------------
# Step 2: Build triplet dataset (filter + span extraction + triplet generation)
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 2 ]; then
    TRIPLET_OUTPUT="${DATA_DIR}/triplet_dataset.json"

    echo "[Step 2] Building triplet dataset (clones repos — may take a while)..."
    python build_triplets.py \
        --input_path      "${DATA_DIR}/bug_localization_dataset.json" \
        --output_path     "${TRIPLET_OUTPUT}" \
        --repos_cache_dir "${PROJECT_DIR}/data/repos" \
        --extensions      .java
    echo "[Step 2] Done."
fi

# ---------------------------------------------------------------------------
# Step 3: Initial CodeBERT fine-tuning
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 3 ]; then
    echo "[Step 3] Fine-tuning CodeBERT (initial, random negatives)..."
    python train_bert.py \
        --triplet_path "${DATA_DIR}/triplet_dataset.json" \
        --output_dir   "${CONTRASTIVE_CKPT_DIR}" \
        --use_path \
        --symmetric_loss
    echo "[Step 3] Done."
fi

# ---------------------------------------------------------------------------
# Step 4: Build initial per-repo FAISS index
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 4 ]; then
    INITIAL_CKPT="${CONTRASTIVE_CKPT_DIR}/best_codebert_triplet.pt"

    echo "[Step 4] Building per-repo FAISS index (initial model)..."
    python build_faiss_index.py \
        --triplet_path    "${DATA_DIR}/triplet_dataset.json" \
        --output_dir      "${FAISS_DIR}" \
        --repos_dir       "${PROJECT_DIR}/data/repos" \
        --tokenizer_path  "${MODELS_DIR}/codebert_tokenizer" \
        --model_ckpt_path "${INITIAL_CKPT}" \
        --extension       .java \
        --use_path
    echo "[Step 4] Done."
fi

# ---------------------------------------------------------------------------
# Step 5: Mine hard negatives
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 5 ]; then
    HARDNEG_OUTPUT="${DATA_DIR}/triplet_dataset_hardneg.json"
    INITIAL_CKPT="${CONTRASTIVE_CKPT_DIR}/best_codebert_triplet.pt"

    echo "[Step 5] Mining hard negatives..."
    python mine_hard_negatives.py \
        --triplet_path    "${DATA_DIR}/triplet_dataset.json" \
        --index_dir       "${FAISS_DIR}" \
        --output_path     "${HARDNEG_OUTPUT}" \
        --tokenizer_path  "${MODELS_DIR}/codebert_tokenizer" \
        --model_ckpt_path "${INITIAL_CKPT}" \
        --num_hard_negs   1
    echo "[Step 5] Done."
fi

# ---------------------------------------------------------------------------
# Step 6: Retrain CodeBERT with hard negatives (warm-start)
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 6 ]; then
    HARDNEG_CKPT_DIR="${CONTRASTIVE_CKPT_DIR}/hardneg"
    INITIAL_CKPT="${CONTRASTIVE_CKPT_DIR}/best_codebert_triplet.pt"

    echo "[Step 6] Retraining CodeBERT with hard negatives (warm-start)..."
    mkdir -p "${HARDNEG_CKPT_DIR}"
    python train_bert.py \
        --triplet_path  "${DATA_DIR}/triplet_dataset_hardneg.json" \
        --output_dir    "${HARDNEG_CKPT_DIR}" \
        --model_path    "${INITIAL_CKPT}" \
        --lr 1e-5 \
        --warmup_steps 0 \
        --patience 5 \
        --epochs 15 \
        --batch_size 16 \
        --grad_accum 4 \
        --use_path \
        --symmetric_loss \
        --freeze_layers 0 \
        --selection_metric loss
    echo "[Step 6] Done."
fi

# ---------------------------------------------------------------------------
# Step 7: Rebuild FAISS index with hard-negative-trained model
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 7 ]; then
    HARDNEG_FAISS_DIR="${DATA_DIR}/faiss_index_codebert_hardneg"
    HARDNEG_CKPT="${CONTRASTIVE_CKPT_DIR}/hardneg/best_codebert_triplet.pt"

    echo "[Step 7] Rebuilding per-repo FAISS index (hard-neg model)..."
    python build_faiss_index.py \
        --triplet_path         "${DATA_DIR}/triplet_dataset_hardneg.json" \
        --output_dir           "${HARDNEG_FAISS_DIR}" \
        --tokenizer_path       "${MODELS_DIR}/codebert_tokenizer" \
        --model_ckpt_path      "${HARDNEG_CKPT}" \
        --reuse_contents_from  "${FAISS_DIR}" \
        --use_path
    echo "[Step 7] Done."
fi

# ---------------------------------------------------------------------------
# Step 8: Build hierarchical dataset for RL training
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 8 ]; then
    HIER_OUTPUT="${DATA_DIR}/hierarchical_dataset"
    HARDNEG_FAISS_DIR="${DATA_DIR}/faiss_index_codebert_hardneg"
    HARDNEG_CKPT="${CONTRASTIVE_CKPT_DIR}/hardneg/best_codebert_triplet.pt"

    echo "[Step 8] Building hierarchical retrieval dataset..."
    python build_hierarchical_dataset.py \
        --triplet_path    "${DATA_DIR}/triplet_dataset_hardneg.json" \
        --index_dir       "${HARDNEG_FAISS_DIR}" \
        --output_dir      "${HIER_OUTPUT}" \
        --tokenizer_path  "${MODELS_DIR}/codebert_tokenizer" \
        --model_ckpt_path "${HARDNEG_CKPT}" \
        --top_k           10
    echo "[Step 8] Done."
fi

# ---------------------------------------------------------------------------
# Step 9: Evaluate retrieval (Hit@K, Recall@K, MRR) — baseline vs. fine-tuned
# ---------------------------------------------------------------------------
if [ "${START_STEP}" -le 9 ]; then
    INITIAL_CKPT="${CONTRASTIVE_CKPT_DIR}/best_codebert_triplet.pt"
    HARDNEG_CKPT="${CONTRASTIVE_CKPT_DIR}/hardneg/best_codebert_triplet.pt"
    HARDNEG_FAISS_DIR="${DATA_DIR}/faiss_index_codebert_hardneg"
    BASE_FAISS_DIR="${DATA_DIR}/faiss_index_codebert_base"

    echo "[Step 9] Evaluating retrieval performance..."

    echo "--- Building base CodeBERT FAISS index (no fine-tuning) ---"
    python build_faiss_index.py \
        --triplet_path         "${DATA_DIR}/triplet_dataset.json" \
        --output_dir           "${BASE_FAISS_DIR}" \
        --tokenizer_path       "${MODELS_DIR}/codebert_tokenizer" \
        --model_ckpt_path      none \
        --reuse_contents_from  "${FAISS_DIR}" \
        --use_path

    echo "--- Base CodeBERT (no fine-tuning) ---"
    python eval_retrieval.py \
        --triplet_path    "${DATA_DIR}/triplet_dataset.json" \
        --index_dir       "${BASE_FAISS_DIR}" \
        --model_ckpt      none \
        --tokenizer_path  "${MODELS_DIR}/codebert_tokenizer" \
        --split           test

    echo "--- Initial model (random negatives) ---"
    for SPLIT in test val train; do
        echo "  split: ${SPLIT}"
        python eval_retrieval.py \
            --triplet_path    "${DATA_DIR}/triplet_dataset.json" \
            --index_dir       "${FAISS_DIR}" \
            --model_ckpt      "${INITIAL_CKPT}" \
            --tokenizer_path  "${MODELS_DIR}/codebert_tokenizer" \
            --split           "${SPLIT}"
    done

    echo "--- Hard-negative model ---"
    for SPLIT in test val train; do
        echo "  split: ${SPLIT}"
        python eval_retrieval.py \
            --triplet_path    "${DATA_DIR}/triplet_dataset_hardneg.json" \
            --index_dir       "${HARDNEG_FAISS_DIR}" \
            --model_ckpt      "${HARDNEG_CKPT}" \
            --tokenizer_path  "${MODELS_DIR}/codebert_tokenizer" \
            --split           "${SPLIT}"
    done

    echo "[Step 9] Done."
fi

echo "=========================================="
echo "Elasticsearch preprocessing complete."
echo "Finished: $(date)"
echo "=========================================="
