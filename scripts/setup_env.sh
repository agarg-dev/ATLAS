#!/bin/bash
# One-time setup: create a Python virtual environment and install dependencies.
# Usage: bash scripts/setup_env.sh [env_dir]
#   env_dir: optional path for the virtual environment (default: ./venv)
#
# Prerequisites:
#   - Python 3.9+ on your PATH
#   - CUDA toolkit installed if you want GPU support
#   - faiss-cpu is installed via pip; on HPC clusters with a system faiss module
#     you may prefer to use that instead and skip faiss-cpu from requirements.txt

set -e

ENV_DIR="${1:-${PWD}/venv}"

echo "=========================================="
echo "Setting up C2C environment"
echo "Python: $(python3 --version)"
echo "Env dir: ${ENV_DIR}"
echo "=========================================="

if [ -d "${ENV_DIR}" ]; then
    echo "Virtual environment already exists at ${ENV_DIR}"
    echo "To recreate: rm -rf ${ENV_DIR} && bash scripts/setup_env.sh"
else
    echo "Creating virtual environment..."
    python3 -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import faiss; print('FAISS: OK')"
python -c "import wandb; print(f'wandb: {wandb.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import git; print('GitPython: OK')"

echo ""
echo "=========================================="
echo "Setup complete."
echo ""
echo "Activate with:  source ${ENV_DIR}/bin/activate"
echo "=========================================="
