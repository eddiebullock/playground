#!/usr/bin/env bash
set -euo pipefail

# This script is intended to be run on CSD3 after SSH:
#   ssh eb2007@login.hpc.cam.ac.uk
#   cd ~/rds/hpc-work/study2
#   ./setup_hpc.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating directory structure under ${PROJECT_ROOT}..."
mkdir -p "${PROJECT_ROOT}/data/eu_emotions"
mkdir -p "${PROJECT_ROOT}/data/mindreading"
mkdir -p "${PROJECT_ROOT}/data/rmet"
mkdir -p "${PROJECT_ROOT}/models/qwen2vl"
mkdir -p "${PROJECT_ROOT}/models/internvl2"
mkdir -p "${PROJECT_ROOT}/models/llavanext"
mkdir -p "${PROJECT_ROOT}/results"
mkdir -p "${PROJECT_ROOT}/scripts"
mkdir -p "${PROJECT_ROOT}/slurm_jobs"

echo "Loading miniconda module..."
module load miniconda || module load miniconda3 || true

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found after loading miniconda module. Please check module names on CSD3."
  exit 1
fi

ENV_NAME="mr_eu_open_llm"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda environment ${ENV_NAME} already exists. Skipping creation."
else
  echo "Creating conda environment ${ENV_NAME} from environment.yml..."
  conda env create -f "${PROJECT_ROOT}/environment.yml" || {
    echo "Falling back to manual environment creation..."
    conda create -y -n "${ENV_NAME}" python=3.10
    conda activate "${ENV_NAME}"
    pip install -r "${PROJECT_ROOT}/requirements.txt"
  }
fi

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Ensuring huggingface_hub / huggingface-cli is available..."
python -m pip install --upgrade huggingface_hub

echo "Downloading models with huggingface-cli..."
MODEL_DIR="${PROJECT_ROOT}/models"

huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir "${MODEL_DIR}/qwen2vl" \
  --local-dir-use-symlinks False || echo "Qwen2-VL-7B-Instruct download failed or partially cached."

huggingface-cli download OpenGVLab/InternVL2-8B \
  --local-dir "${MODEL_DIR}/internvl2" \
  --local-dir-use-symlinks False || echo "InternVL2-8B download failed or partially cached."

huggingface-cli download lmms-lab/llava-next-interleave-7b \
  --local-dir "${MODEL_DIR}/llavanext" \
  --local-dir-use-symlinks False || echo "LLaVA-NeXT download failed or partially cached."

echo "HPC setup complete."

