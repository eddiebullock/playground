#!/usr/bin/env bash
#SBATCH -J msr_gemma4_smoke
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/gemma4_smoke_%j.out
#SBATCH -e logs/gemma4_smoke_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p results/test_runs logs

# Keep caches on project storage (not $HOME).
HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}/modules" "${HF_CACHE_DIR}/transformers" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/torch"
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

MAX_TRIALS="${MAX_TRIALS:-5}"
CONDITION="${CONDITION:-multimodal}"
STAGE="${STAGE:-both}"

echo "Running Gemma4 EU-Emotions ${CONDITION} smoke (${MAX_TRIALS} trials, stage=${STAGE})..."

python -m scripts.evaluate \
  --model gemma4 \
  --dataset eu_emotions \
  --condition "${CONDITION}" \
  --stage "${STAGE}" \
  --max_frames 16 \
  --fps 1 \
  --data_root data/eu_emotions_118 \
  --manifest data/eu_emotions_118_manifest.json \
  --max_trials "${MAX_TRIALS}" \
  --output "results/test_runs/test_gemma4_eu_emotions_${CONDITION}_${MAX_TRIALS}trials.json"

echo "Gemma4 smoke test complete."

