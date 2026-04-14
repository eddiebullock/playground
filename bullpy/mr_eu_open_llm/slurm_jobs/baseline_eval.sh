#!/usr/bin/env bash
#SBATCH -J msr_baseline
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/baseline_%x_%j.out
#SBATCH -e logs/baseline_%x_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-qwen2vl}"
DATASET="${DATASET:-eu_emotions}"
N_FRAMES="${N_FRAMES:-4}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"
MAX_TRIALS="${MAX_TRIALS:-118}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.1}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p "results/baseline/${DATASET}/${MODEL}" logs

# Avoid filling the login/home filesystem when Transformers caches dynamic modules.
# Keep caches on the HPC project storage.
HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}/modules" "${HF_CACHE_DIR}/transformers" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/torch"
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

echo "Running baseline evaluation for model=${MODEL}, dataset=${DATASET}, n_frames=${N_FRAMES}..."

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --n_frames "${N_FRAMES}" \
  --data_root "${DATA_ROOT}" \
  --manifest "${MANIFEST}" \
  --max_trials "${MAX_TRIALS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --output "results/baseline/${DATASET}/${MODEL}/baseline_${DATASET}_${MODEL}_frames${N_FRAMES}_${MAX_TRIALS}trials.json" \
  ${EXTRA_ARGS:-}

echo "Baseline evaluation complete."

