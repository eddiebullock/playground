#!/usr/bin/env bash
#SBATCH -J msr_test
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o logs/test_job_%j.out
#SBATCH -e logs/test_job_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p results/test_runs logs

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

# Override on submit: MAX_TRIALS=20 sbatch test_job.sh
MAX_TRIALS="${MAX_TRIALS:-50}"

echo "Running EU-Emotions manifest pipeline test (${MAX_TRIALS} trials)..."

python -m scripts.evaluate \
  --model qwen2vl \
  --dataset eu_emotions \
  --n_frames 4 \
  --data_root data/eu_emotions_118 \
  --manifest data/eu_emotions_118_manifest.json \
  --max_trials "${MAX_TRIALS}" \
  --max_new_tokens 96 \
  --output "results/test_runs/test_qwen2vl_eu_emotions_${MAX_TRIALS}trials.json"

echo "Test job complete."

