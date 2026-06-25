#!/usr/bin/env bash
#SBATCH -J msr_baseline
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Full 118-trial baseline eval. Gemma4 both-stage ~25-45 min; request 2h for queue fit.
# Smoke: MAX_TRIALS=5 with sbatch --time=00:30:00
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/baseline_%x_%j.out
#SBATCH -e logs/baseline_%x_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-qwen2vl}"
DATASET="${DATASET:-eu_emotions}"
CONDITION="${CONDITION:-video_only}"
MAX_FRAMES="${MAX_FRAMES:-16}"
FPS="${FPS:-1}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"
MAX_TRIALS="${MAX_TRIALS:-118}"
TEMPERATURE="${TEMPERATURE:-0.1}"
STAGE="${STAGE:-both}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

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

echo "Running protocol v2 eval: model=${MODEL}, dataset=${DATASET}, condition=${CONDITION}, stage=${STAGE}, fps=${FPS}, max_frames=${MAX_FRAMES}..."

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --condition "${CONDITION}" \
  --max_frames "${MAX_FRAMES}" \
  --fps "${FPS}" \
  --stage "${STAGE}" \
  --data_root "${DATA_ROOT}" \
  --manifest "${MANIFEST}" \
  --max_trials "${MAX_TRIALS}" \
  --temperature "${TEMPERATURE}" \
  --output "results/baseline/${DATASET}/${MODEL}/eval_v2_${DATASET}_${MODEL}_${CONDITION}_fps${FPS}_cap${MAX_FRAMES}_two_stage_seed42.json" \
  ${EXTRA_ARGS:-}

echo "Baseline evaluation complete."

