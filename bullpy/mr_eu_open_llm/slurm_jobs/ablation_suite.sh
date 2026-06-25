#!/usr/bin/env bash
#SBATCH -J msr_ablation
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o logs/ablation_%j.out
#SBATCH -e logs/ablation_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

DATA_DIR_EU="${DATA_DIR_EU:-data/eu_emotions_118}"
DATA_DIR_MR="${DATA_DIR_MR:-data/mindreading}"
MANIFEST_EU="${MANIFEST_EU:-data/eu_emotions_118_manifest.json}"
MANIFEST_MR="${MANIFEST_MR:-data/mindreading_test_manifest.json}"
RESULTS_DIR="${RESULTS_DIR:-results/ablation}"
MAX_TRIALS="${MAX_TRIALS:-}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs "${RESULTS_DIR}"

EXTRA=()
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA+=(--max-trials "${MAX_TRIALS}")
fi

python run_ablation_suite.py \
  --data-dir-eu "${DATA_DIR_EU}" \
  --data-dir-mr "${DATA_DIR_MR}" \
  --manifest-eu "${MANIFEST_EU}" \
  --manifest-mr "${MANIFEST_MR}" \
  --results-dir "${RESULTS_DIR}" \
  "${EXTRA[@]}"

echo "Ablation suite complete."
