#!/usr/bin/env bash
#SBATCH -J msr_baseline
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o logs/baseline_%x_%j.out
#SBATCH -e logs/baseline_%x_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-qwen2vl}"
DATASET="${DATASET:-eu_emotions}"
N_FRAMES="${N_FRAMES:-4}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p results/baseline logs

echo "Running baseline evaluation for model=${MODEL}, dataset=${DATASET}, n_frames=${N_FRAMES}..."

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --n_frames "${N_FRAMES}" \
  ${EXTRA_ARGS:-}

echo "Baseline evaluation complete."

