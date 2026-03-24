#!/usr/bin/env bash
#SBATCH -J msr_finetune
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --qos=sl2
#SBATCH -o logs/finetune_full_%j.out
#SBATCH -e logs/finetune_full_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-qwen2vl}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p results/finetune/full_runs logs

TRAIN_FILE=data/mindreading/train_full.jsonl
VAL_FILE=data/mindreading/val_full.jsonl

OUT_DIR="results/finetune/full_runs/${MODEL}/run_${SLURM_JOB_ID}"

echo "Running full finetuning for model=${MODEL}..."

python -m scripts.finetune \
  --model "${MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --output_dir "${OUT_DIR}"

echo "Full finetuning job complete."

