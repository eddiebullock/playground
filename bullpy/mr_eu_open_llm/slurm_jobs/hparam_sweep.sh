#!/usr/bin/env bash
#SBATCH -J msr_hparam
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-8
#SBATCH -o logs/hparam_sweep_%A_%a.out
#SBATCH -e logs/hparam_sweep_%A_%a.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-qwen2vl}"

# 3 learning rates × 3 ranks = 9 conditions
LRS=(1e-4 5e-5 1e-5)
RANKS=(8 16 32)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
LR_IDX=$(( IDX / 3 ))
RANK_IDX=$(( IDX % 3 ))

LR="${LRS[LR_IDX]}"
RANK="${RANKS[RANK_IDX]}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p results/finetune/hparam_sweep logs

TRAIN_FILE=data/mindreading/train_subset_100.jsonl
VAL_FILE=data/mindreading/val_subset_50.jsonl

OUT_DIR="results/finetune/hparam_sweep/${MODEL}/run_lr${LR}_r${RANK}"

echo "Running hparam sweep for model=${MODEL}, lr=${LR}, r=${RANK}..."

python -m scripts.finetune \
  --model "${MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --output_dir "${OUT_DIR}" \
  --learning_rate "${LR}" \
  --lora_r "${RANK}"

echo "Hyperparameter sweep job complete."

