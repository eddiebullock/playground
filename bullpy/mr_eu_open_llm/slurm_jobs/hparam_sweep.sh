#!/usr/bin/env bash
#SBATCH -J msr_hparam
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-8
#SBATCH -o logs/hparam_sweep_%A_%a.out
#SBATCH -e logs/hparam_sweep_%A_%a.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

# Best EU baseline model (gemma4 multimodal)
MODEL="${MODEL:-gemma4}"
CONDITION="${CONDITION:-multimodal}"

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
conda activate "${ENV_NAME}"
set -u

if ! ffmpeg -version >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not working; run: conda install -c conda-forge ffmpeg"
fi

cd "${PROJECT_ROOT}"
mkdir -p results/finetune/hparam_sweep logs

TRAIN_FILE=data/mindreading/train_subset_100.jsonl
VAL_FILE=data/mindreading/val_subset_50.jsonl
PREP_MARKER=data/mindreading/.finetune_jsonl_ready

# Only task 0 prepares JSONL. Others must wait for PREP_MARKER (not just file existence:
# stale JSONL from a prior run caused tasks 1-3 to start before task 0 finished).
if [[ "${IDX}" == "0" ]]; then
  rm -f "${PREP_MARKER}"
  python -m scripts.prepare_finetune_data \
    --root data/mindreading \
    --output_dir data/mindreading \
    --modality "${CONDITION}" \
    --write_manifest
  touch "${PREP_MARKER}"
else
  echo "Waiting for task 0 prep marker ${PREP_MARKER}..."
  for _ in $(seq 1 120); do
    [[ -f "${PREP_MARKER}" ]] && break
    sleep 10
  done
fi
if [[ ! -f "${PREP_MARKER}" ]] || [[ ! -f "${TRAIN_FILE}" ]] || [[ ! -f "${VAL_FILE}" ]]; then
  echo "ERROR: prep incomplete (marker or JSONL missing)."
  exit 1
fi

OUT_DIR="results/finetune/hparam_sweep/${MODEL}/run_lr${LR}_r${RANK}"

echo "Running hparam sweep for model=${MODEL}, condition=${CONDITION}, lr=${LR}, r=${RANK}..."

LORA_ALPHA=$(( RANK * 2 ))

python -m scripts.finetune \
  --model "${MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --data_root data/mindreading \
  --condition "${CONDITION}" \
  --output_dir "${OUT_DIR}" \
  --learning_rate "${LR}" \
  --lora_r "${RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --epochs 2

echo "Hyperparameter sweep job complete."

