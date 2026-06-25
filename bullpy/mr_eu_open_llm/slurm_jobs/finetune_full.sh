#!/usr/bin/env bash
#SBATCH -J msr_finetune
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/finetune_full_%j.out
#SBATCH -e logs/finetune_full_%j.err
#
# Walltime tuned for CSD3 backfill. Gemma4 multimodal full FT ~2-3h observed;
# video-only VLMs often faster. Override: sbatch --time=04:00:00 ...

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-gemma4}"
# Auto-resolve training modality: gemma4=multimodal, vision-only VLMs=video_only.
if [[ -z "${CONDITION:-}" ]]; then
  if [[ "${MODEL}" == "gemma4" ]]; then
    CONDITION=multimodal
  else
    CONDITION=video_only
  fi
else
  CONDITION="${CONDITION}"
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

HPARAMS_FILE="${PROJECT_ROOT}/results/stats/best_finetune_hparams.json"
if [[ -z "${LR:-}" || -z "${RANK:-}" ]] && [[ -f "${HPARAMS_FILE}" ]]; then
  read -r _LR _RANK < <(python -c "
import json
from pathlib import Path
b = json.loads(Path('${HPARAMS_FILE}').read_text())['best']
print(b['learning_rate'], b['lora_r'])
")
  LR="${LR:-${_LR}}"
  RANK="${RANK:-${_RANK}}"
fi
LR="${LR:-1e-4}"
RANK="${RANK:-32}"

if ! ffmpeg -version >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not working; run: conda install -c conda-forge ffmpeg"
fi

cd "${PROJECT_ROOT}"
mkdir -p results/finetune/full_runs logs

python -m scripts.prepare_finetune_data \
  --root data/mindreading \
  --output_dir data/mindreading \
  --modality "${CONDITION}" \
  --write_manifest

TRAIN_FILE=data/mindreading/train_full.jsonl
VAL_FILE=data/mindreading/val_full.jsonl

OUT_DIR="results/finetune/full_runs/${MODEL}/run_${SLURM_JOB_ID}"

LORA_ALPHA=$(( RANK * 2 ))

echo "Running full finetuning for model=${MODEL}, condition=${CONDITION}, lr=${LR}, r=${RANK}, alpha=$(( RANK * 2 ))..."

python -m scripts.finetune \
  --model "${MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --data_root data/mindreading \
  --condition "${CONDITION}" \
  --output_dir "${OUT_DIR}" \
  --learning_rate "${LR}" \
  --lora_r "${RANK}" \
  --lora_alpha "${LORA_ALPHA}"

echo "Full finetuning job complete."

