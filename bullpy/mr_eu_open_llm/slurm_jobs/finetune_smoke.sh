#!/usr/bin/env bash
#SBATCH -J msr_ft_smoke
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/finetune_smoke_%j.out
#SBATCH -e logs/finetune_smoke_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MODEL="${MODEL:-gemma4}"
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

if ! ffmpeg -version >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not working; after conda activate run:"
  echo "  export CONDA_PKGS_DIRS=\"\${PROJECT_ROOT}/conda_pkgs\""
  echo "  conda install -y -c conda-forge ffmpeg"
fi

cd "${PROJECT_ROOT}"
mkdir -p logs results/finetune/smoke

HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
export HF_HOME="${HF_CACHE_DIR}"
export TOKENIZERS_PARALLELISM="false"

echo "=== Prepare Mindreading JSONL (multimodal) ==="
python -m scripts.prepare_finetune_data \
  --root data/mindreading \
  --output_dir data/mindreading \
  --modality "${CONDITION}" \
  --write_manifest

echo "=== Fine-tune smoke: 4 train samples, 1 step, 4 val eval ==="
OUT_DIR="results/finetune/smoke/${MODEL}_${SLURM_JOB_ID}"
python -m scripts.finetune \
  --model "${MODEL}" \
  --train_file data/mindreading/train_subset_100.jsonl \
  --val_file data/mindreading/val_subset_50.jsonl \
  --data_root data/mindreading \
  --condition "${CONDITION}" \
  --output_dir "${OUT_DIR}" \
  --max_train_samples 4 \
  --max_val_eval 4 \
  --max_steps 1 \
  --epochs 1 \
  --learning_rate 5e-5 \
  --lora_r 8 \
  --lora_alpha 16

echo "Smoke complete: ${OUT_DIR}"
