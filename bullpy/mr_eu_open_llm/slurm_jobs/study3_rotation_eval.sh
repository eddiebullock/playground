#!/usr/bin/env bash
#SBATCH -J s3_rotation
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_rotation_%x_%A_%a.out
#SBATCH -e logs/study3_rotation_%x_%A_%a.err
#
# Prereg amendment A1: rotated 6AFC, one array task per rotation (0..4).
# CONDITION=real shows the item's video; CONDITION=mismatch keeps its prompt and
# shows another item's video.
#
#   MODEL=gemma4 CONDITION=real sbatch --array=0-4%2 slurm_jobs/study3_rotation_eval.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"
ROTATION="${SLURM_ARRAY_TASK_ID:-${ROTATION:-0}}"
CONDITION="${CONDITION:-real}"
MISMATCH_SEED="${MISMATCH_SEED:-42}"
MAX_TRIALS="${MAX_TRIALS:-}"
SEED="${SEED:-42}"
N_OPTIONS="${N_OPTIONS:-6}"
MAX_FRAMES="${MAX_FRAMES:-16}"
FC_SAMPLES="${FC_SAMPLES:-20}"

MOLMO2_VENV="${MOLMO2_VENV:-${PROJECT_ROOT}/venvs/molmo2}"
USE_MOLMO2_VENV=0
if [ "${MODEL}" = "molmo2" ] && [ -d "${MOLMO2_VENV}" ]; then
  USE_MOLMO2_VENV=1
fi
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
if [ "${USE_MOLMO2_VENV}" = "1" ]; then
  source "${MOLMO2_VENV}/bin/activate"
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/baseline/eu_emotions/"${MODEL}"
# cv2 4.13 corrupts interlaced clips, so pin 4.11 ahead of the environment's copy.
export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"

python research/scripts/decoder_preflight.py
python -m paper_v2.option_score --write_rotation_lookups
LOOKUP="data/eu_emotions_human_entropy_rot${ROTATION}.json"

FC_SEQUENTIAL="${FC_SEQUENTIAL:-0}"
EVAL_EXTRA=()
if [ "${FC_SEQUENTIAL}" = "1" ] || [ "${MODEL}" = "qwen3vl" ]; then
  EVAL_EXTRA+=(--fc_sequential)
fi
if [ "${CONDITION}" = "mismatch" ]; then
  EVAL_EXTRA+=(--vision_mismatch_seed "${MISMATCH_SEED}")
fi
OUT_TAG=""
if [ -n "${MAX_TRIALS}" ]; then
  EVAL_EXTRA+=(--max_trials "${MAX_TRIALS}")
  OUT_TAG="_smoke"
fi

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset eu_emotions \
  --condition video_only \
  --stage stage2 \
  --max_frames "${MAX_FRAMES}" \
  --fps 1 \
  --seed "${SEED}" \
  --n_options "${N_OPTIONS}" \
  --data_root data/eu_emotions \
  --manifest data/eu_emotions_full_manifest.json \
  --human_options "${LOOKUP}" \
  --fc_samples "${FC_SAMPLES}" \
  "${EVAL_EXTRA[@]}" \
  --output "results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_video_only_seed${SEED}_rot${ROTATION}_${CONDITION}${OUT_TAG}.json"

echo "Done."
