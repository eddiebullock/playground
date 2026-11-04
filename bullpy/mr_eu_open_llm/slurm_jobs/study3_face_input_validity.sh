#!/usr/bin/env bash
#SBATCH -J s3_a3
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_a3_%x_%j.out
#SBATCH -e logs/study3_a3_%x_%j.err
#
# Amendment A3: A1's design on single_face, one job per model. See research/PREREG_A3_DRAFT.md.
# Scoring needs A3_FROZEN=1, which may be set only after A3 is frozen. Resumable.
#
#   Mechanics, no scores written: MECHANICS_ONLY=2 MODEL=gemma4 sbatch --time=00:15:00 slurm_jobs/study3_face_input_validity.sh
#   Scored, after freezing:       A3_FROZEN=1 MODEL=gemma4 sbatch slurm_jobs/study3_face_input_validity.sh
#   Combine, after all three:     A3_FROZEN=1 SUMMARISE=1 sbatch --time=00:10:00 slurm_jobs/study3_face_input_validity.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-gemma4}"
A3_FROZEN="${A3_FROZEN:-0}"
MECHANICS_ONLY="${MECHANICS_ONLY:-}"
SUMMARISE="${SUMMARISE:-0}"

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
mkdir -p logs
# The cv2 4.11 overlay comes first: cv2 4.13 returns junk frames on interlaced clips.
export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python research/scripts/decoder_preflight.py

EXTRA=()
if [ "${A3_FROZEN}" = "1" ]; then
  EXTRA+=(--a3_frozen)
fi
if [ "${SUMMARISE}" = "1" ]; then
  python -m scripts.face_input_validity --summarise "${EXTRA[@]}"
elif [ -n "${MECHANICS_ONLY}" ]; then
  python -m scripts.face_input_validity --model "${MODEL}" --mechanics_only "${MECHANICS_ONLY}"
else
  python -m scripts.face_input_validity --model "${MODEL}" "${EXTRA[@]}"
fi
echo "Done: ${MODEL}"
