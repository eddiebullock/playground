#!/usr/bin/env bash
#SBATCH -J s3_vgates
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -o logs/study3_vision_gates_%x_%j.out
#SBATCH -e logs/study3_vision_gates_%x_%j.err
#
# Validity gates G2 to G4 on vision paths, one model per job.
#
#   Smoke: MODEL=qwen3vl MAX_CLIPS=2 sbatch --time=00:40:00 --qos=intr slurm_jobs/study3_vision_gates.sh
#   Full:  MODEL=qwen3vl sbatch slurm_jobs/study3_vision_gates.sh
#
# Reads the base human option lookup and rotates in memory. It does not rewrite the
# rotation lookups, which pending s3_rotation arrays read when they start.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"
MAX_CLIPS="${MAX_CLIPS:-}"
GATES="${GATES:-g2,g3,g4}"
DTYPES="${DTYPES:-bfloat16,float32}"
N_G3="${N_G3:-12}"
FLOAT32_MAX_FRAMES="${FLOAT32_MAX_FRAMES:-}"

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
mkdir -p logs research/reproduction
# Amendment A1.1: cv2 4.11 ahead of the environment's copy for all three models. cv2 4.13
# returns junk frames on interlaced clips.
export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Stops the job under set -e unless the decoder is pinned and reads frames correctly.
python research/scripts/decoder_preflight.py

EXTRA=()
# Qwen has up to about 20,000 image tokens, which does not fit in float32 SDPA.
if [ "${MODEL}" = "qwen3vl" ]; then
  FLOAT32_MAX_FRAMES="${FLOAT32_MAX_FRAMES:-6}"
fi
if [ -n "${FLOAT32_MAX_FRAMES}" ]; then
  EXTRA+=(--float32_max_frames "${FLOAT32_MAX_FRAMES}")
fi
OUTPUT="research/reproduction/vision_gates_${MODEL}.json"
if [ -n "${MAX_CLIPS}" ]; then
  EXTRA+=(--max_clips "${MAX_CLIPS}")
  OUTPUT="logs/vision_gates_${MODEL}_smoke_${SLURM_JOB_ID:-local}.json"
fi

python -m scripts.vision_gates \
  --model "${MODEL}" \
  --gates "${GATES}" \
  --dtypes "${DTYPES}" \
  --n_g3 "${N_G3}" \
  --output "${OUTPUT}" \
  "${EXTRA[@]}"

echo "Done: ${OUTPUT}"
