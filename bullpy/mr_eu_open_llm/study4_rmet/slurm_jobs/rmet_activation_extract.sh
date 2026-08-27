#!/usr/bin/env bash
#SBATCH -J s4_rmet_act
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o logs/study4_rmet_act_%x_%j.out
#SBATCH -e logs/study4_rmet_act_%x_%j.err
#
# Step 5: extract RMET activations at LAYER_DEPTH_FRACTIONS (last-token pool).
# ONE MODEL PER JOB — never mix gemma4 + molmo2 envs.
#
# Smoke (3 items):
#   unset VIRTUAL_ENV
#   MODEL=qwen3vl MAX_ITEMS=3 sbatch -J s4_act_smoke_qwen3vl study4_rmet/slurm_jobs/rmet_activation_extract.sh
#
# Full 36 items (after smoke OK):
#   MODEL=qwen3vl sbatch -J s4_act_qwen3vl study4_rmet/slurm_jobs/rmet_activation_extract.sh
#   MODEL=gemma4  sbatch -J s4_act_gemma4  study4_rmet/slurm_jobs/rmet_activation_extract.sh
#   MODEL=molmo2  sbatch -J s4_act_molmo2  study4_rmet/slurm_jobs/rmet_activation_extract.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study4_rmet
ENV_ROOT=~/rds/hpc-work/study2
MOLMO2_VENV="${MOLMO2_VENV:-~/rds/hpc-work/study3/venvs/molmo2}"
MODEL="${MODEL:-qwen3vl}"
MAX_ITEMS="${MAX_ITEMS:-}"
SEED="${SEED:-42}"

USE_MOLMO2_VENV=0
if [ "${MODEL}" = "molmo2" ]; then
  MOLMO2_VENV_EXPANDED=$(eval echo "${MOLMO2_VENV}")
  if [ -d "${MOLMO2_VENV_EXPANDED}" ]; then
    USE_MOLMO2_VENV=1
    MOLMO2_VENV="${MOLMO2_VENV_EXPANDED}"
  else
    ENV_NAME="${ENV_NAME:-mr_eu_molmo2}"
  fi
fi
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "note: dropping inherited VIRTUAL_ENV=${VIRTUAL_ENV}"
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
TAG="full"
EXTRA_ARGS=()
if [ -n "${MAX_ITEMS}" ]; then
  TAG="smoke${MAX_ITEMS}"
  EXTRA_ARGS+=(--max_items "${MAX_ITEMS}")
fi
OUT_DIR="study4_rmet/results/activations/${MODEL}/${TAG}"
mkdir -p logs "${OUT_DIR}"

HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}"/{modules,transformers,datasets,torch}
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export MR_EU_HPC_PROJECT="study4_rmet"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "study4 RMET activation extract: model=${MODEL} tag=${TAG}"
echo "env: ${ENV_NAME}$([ "${USE_MOLMO2_VENV}" = "1" ] && echo " + ${MOLMO2_VENV}")"

python study4_rmet/scripts/extract_rmet_activations.py \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --manifest study4_rmet/data/rmet/stimuli/manifest.json \
  --stim_root study4_rmet/data/rmet/stimuli \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "activation extract complete -> ${OUT_DIR}"
ls -lh "${OUT_DIR}"
