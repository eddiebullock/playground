#!/usr/bin/env bash
#SBATCH -J s4_rmet_smoke
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/study4_rmet_smoke_%x_%j.out
#SBATCH -e logs/study4_rmet_smoke_%x_%j.err
#
# study4 RMET smoke (few items). ONE MODEL PER JOB — never mix gemma4 + molmo2 envs.
#
#   unset VIRTUAL_ENV
#   MODEL=qwen3vl sbatch -J s4_smoke_qwen3vl study4_rmet/slurm_jobs/rmet_model_smoke.sh
#   MODEL=gemma4  sbatch -J s4_smoke_gemma4  study4_rmet/slurm_jobs/rmet_model_smoke.sh
#   MODEL=molmo2  sbatch -J s4_smoke_molmo2  study4_rmet/slurm_jobs/rmet_model_smoke.sh
#
# Prefight (run on login node before full jobs):
#   ls ~/rds/hpc-work/study2/models/{qwen3vl,gemma4,molmo2}/config.json

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study4_rmet
ENV_ROOT=~/rds/hpc-work/study2
# Molmo2 venv lives under study3 (shared Transformers 4.57 install); read-only reuse.
MOLMO2_VENV="${MOLMO2_VENV:-~/rds/hpc-work/study3/venvs/molmo2}"
MODEL="${MODEL:-qwen3vl}"
MAX_ITEMS="${MAX_ITEMS:-3}"
N_SAMPLES="${N_SAMPLES:-2}"
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
mkdir -p logs study4_rmet/results/model/"${MODEL}"

HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}/modules" "${HF_CACHE_DIR}/transformers" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/torch"
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
# Shared weights from study2 (read-only).
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export MR_EU_HPC_PROJECT="study4_rmet"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "study4 RMET smoke: model=${MODEL} max_items=${MAX_ITEMS} n_samples=${N_SAMPLES}"
echo "env: ${ENV_NAME}$([ "${USE_MOLMO2_VENV}" = "1" ] && echo " + ${MOLMO2_VENV}") | $(command -v python)"
echo "transformers $(python -c 'import transformers; print(transformers.__version__)')"

python study4_rmet/scripts/evaluate_rmet.py \
  --model "${MODEL}" \
  --max_items "${MAX_ITEMS}" \
  --n_samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --manifest study4_rmet/data/rmet/stimuli/manifest.json \
  --stim_root study4_rmet/data/rmet/stimuli \
  --output "study4_rmet/results/model/${MODEL}/rmet_eval_${MODEL}_smoke${MAX_ITEMS}_seed${SEED}.json"

echo "study4 RMET smoke complete."
