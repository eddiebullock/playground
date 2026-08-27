#!/usr/bin/env bash
#SBATCH -J s4_rmet_full
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# 36 items x (1 det + 10 samples) — short for A100; override if needed.
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o logs/study4_rmet_full_%x_%j.out
#SBATCH -e logs/study4_rmet_full_%x_%j.err
#
# Full 36-item RMET eval. Submit ONE JOB PER MODEL after smoke passes.
# Do NOT submit until smoke is green — walltime across 3 models is non-trivial.
#
#   unset VIRTUAL_ENV
#   MODEL=qwen3vl N_SAMPLES=10 sbatch -J s4_full_qwen3vl study4_rmet/slurm_jobs/rmet_model_full.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study4_rmet
ENV_ROOT=~/rds/hpc-work/study2
MOLMO2_VENV="${MOLMO2_VENV:-~/rds/hpc-work/study3/venvs/molmo2}"
MODEL="${MODEL:-qwen3vl}"
N_SAMPLES="${N_SAMPLES:-10}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.7}"
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
mkdir -p "${HF_CACHE_DIR}"/{modules,transformers,datasets,torch}
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export MR_EU_HPC_PROJECT="study4_rmet"

echo "study4 RMET FULL: model=${MODEL} n_samples=${N_SAMPLES} T=${SAMPLE_TEMPERATURE}"
python study4_rmet/scripts/evaluate_rmet.py \
  --model "${MODEL}" \
  --n_samples "${N_SAMPLES}" \
  --sample_temperature "${SAMPLE_TEMPERATURE}" \
  --seed "${SEED}" \
  --manifest study4_rmet/data/rmet/stimuli/manifest.json \
  --stim_root study4_rmet/data/rmet/stimuli \
  --output "study4_rmet/results/model/${MODEL}/rmet_eval_${MODEL}_full_seed${SEED}.json"

echo "study4 RMET full complete."
