#!/usr/bin/env bash
#SBATCH -J s3_a9
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_a9_%x_%j.out
#SBATCH -e logs/study3_a9_%x_%j.err
#
# Draft amendment A9 (Liu et al. 2022, Experiment II). See research/PREREG_A9_DRAFT.md.
# A copy of slurm_jobs/study3_a4.sh with the A9 runner. Resumable; shards are by stimulus id.
# scripts/a9_run.py refuses to run unless A9's code is pinned in the draft and --a9_frozen is set.
#
#   Scored, after freezing: A9_FROZEN=1 MODEL=gemma4 N_SHARDS=12 sbatch --array=0-11 slurm_jobs/study3_a9.sh
#   Timing smoke:           A9_FROZEN=1 MODEL=gemma4 N_SHARDS=12 MAX_CELLS=40 sbatch --array=0 --time=00:30:00 slurm_jobs/study3_a9.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:?set MODEL}"
N_SHARDS="${N_SHARDS:?set N_SHARDS}"
A9_FROZEN="${A9_FROZEN:-0}"
MAX_CELLS="${MAX_CELLS:-}"
SHARD="${SLURM_ARRAY_TASK_ID:?submit as an array}"
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
if [ "${MODEL}" = "molmo2" ]; then
  source "${PROJECT_ROOT}/venvs/molmo2/bin/activate"
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

EXTRA=()
if [ "${A9_FROZEN}" = "1" ]; then
  EXTRA+=(--a9_frozen)
fi
if [ -n "${MAX_CELLS}" ]; then
  EXTRA+=(--max_cells "${MAX_CELLS}")
fi
python -m scripts.a9_run --model "${MODEL}" --shard "${SHARD}" --n_shards "${N_SHARDS}" "${EXTRA[@]}"
