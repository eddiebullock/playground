#!/usr/bin/env bash
#SBATCH -J s3_a7
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_a7_%x_%j.out
#SBATCH -e logs/study3_a7_%x_%j.err
#
# Draft amendment A7, vision-path cuts. See research/PREREG_A7_DRAFT.md and scripts/a7_*.py.
# Nothing here runs before the freeze (tag a7-freeze) and the code sync with hash check
# (ROADMAP 2.1). Every task is capped at 59 minutes.
#
#   Stage 0:  A7_FROZEN=1 STEP=stage0 MODEL=gemma4 RECORDS=<ladder records of the 24 calibration clips> \
#                 REGION=1 sbatch slurm_jobs/study3_a7.sh
#             Calibration clips only; the job refuses any other clip id.
#   Stage 1:  A7_FROZEN=1 STEP=run MODEL=gemma4 ARMS=route DECISION_DATE=2026-11-XX N_SHARDS=2 \
#                 sbatch --array=0-1 slurm_jobs/study3_a7.sh
#             Only arms that Stage 0 admits, and only with a DECISIONS.md entry made before any
#             A2 analysis output was opened.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
MODEL="${MODEL:-}"
STEP="${STEP:-}"
A7_FROZEN="${A7_FROZEN:-0}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-1}"

if [ "${A7_FROZEN}" != "1" ]; then
  echo "A7 is not frozen. Set A7_FROZEN=1 only after the tag a7-freeze exists and the hashes are checked." >&2
  exit 1
fi
case "${MODEL}" in
  gemma4|qwen3vl) ;;
  *) echo "MODEL must be gemma4 or qwen3vl, got '${MODEL}'." >&2; exit 1 ;;
esac

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi
source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs results/a7
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR=~/rds/hpc-work/study2/models
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore::UserWarning"

case "${STEP}" in
  stage0)
    EXTRA=()
    if [ "${REGION:-0}" = "1" ]; then
      EXTRA+=(--region)
    fi
    python -m scripts.a7_stage0 --model "${MODEL}" --records "${RECORDS:?RECORDS is required}" "${EXTRA[@]}" \
      --a7_frozen --output "results/a7/a7_stage0_${MODEL}.json"
    ;;
  run)
    python -m scripts.a7_run --model "${MODEL}" --arms "${ARMS:?ARMS is required}" \
      --stage0 "results/a7/a7_stage0_${MODEL}.json" --decision_date "${DECISION_DATE:?DECISION_DATE is required}" \
      --a7_frozen --shard "${SHARD}" --n_shards "${N_SHARDS}"
    ;;
  *)
    echo "STEP must be stage0 or run, got '${STEP}'." >&2
    exit 1
    ;;
esac
