#!/usr/bin/env bash
#SBATCH -J s3_a6
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_a6_%x_%j.out
#SBATCH -e logs/study3_a6_%x_%j.err
#
# Draft amendment A6, shared errors on CREMA-D. See research/PREREG_A6_DRAFT.md and
# scripts/a6_run.py. Frames are edited in memory; nothing here writes a frame or an edit.
# Do not submit any of this before the wait in the draft has ended, and no full run before
# the amendment is frozen and tagged.
#
#   Decoder:   STEP=decoder sbatch --time=00:15:00 -A BARON-COHEN-SL3-CPU -p icelake --gres=gpu:0 slurm_jobs/study3_a6.sh
#   Landmarks: STEP=landmarks sbatch --time=02:00:00 -A BARON-COHEN-SL3-CPU -p icelake --gres=gpu:0 slurm_jobs/study3_a6.sh
#              Pin the output in the design block. The step refuses to overwrite it.
#   AU list:   STEP=au_manifest sbatch --time=00:10:00 -A BARON-COHEN-SL3-CPU -p icelake --gres=gpu:0 slurm_jobs/study3_a6.sh
#              Then research/scripts/extract_au_measures.py, unchanged, with
#              --manifest results/a6/au_manifest.json --data-root data/crema_d.
#   Mechanics: STEP=mechanics MODEL=gemma4 MECHANICS_ITEMS=2 sbatch --time=01:00:00 slurm_jobs/study3_a6.sh
#   Baselines: A6_FROZEN=1 CONFIRM_GATES_PASSED=1 STEP=baselines MODEL=... sbatch --time=01:00:00 slurm_jobs/study3_a6.sh
#   Stage 1:   A6_FROZEN=1 CONFIRM_GATES_PASSED=1 STEP=run STAGE=1 N_SHARDS=2 MODEL=... sbatch --array=0-1 slurm_jobs/study3_a6.sh
#   Stage 3:   A6_FROZEN=1 CONFIRM_GATES_PASSED=1 STEP=run STAGE=3 N_SHARDS=2 MODEL=... sbatch --array=0-1 slurm_jobs/study3_a6.sh
#              Needs results/a6/a6_stage2_<model>.json, which must reach Stage 3.
#   Analysis:  A6_FROZEN=1 CONFIRM_GATES_PASSED=1 STEP=analyse STAGE=1 sbatch --time=00:30:00 -A BARON-COHEN-SL3-CPU -p icelake --gres=gpu:0 slurm_jobs/study3_a6.sh
#              Stages 2 and 3 also take MODEL. Read one stage before running the next.
#
# Resubmit a lost task with the same N_SHARDS and --array=<task>. Qwen fits two shards at the
# 12 hour limit, and Stage 3 on two shards is an estimate until the mechanics run measures it.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
MODEL="${MODEL:-}"
STEP="${STEP:-}"
STAGE="${STAGE:-}"
MECHANICS_ITEMS="${MECHANICS_ITEMS:-2}"
A6_FROZEN="${A6_FROZEN:-0}"
CONFIRM_GATES_PASSED="${CONFIRM_GATES_PASSED:-0}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-}"

case "${STEP}" in
  decoder|landmarks|au_manifest|mechanics|baselines|run|analyse) ;;
  *)
    echo "STEP must be decoder, landmarks, au_manifest, mechanics, baselines, run or analyse, got '${STEP}'." >&2
    exit 1
    ;;
esac

if [ "${STEP}" = "run" ]; then
  if ! [[ "${N_SHARDS}" =~ ^[1-9][0-9]*$ ]] || [ "${SHARD}" -ge "${N_SHARDS}" ]; then
    echo "A run needs N_SHARDS set explicitly and a task below it, got '${N_SHARDS}' and ${SHARD}." >&2
    exit 1
  fi
fi

if [ "${STEP}" = "run" ] || [ "${STEP}" = "baselines" ] || [ "${STEP}" = "analyse" ]; then
  if [ "${A6_FROZEN}" != "1" ] || [ "${CONFIRM_GATES_PASSED}" != "1" ]; then
    echo "Runs wait on the freeze, G1 to G4 and the wait in the draft. Set A6_FROZEN=1 and CONFIRM_GATES_PASSED=1 only when they hold." >&2
    exit 1
  fi
fi

if { [ "${STEP}" = "run" ] || [ "${STEP}" = "analyse" ]; } && [ -z "${STAGE}" ]; then
  echo "STEP=${STEP} needs STAGE." >&2
  exit 1
fi
if { [ "${STEP}" = "run" ] || [ "${STEP}" = "baselines" ] || [ "${STEP}" = "mechanics" ]; } && [ -z "${MODEL}" ]; then
  echo "STEP=${STEP} needs MODEL." >&2
  exit 1
fi
if [ "${STEP}" = "analyse" ] && [ "${STAGE}" != "1" ] && [ -z "${MODEL}" ]; then
  echo "Stages 2 and 3 are analysed one model at a time, so MODEL is needed." >&2
  exit 1
fi
if [ "${MODEL}" = "molmo2" ]; then
  echo "Molmo2 fails G2 and G4, and has no route back without a dated rule." >&2
  exit 1
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi
source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs results/a6
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR=~/rds/hpc-work/study2/models
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore::UserWarning"

FLAGS=()
if [ "${A6_FROZEN}" = "1" ]; then
  FLAGS+=(--a6_frozen)
fi

case "${STEP}" in
  decoder|landmarks|au_manifest)
    python -m scripts.a6_run --step "${STEP}"
    ;;
  mechanics)
    python -m scripts.a6_run --step decoder
    python -m scripts.a6_run --step run --model "${MODEL}" --mechanics_only "${MECHANICS_ITEMS}"
    ;;
  baselines)
    python -m scripts.a6_run --step decoder
    python -m scripts.a6_run --step baselines --model "${MODEL}" "${FLAGS[@]}" --confirm_gates_passed
    ;;
  run)
    python -m scripts.a6_run --step decoder
    EXTRA=(--stage "${STAGE}" --shard "${SHARD}" --n_shards "${N_SHARDS}")
    if [ "${STAGE}" = "3" ]; then
      EXTRA+=(--stage2_output "results/a6/a6_stage2_${MODEL}.json")
    fi
    python -m scripts.a6_run --step run --model "${MODEL}" "${FLAGS[@]}" --confirm_gates_passed "${EXTRA[@]}"
    ;;
  analyse)
    EXTRA=(--stage "${STAGE}")
    if [ -n "${MODEL}" ]; then
      EXTRA+=(--model "${MODEL}")
    fi
    python -m scripts.a6_analyse "${FLAGS[@]}" "${EXTRA[@]}"
    ;;
esac

echo "Done: ${STEP} ${MODEL} ${STAGE}"
