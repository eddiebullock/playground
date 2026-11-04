#!/usr/bin/env bash
#SBATCH -J s3_stim_int
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_stim_int_%x_%j.out
#SBATCH -e logs/study3_stim_int_%x_%j.err
#
# Stimulus interventions for RQ1 and RQ2 (draft amendment A2). See
# scripts/run_stimulus_interventions.py. Frames are edited in memory; nothing here writes
# a frame, crop or edited clip.
#
#   Smoke:     MODEL=gemma4 MAX_ITEMS=3 sbatch --time=00:50:00 --qos=intr slurm_jobs/study3_stimulus_interventions.sh
#   Processor: STEP=probe_processor MODEL=qwen3vl sbatch --time=00:15:00 --qos=intr slurm_jobs/study3_stimulus_interventions.sh
#   Prompt:    STEP=probe_prompt MAX_ITEMS=3 MODEL=gemma4 sbatch --time=00:20:00 slurm_jobs/study3_stimulus_interventions.sh
#   Full:      not yet. Wait for G2 to G4 and RQ0, then STEP=landmarks once, then
#              CONFIRM_GATES_PASSED=1 STEP=run MODEL=... sbatch --array=0-3 ...
#
# Landmarks need MediaPipe, which lives in the au_tools overlay. That overlay sits on the
# shared study2 conda env, so it also runs Gemma 4 and Qwen and no new env is needed.
# Molmo2 needs its own venv, and reads the geometry written by the landmarks step.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-gemma4}"
STEP="${STEP:-smoke}"
MAX_ITEMS="${MAX_ITEMS:-}"
INPUTS="${INPUTS:-single_face,single_full,video_face,legacy_full}"
CONFIRM_GATES_PASSED="${CONFIRM_GATES_PASSED:-0}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-1}"
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
  N_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
fi

if { [ "${STEP}" = "run" ] || [ "${STEP}" = "smoke" ]; } && [ -z "${MAX_ITEMS}" ] && [ "${CONFIRM_GATES_PASSED}" != "1" ]; then
  echo "Full runs wait on G2 to G4 and RQ0. Set MAX_ITEMS for a smoke run." >&2
  exit 1
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

if [ "${MODEL}" = "molmo2" ]; then
  ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"
  module load miniconda || module load miniconda3 || true
  export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
  export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  source "${PROJECT_ROOT}/venvs/molmo2/bin/activate"
  set -u
else
  source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"
fi

cd "${PROJECT_ROOT}"
OUT_DIR="results/stimulus_interventions"
if [ -n "${MAX_ITEMS}" ]; then
  OUT_DIR="${OUT_DIR}/smoke"
fi
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore::UserWarning"  # protobuf warns once per frame

COMMON=(--model "${MODEL}" --geometry "${OUT_DIR}/geometry.json" --output_dir "${OUT_DIR}")
if [ -n "${MAX_ITEMS}" ]; then
  COMMON+=(--max_items "${MAX_ITEMS}")
fi

run_step() {
  python -m scripts.run_stimulus_interventions --step "$1" "${COMMON[@]}" "${@:2}"
}

case "${STEP}" in
  probe_processor)
    run_step probe_processor
    ;;
  landmarks)
    run_step landmarks
    ;;
  probe_prompt)
    run_step probe_prompt
    ;;
  delivery)
    python -m scripts.run_stimulus_interventions --step delivery --model "${MODEL}" \
      --geometry results/stimulus_interventions/geometry.json --max_items "${MAX_ITEMS:-3}"
    ;;
  smoke)
    run_step landmarks
    run_step run --inputs "${INPUTS}"
    ;;
  run)
    EXTRA=(--shard "${SHARD}" --n_shards "${N_SHARDS}")
    if [ "${CONFIRM_GATES_PASSED}" = "1" ]; then
      EXTRA+=(--confirm_gates_passed)
    fi
    run_step run --inputs "${INPUTS}" "${EXTRA[@]}"
    ;;
  *)
    echo "Unknown STEP: ${STEP}" >&2
    exit 1
    ;;
esac

echo "Done: ${STEP} ${MODEL}"
