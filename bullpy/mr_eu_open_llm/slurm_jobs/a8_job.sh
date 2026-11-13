#!/usr/bin/env bash
#SBATCH -J a8
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/a8_%x_%A_%a.out
#SBATCH -e logs/a8_%x_%A_%a.err
#
# Amendment A8: every A8 job goes through research/scripts/a8_launch.py, which builds the
# frozen command from the design block, sets MR_EU_HPC_MODELS_DIR from the design, checks
# eligibility, data, weights and versions, and guards the frozen outputs. This file sets up,
# for each step, the same environment the frozen job of that step sets up for a non-Molmo
# model, line for line:
#   a1_shard   slurm_jobs/study3_rotation_eval.sh       conda; PYTHONPATH cv411:src
#   gates      slurm_jobs/study3_vision_gates.sh        conda; PYTHONPATH cv411:src:root
#   a3         slurm_jobs/study3_face_input_validity.sh conda; PYTHONPATH cv411:src:root; offline
#   delivery, a2_run  slurm_jobs/study3_stimulus_interventions.sh  au_tools venv alone
#   ladder     slurm_jobs/study3_readout_ladder.sh      au_tools venv alone
#   a4_run     slurm_jobs/study3_a4.sh                  conda; PYTHONPATH src:root; offline
# Differences, stated in the draft's Limits: MR_EU_HPC_MODELS_DIR is set by the launcher, not
# here; the rotation job's `--write_rotation_lookups` is not rerun (the launcher checks the
# lookups against their pins); the frozen code runs inside the launcher's process.
# GPU count is set with --gres at submission and is the same for every shard and for real and
# mismatch (design rule gpu_settings).
#
#   STEP=gates A8_ID=qwen3vl_32b sbatch slurm_jobs/a8_job.sh
#   STEP=a1_shard A8_ID=qwen3vl_32b ROTATION=0 CONDITION=mismatch N_SHARDS=4 sbatch --array=0-3 slurm_jobs/a8_job.sh
#   STEP=a2_run A8_ID=qwen3vl_32b N_SHARDS=12 sbatch --array=0-11 slurm_jobs/a8_job.sh
#   CPU steps (a1_merge, natural_rq1, a4_analyse) may be submitted with -A BARON-COHEN-SL3-CPU -p icelake --gres=none

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
STEP="${STEP:?set STEP}"
A8_ID="${A8_ID:-}"
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

case "${STEP}" in
  a1_shard|gates|a3|a4_run|a1_merge|natural_rq1|a4_analyse) PROFILE=conda ;;
  delivery|a2_run|ladder) PROFILE=au_tools ;;
  *) echo "Unknown STEP ${STEP}" >&2; exit 1 ;;
esac

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

if [ "${PROFILE}" = "conda" ]; then
  module load miniconda || module load miniconda3 || true
  export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
  export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  set -u
else
  source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"
fi

cd "${PROJECT_ROOT}"
mkdir -p logs
unset MR_EU_HPC_MODELS_DIR

case "${STEP}" in
  a1_shard)
    export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${PROJECT_ROOT}/hf_cache"
    export TOKENIZERS_PARALLELISM="false"
    ;;
  gates)
    export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${PROJECT_ROOT}/hf_cache"
    export TOKENIZERS_PARALLELISM="false"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    ;;
  a3)
    export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${PROJECT_ROOT}/hf_cache"
    export HF_HUB_OFFLINE=1
    export TOKENIZERS_PARALLELISM="false"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    ;;
  delivery|a2_run|ladder)
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${PROJECT_ROOT}/hf_cache"
    export HF_HUB_OFFLINE=1
    export TOKENIZERS_PARALLELISM="false"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export PYTHONWARNINGS="ignore::UserWarning"
    ;;
  a4_run|a1_merge|natural_rq1|a4_analyse)
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${PROJECT_ROOT}/hf_cache"
    export HF_HUB_OFFLINE=1
    export TOKENIZERS_PARALLELISM="false"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    ;;
esac

# A4's analysis runs with single-threaded BLAS (A4 pre-data ruling 7, DECISIONS 2026-11-12); the
# launcher refuses a4_analyse otherwise.
if [ "${STEP}" = "a4_analyse" ]; then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
fi

ARGS=(--step "${STEP}" --confirm_phase2_completed)
[ -n "${A8_ID}" ] && ARGS+=(--a8_id "${A8_ID}")
[ -n "${ROTATION:-}" ] && ARGS+=(--rotation "${ROTATION}")
[ -n "${CONDITION:-}" ] && ARGS+=(--condition "${CONDITION}")
[ -n "${N_SHARDS:-}" ] && ARGS+=(--n_shards "${N_SHARDS}")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "${STEP}" != "a1_merge" ]; then
  ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}")
fi

python research/scripts/a8_launch.py "${ARGS[@]}"
echo "Done."
