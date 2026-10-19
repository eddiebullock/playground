#!/usr/bin/env bash
#SBATCH -J esvlm_extract_task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# Task-conditioned social vs mechanical residuals.
# Submit via scripts/submit.sh.

set -euo pipefail
if [ -z "${ESVLM_ROOT:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/_preamble.sh" ]; then
    ESVLM_ROOT="${SLURM_SUBMIT_DIR}"
  else
    ESVLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/scripts/activate_env.sh" "${ESVLM_PROFILE:-main}"

cd "${ESVLM_ROOT}"
mkdir -p logs outputs

MODEL="${MODEL:-qwen3vl_8b}"
TASKS="${TASKS:-rmet,clevrer}"
SHARD="${SHARD:-0}"
FRAMES="${FRAMES:-8}"
SOC_LIM="${TRIAL_LIMIT_SOCIAL:-0}"
MECH_LIM="${TRIAL_LIMIT_MECHANICAL:-256}"

WEIGHTS="${ESVLM_MODELS_DIR}/${MODEL}"
if [ ! -f "${WEIGHTS}/config.json" ]; then
  case "${MODEL}" in
    qwen3vl_8b) WEIGHTS="${ESVLM_MODELS_DIR}/qwen3vl" ;;
    gemma4_e4b) WEIGHTS="${ESVLM_MODELS_DIR}/gemma4" ;;
    qwen2vl_7b) WEIGHTS="${ESVLM_MODELS_DIR}/qwen2vl" ;;
  esac
fi
MODELS_DIR_ARG="${ESVLM_MODELS_DIR:-}"
if [ ! -f "${WEIGHTS}/config.json" ] && [ -f "${ESVLM_NEW_MODELS_DIR}/${MODEL}/config.json" ]; then
  MODELS_DIR_ARG="${ESVLM_NEW_MODELS_DIR}"
fi

python -m esvlm.cli.extract_task \
  --model "${MODEL}" \
  --tasks "${TASKS}" \
  --out "outputs/task_extract_${MODEL}_${SHARD}" \
  --trial_limit_social "${SOC_LIM}" \
  --trial_limit_mechanical "${MECH_LIM}" \
  --frames "${FRAMES}" \
  --offline \
  --models-dir "${MODELS_DIR_ARG}"
