#!/usr/bin/env bash
#SBATCH -J esvlm_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# Submit via scripts/submit.sh.
# MODEL=qwen3vl_8b LAYERS=all scripts/submit.sh --script slurm/extract.sbatch -J qwen3vl_8b_extract

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
LAYERS="${LAYERS:-all}"
SHARD="${SHARD:-0}"

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

ARGS=(
  --model "${MODEL}"
  --layers "${LAYERS}"
  --out "outputs/extract_${MODEL}_${SHARD}"
  --offline
)
if [ -n "${MODELS_DIR_ARG}" ]; then
  ARGS+=(--models-dir "${MODELS_DIR_ARG}")
fi
python -m esvlm.cli.extract "${ARGS[@]}"
