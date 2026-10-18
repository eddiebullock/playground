#!/usr/bin/env bash
#SBATCH -J esvlm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# Submit only via scripts/submit.sh (sets -A and -p).
# Example (after approval): MODEL=qwen3vl_8b TASK=synthetic MODE=smoke \
#   scripts/submit.sh --script slurm/gpu1.sbatch -J qwen3vl_8b_synthetic_smoke

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/scripts/activate_env.sh" "${ESVLM_PROFILE:-main}"

cd "${ESVLM_ROOT}"
mkdir -p logs outputs

MODEL="${MODEL:-qwen3vl_8b}"
TASK="${TASK:-synthetic}"
MODE="${MODE:-smoke}"
FRAMES="${FRAMES:-8}"
SHARD="${SHARD:-0}"

WEIGHTS="${ESVLM_MODELS_DIR}/${MODEL}"
if [ ! -f "${WEIGHTS}/config.json" ]; then
  case "${MODEL}" in
    qwen3vl_8b) WEIGHTS="${ESVLM_MODELS_DIR}/qwen3vl" ;;
    gemma4_e4b) WEIGHTS="${ESVLM_MODELS_DIR}/gemma4" ;;
    molmo2) WEIGHTS="${ESVLM_MODELS_DIR}/molmo2" ;;
    qwen2vl_7b) WEIGHTS="${ESVLM_MODELS_DIR}/qwen2vl" ;;
  esac
fi
if [ ! -f "${WEIGHTS}/config.json" ] && [ ! -f "${ESVLM_NEW_MODELS_DIR}/${MODEL}/config.json" ]; then
  echo "ERROR: missing weights for ${MODEL}. Fetch with slurm/fetch_weights.sbatch" >&2
  exit 2
fi

python -m esvlm.cli.score \
  --model "${MODEL}" \
  --task "${TASK}" \
  --out "outputs/${MODE}_${MODEL}_${TASK}_${SHARD}" \
  --frames "${FRAMES}" \
  --trial_offset "${TRIAL_OFFSET:-0}" \
  --trial_limit "${TRIAL_LIMIT:-0}"
