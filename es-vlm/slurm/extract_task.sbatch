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
# DESIGN=same_image_rmet | video_eu_clevrer | paper8b
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
PACK="${PACK:-paper8b}"
SHARD="${SHARD:-0}"
FRAMES="${FRAMES:-8}"
SOC_LIM="${TRIAL_LIMIT_SOCIAL:-0}"
MECH_LIM="${TRIAL_LIMIT_MECHANICAL:-256}"

case "${PACK}" in
  same_image_rmet) DESIGNS=(same_image_rmet) ;;
  video_eu_clevrer) DESIGNS=(video_eu_clevrer) ;;
  paper8b) DESIGNS=(same_image_rmet video_eu_clevrer) ;;
  *)
    echo "ERROR: unknown PACK=${PACK}" >&2
    exit 2
    ;;
esac

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
if [ ! -f "${WEIGHTS}/config.json" ]; then
  echo "ERROR: no config.json at ${WEIGHTS} or ${ESVLM_NEW_MODELS_DIR}/${MODEL}" >&2
  exit 2
fi
echo "weights=${WEIGHTS} models_dir=${MODELS_DIR_ARG}"

failed=0
for DESIGN in "${DESIGNS[@]}"; do
  echo "===== task extract ${MODEL} ${DESIGN} ====="
  set +e
  python -m esvlm.cli.extract_task \
    --model "${MODEL}" \
    --design "${DESIGN}" \
    --out "outputs/task_extract_${MODEL}_${DESIGN}_${SHARD}" \
    --trial_limit_social "${SOC_LIM}" \
    --trial_limit_mechanical "${MECH_LIM}" \
    --frames "${FRAMES}" \
    --offline \
    --models-dir "${MODELS_DIR_ARG}"
  rc=$?
  set -e
  echo "task_extract_exit design=${DESIGN} rc=${rc}"
  if [ "${rc}" -ne 0 ]; then
    failed=$((failed + 1))
  fi
done
echo "extract_task_done pack=${PACK} failed=${failed}"
if [ "${failed}" -gt 0 ]; then
  exit 1
fi
