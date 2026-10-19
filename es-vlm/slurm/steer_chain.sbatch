#!/usr/bin/env bash
#SBATCH -J esvlm_steer_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# PACK=first_causal (default): 8B layer 27 add-sweep, RMET then CLEVRER 128.
# Resume skips finished item_ids. Submit via scripts/submit.sh.

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

PACK="${PACK:-first_causal}"
LAYERS="${LAYERS:-27}"
ALPHAS="${ALPHAS:--2,-1,0,1,2}"
VECTORS="${VECTORS:-E,S,E_minus_S}"
MODES="${MODES:-add}"
SHARD="${SHARD:-0}"
FRAMES="${FRAMES:-8}"

case "${PACK}" in
  first_causal)
    # RMET all 36; CLEVRER first 128 of the already-scored 6000.
    JOBS=(
      "qwen3vl_8b:rmet:0:0"
      "qwen3vl_8b:clevrer:0:128"
    )
    ;;
  *)
    echo "ERROR: unknown PACK=${PACK}" >&2
    exit 2
    ;;
esac

resolve_models_dir() {
  local model="$1"
  local weights="${ESVLM_MODELS_DIR}/${model}"
  if [ ! -f "${weights}/config.json" ]; then
    case "${model}" in
      qwen3vl_8b) weights="${ESVLM_MODELS_DIR}/qwen3vl" ;;
      gemma4_e4b) weights="${ESVLM_MODELS_DIR}/gemma4" ;;
      qwen2vl_7b) weights="${ESVLM_MODELS_DIR}/qwen2vl" ;;
    esac
  fi
  if [ -f "${weights}/config.json" ]; then
    printf '%s' "${ESVLM_MODELS_DIR}"
    return 0
  fi
  if [ -f "${ESVLM_NEW_MODELS_DIR}/${model}/config.json" ]; then
    printf '%s' "${ESVLM_NEW_MODELS_DIR}"
    return 0
  fi
  return 1
}

failed=0
for spec in "${JOBS[@]}"; do
  MODEL="${spec%%:*}"
  rest="${spec#*:}"
  TASK="${rest%%:*}"
  rest="${rest#*:}"
  OFFSET="${rest%%:*}"
  LIMIT="${rest#*:}"
  if ! MODELS_DIR_ARG="$(resolve_models_dir "${MODEL}")"; then
    echo "SKIP missing weights ${MODEL}" >&2
    failed=$((failed + 1))
    continue
  fi
  DIR="outputs/extract_${MODEL}_${SHARD}"
  if [ ! -f "${DIR}/extract_summary.json" ]; then
    echo "SKIP missing directions ${MODEL}" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "===== steer ${MODEL} ${TASK} layers=${LAYERS} ====="
  set +e
  python -m esvlm.cli.steer \
    --model "${MODEL}" \
    --task "${TASK}" \
    --layers "${LAYERS}" \
    --alphas="${ALPHAS}" \
    --vectors "${VECTORS}" \
    --modes "${MODES}" \
    --directions "${DIR}" \
    --out "outputs/steer_${MODEL}_${TASK}_${SHARD}" \
    --trial_offset "${OFFSET}" \
    --trial_limit "${LIMIT}" \
    --frames "${FRAMES}" \
    --offline \
    --models-dir "${MODELS_DIR_ARG}"
  rc=$?
  set -e
  echo "steer_exit model=${MODEL} task=${TASK} rc=${rc}"
  if [ "${rc}" -ne 0 ]; then
    failed=$((failed + 1))
  fi
done
echo "steer_chain_done pack=${PACK} failed=${failed}"
if [ "${failed}" -gt 0 ]; then
  exit 1
fi
