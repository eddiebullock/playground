#!/usr/bin/env bash
#SBATCH -J esvlm_persona_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# PACK=confirmatory (default): 8B + 2B + Gemma + text Qwen3.
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

PACK="${PACK:-confirmatory}"
LAYERS="${LAYERS:-all}"
SHARD="${SHARD:-0}"
N_PERSONAS="${N_PERSONAS:-500}"

case "${PACK}" in
  confirmatory)
    MODELS=(qwen3vl_8b qwen3vl_2b gemma4_e4b qwen3_text)
    ;;
  core)
    MODELS=(qwen3vl_2b gemma4_e4b qwen3_text)
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
for MODEL in "${MODELS[@]}"; do
  if [ -z "${FORCE:-}" ] && [ -f "outputs/personas_${MODEL}_${SHARD}/persona_summary.json" ]; then
    echo "SKIP already scored personas ${MODEL}"
    continue
  fi
  if [ ! -f "outputs/extract_${MODEL}_${SHARD}/extract_summary.json" ]; then
    echo "SKIP missing directions ${MODEL}" >&2
    failed=$((failed + 1))
    continue
  fi
  if ! MODELS_DIR_ARG="$(resolve_models_dir "${MODEL}")"; then
    echo "SKIP missing weights ${MODEL}" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "===== personas ${MODEL} ====="
  set +e
  python -m esvlm.cli.extract_personas \
    --model "${MODEL}" \
    --directions "outputs/extract_${MODEL}_${SHARD}" \
    --out "outputs/personas_${MODEL}_${SHARD}" \
    --n "${N_PERSONAS}" \
    --layers "${LAYERS}" \
    --offline \
    --models-dir "${MODELS_DIR_ARG}"
  rc=$?
  set -e
  echo "persona_exit model=${MODEL} rc=${rc}"
  if [ "${rc}" -ne 0 ]; then
    failed=$((failed + 1))
  fi
done
echo "persona_chain_done pack=${PACK} failed=${failed}"
if [ "${failed}" -gt 0 ]; then
  exit 1
fi
