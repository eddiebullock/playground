#!/usr/bin/env bash
#SBATCH -J esvlm_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# One GPU hour, several unsteered baselines. Resume is per parquet:
# rerunning the same PACK skips finished item_ids.
# Submit via scripts/submit.sh. PACK=e_panel (default).

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

PACK="${PACK:-e_panel}"
MODE="${MODE:-baseline}"
SHARD="${SHARD:-0}"
FRAMES="${FRAMES:-8}"

# model:task pairs. No PhysBench, no more CLEVRER, no Mind Reading, no Molmo.
case "${PACK}" in
  e_panel)
    PAIRS=(
      "qwen3vl_2b:rmet"
      "gemma4_e4b:rmet"
      "qwen2vl_7b:rmet"
      "qwen25vl_7b:rmet"
      "qwen3vl_32b:rmet"
      "qwen3_text:rmet"
      "qwen3vl_2b:eu_emotions"
      "gemma4_e4b:eu_emotions"
    )
    ;;
  eu_rest)
    PAIRS=(
      "qwen2vl_7b:eu_emotions"
      "qwen25vl_7b:eu_emotions"
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

echo "pack=${PACK} n_pairs=${#PAIRS[@]}"
failed=0
for pair in "${PAIRS[@]}"; do
  MODEL="${pair%%:*}"
  TASK="${pair##*:}"
  if ! MODELS_DIR_ARG="$(resolve_models_dir "${MODEL}")"; then
    echo "SKIP missing weights ${MODEL}" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "===== ${MODEL} ${TASK} ====="
  set +e
  python -m esvlm.cli.score \
    --model "${MODEL}" \
    --task "${TASK}" \
    --out "outputs/${MODE}_${MODEL}_${TASK}_${SHARD}" \
    --frames "${FRAMES}" \
    --trial_offset 0 \
    --trial_limit 0 \
    --offline \
    --models-dir "${MODELS_DIR_ARG}"
  rc=$?
  set -e
  echo "pair_exit model=${MODEL} task=${TASK} rc=${rc}"
  if [ "${rc}" -ne 0 ]; then
    failed=$((failed + 1))
  fi
done

echo "chain_done pack=${PACK} failed=${failed}"
if [ "${failed}" -gt 0 ]; then
  exit 1
fi
