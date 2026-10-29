#!/bin/bash
#SBATCH --job-name=scog-vllm
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@180
# Account, partition and memory are added by jobs/submit.sh gpu.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
export SCOG_EVAL_JOB=1
# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"
# shellcheck disable=SC1091
source "${SITE_PROJECT_ROOT}/conda/etc/profile.d/conda.sh"
conda activate "${SITE_PROJECT_ROOT}/conda/envs/scog"
MODEL_PATH="${1:?path to a local model directory}"
PORT="${2:-8000}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi
exec vllm serve "${MODEL_PATH}" --port "${PORT}" --max-model-len 4096 --dtype bfloat16
