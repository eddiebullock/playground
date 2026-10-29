#!/bin/bash
#SBATCH --job-name=a2prep
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
# Download the base checkpoint, write the training JSONL, and freeze the MMLU ids.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
bash "${ROOT}/jobs/setup_train_env.sh"
# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"
# shellcheck disable=SC1091
source "${SITE_PROJECT_ROOT}/conda/etc/profile.d/conda.sh"
conda activate "${SITE_PROJECT_ROOT}/conda/envs/scog"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export C4_CSV="${SITE_PROJECT_ROOT}/data/data_c4_raw.csv"
export A2_ROOT="${SITE_PROJECT_ROOT}/a2"
export A2_MODEL="${A2_ROOT}/models/Qwen3-8B-Base"
export A2_MODEL_ID="Qwen/Qwen3-8B-Base"
mkdir -p "${A2_ROOT}"
if [[ ! -f "${C4_CSV}" ]]; then
  echo "Channel 4 file is missing" >&2
  exit 1
fi
IFS= read -r header < "${C4_CSV}"
if [[ "${header}" != userid,* ]]; then
  echo "Channel 4 header does not start with userid" >&2
  exit 1
fi
python -m scog.train.prepare
