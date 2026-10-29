#!/bin/bash
#SBATCH --job-name=a2mmlu
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --open-mode=append
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
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export A2_ROOT="${SITE_PROJECT_ROOT}/a2"
export A2_MODEL="${A2_ROOT}/models/Qwen3-8B-Base"
export A2_MMLU_TAG="${A2_MMLU_TAG:-before}"
python -m scog.train.mmlu
