#!/bin/bash
#SBATCH --job-name=a1-explore
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# One resumable chunk of the Qwen3-8B exploratory run. Local weights only.
set -euo pipefail
# sbatch copies this script into a spool directory, so $0 is not the repo.
ROOT="${SLURM_SUBMIT_DIR:?submit the job from the repo root}"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
export SCOG_EVAL_JOB=1
# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"
# shellcheck disable=SC1091
source "${SITE_PROJECT_ROOT}/conda/etc/profile.d/conda.sh"
conda activate "${SITE_PROJECT_ROOT}/conda/envs/scog"
export PYTHONPATH="${ROOT}/src"
export SCOG_MODEL_PATH="${SITE_PROJECT_ROOT}/weights/Qwen3-8B"
export SCOG_MANIFEST="${SITE_PROJECT_ROOT}/runs/a1/explore_manifest.jsonl"
export SCOG_JSONL="${SITE_PROJECT_ROOT}/runs/a1/explore.jsonl"
export SCOG_MAX_NEW="${SCOG_MAX_NEW:-30}"
python -m scog.a1.generate
echo "explore chunk done"
