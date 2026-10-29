#!/bin/bash
#SBATCH --job-name=scog-smoke
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
# One multiple-choice call and one seeded generation. Must finish inside 10 minutes.
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
export MODEL_PATH="${1:?path to a local model directory}"
export BASE_URL="${2:?openai-compatible base url}"
python - << 'PY'
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(":")[0])
from scog.infer.smoke import smoke_calls
calls = smoke_calls(os.environ["MODEL_PATH"])
print("smoke_plan", len(calls), sorted(c["item"] for c in calls))
print("base_url_set", bool(os.environ.get("BASE_URL")))
PY
echo "Smoke plan printed. Point BASE_URL at a running server before sending the calls."
