#!/bin/bash
# Submit the prepare job, the base MMLU score, and the six exploratory chains.
# Run this on the cluster, from a checkout that already has local/site.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
export A2_ROOT="${SITE_PROJECT_ROOT}/a2"
export A2_DATA_DIR="${A2_ROOT}/data"
export A2_MODEL="${A2_ROOT}/models/Qwen3-8B-Base"
export A2_RUNS="${A2_ROOT}/runs"
export A2_MAX_MINUTES=25
mkdir -p "${SITE_PROJECT_ROOT}/logs" "${A2_RUNS}"

if [[ -n "${A2_PREP_JOB:-}" ]]; then
  PREP="${A2_PREP_JOB}"
else
  PREP="$("${ROOT}/jobs/submit.sh" cpu \
    --nice=0 \
    --job-name="a2prep" \
    --cpus-per-task=8 \
    --output="${SITE_PROJECT_ROOT}/logs/%x_%j.out" \
    --parsable \
    "${ROOT}/jobs/prepare_a2.sh")"
fi
echo "prepare_job ${PREP}"

MMLU="$("${ROOT}/jobs/submit.sh" gpu \
  --nice=0 \
  --job-name="a2mmlu" \
  --dependency="afterok:${PREP}" \
  --output="${SITE_PROJECT_ROOT}/logs/%x_%j.out" \
  --export="ALL,A2_MMLU_TAG=before" \
  --parsable \
  "${ROOT}/jobs/mmlu_a2.sh")"
echo "mmlu_before_job ${MMLU}"

while read -r run_id rank lr epochs chunks split; do
  export A2_RUN_ID="${run_id}"
  export A2_RUN_DIR="${A2_RUNS}/${run_id}"
  export A2_RANK="${rank}"
  export A2_LR="${lr}"
  export A2_EPOCHS="${epochs}"
  export A2_MAX_CHUNK="${chunks}"
  export A2_SPLIT="${split}"
  export A2_CHUNK=1
  job="$("${ROOT}/jobs/submit.sh" gpu \
    --nice=0 \
    --job-name="a2${run_id}" \
    --dependency="afterok:${PREP}" \
    --output="${SITE_PROJECT_ROOT}/logs/%x_%j.out" \
    --export="ALL" \
    --parsable \
    "${ROOT}/jobs/train_a2.sh")"
  echo "train_job ${run_id} ${job}"
done < <(python3 - "${ROOT}/configs/a2_grid.json" << 'PY'
import json, sys
from pathlib import Path
grid = json.loads(Path(sys.argv[1]).read_text())
chunks = grid["max_chunks_exploratory"]
for cell in grid["grid"]:
    print(cell["run_id"], cell["rank"], cell["lr"], cell["epochs"], chunks, cell["split"])
PY
)
echo "grid_submitted"
