#!/bin/bash
#SBATCH --job-name=a2train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
# One 30-minute LoRA chunk. Submits the next chunk, then trains. Exit 3 means resume.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
: "${A2_RUN_DIR:?}" "${A2_CHUNK:?}" "${A2_MAX_CHUNK:?}" "${A2_SPLIT:?}"
: "${A2_RANK:?}" "${A2_LR:?}" "${A2_EPOCHS:?}" "${A2_RUN_ID:?}"
export A2_ROOT="${SITE_PROJECT_ROOT}/a2"
export A2_DATA_DIR="${A2_ROOT}/data"
export A2_MODEL="${A2_ROOT}/models/Qwen3-8B-Base"
export A2_RUNS="${A2_ROOT}/runs"
export A2_MAX_MINUTES="${A2_MAX_MINUTES:-25}"
mkdir -p "${SITE_PROJECT_ROOT}/logs" "${A2_RUN_DIR}"

if [[ -f "${A2_RUN_DIR}/COMPLETE" ]]; then
  echo "already_complete ${A2_RUN_ID}"
  exit 0
fi

NEXT_JOB=""
NEXT=$((A2_CHUNK + 1))
if (( NEXT <= A2_MAX_CHUNK )); then
  NEXT_JOB="$(A2_CHUNK="${NEXT}" "${ROOT}/jobs/submit.sh" gpu \
    --nice=0 \
    --job-name="a2${A2_RUN_ID}" \
    --dependency="afterany:${SLURM_JOB_ID}" \
    --output="${SITE_PROJECT_ROOT}/logs/%x_%j.out" \
    --export="ALL" \
    --parsable \
    "${ROOT}/jobs/train_a2.sh")"
  echo "next_chunk ${NEXT} job ${NEXT_JOB}"
fi

# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"
# shellcheck disable=SC1091
source "${SITE_PROJECT_ROOT}/conda/etc/profile.d/conda.sh"
conda activate "${SITE_PROJECT_ROOT}/conda/envs/scog"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c "from scog.train.prepare import quota_headroom; quota_headroom()"

# USR1 is delivered to this shell. Forward it and keep waiting so the trainer
# can write trainer.pt. If this shell dies on the default action, the chunk is lost.
set +e
export PYTHONUNBUFFERED=1
python -m scog.train.lora &
train_pid=$!
trap 'kill -USR1 "${train_pid}" 2>/dev/null || true' USR1
rc=0
while true; do
  wait "${train_pid}"
  rc=$?
  if ! kill -0 "${train_pid}" 2>/dev/null; then
    break
  fi
done
trap - USR1
set -e
echo "trainer_exit ${rc}"

stop_next() {
  if [[ -n "${NEXT_JOB}" ]]; then
    scancel "${NEXT_JOB}" || true
  fi
}

if [[ "${rc}" -eq 0 ]]; then
  stop_next
  set +e
  python -m scog.train.select
  sel=$?
  set -e
  if [[ "${sel}" -eq 0 ]] && mkdir "${A2_ROOT}/confirmatory.lock" 2>/dev/null; then
    export HOME="${REAL_HOME}"
    # shellcheck disable=SC1091
    source "${ROOT}/local/site.sh"
    python - << 'PY'
import json, os
from pathlib import Path
winner = json.loads(Path(os.environ["A2_RUNS"], "winner.json").read_text())
Path(os.environ["A2_ROOT"], "confirmatory.env").write_text(
    "A2_RANK={rank}\nA2_LR={lr}\nA2_EPOCHS={epoch}\n".format(**winner)
)
print("confirmatory_spec_written")
PY
    # shellcheck disable=SC1091
    source "${A2_ROOT}/confirmatory.env"
    export A2_SPLIT="confirmatory"
    export A2_CHUNK="1"
    export A2_MAX_CHUNK="200"
    export A2_RUN_ID="confirmatory"
    export A2_RUN_DIR="${A2_ROOT}/runs/confirmatory"
    if ! "${ROOT}/jobs/submit.sh" gpu \
      --nice=0 \
      --job-name="a2conf" \
      --output="${SITE_PROJECT_ROOT}/logs/%x_%j.out" \
      --export="ALL" \
      --parsable \
      "${ROOT}/jobs/train_a2.sh"; then
      rmdir "${A2_ROOT}/confirmatory.lock" || true
      exit 1
    fi
    echo "confirmatory_submitted"
  fi
  exit 0
fi

if [[ "${rc}" -eq 3 ]]; then
  exit 0
fi
stop_next
exit "${rc}"
