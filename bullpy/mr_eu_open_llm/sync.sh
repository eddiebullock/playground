#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_USER="${REMOTE_USER:-eb2007}"
REMOTE_HOST="${REMOTE_HOST:-login.hpc.cam.ac.uk}"
REMOTE_BASE="${REMOTE_BASE:-~/rds/hpc-work/study2}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# One rsync = one SSH login (password/TOTP once per push or pull).
RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude 'results/'
  --exclude '__pycache__/'
  --exclude '.ipynb_checkpoints/'
  --exclude 'models/'
  --exclude 'data/mindreading/'
  --exclude 'data/eu_emotions_118/'
  --exclude 'data/cache/'
  --exclude 'conda_envs/'
  --exclude 'hf_cache/'
  --exclude '.venv/'
  --exclude 'venv/'
)

usage() {
  echo "Usage: $0 [push|pull|pull-artifact]"
  echo "  push: sync local code/configs to CSD3 (one SSH session)"
  echo "  pull: sync all results/ from CSD3 (slow; includes finetune checkpoints)"
  echo "  pull-artifact: sync only eval JSONs, stats, ablation (Workstream A)"
  echo ""
  echo "Tip: ssh-copy-id ${REMOTE}  # avoid passwords entirely"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

CMD="$1"

case "${CMD}" in
  push)
    echo "Pushing to CSD3 (${REMOTE}:${REMOTE_BASE})..."
    rsync -av "${RSYNC_EXCLUDES[@]}" \
      "${PROJECT_ROOT}/" "${REMOTE}:${REMOTE_BASE}/"
    echo "Push complete."
    ;;
  pull)
    echo "Pulling results from CSD3..."
    rsync -av \
      "${REMOTE}:${REMOTE_BASE}/results/" \
      "${PROJECT_ROOT}/results/"
    echo "Pull complete."
    ;;
  pull-artifact)
    echo "Pulling artifact results only (eval JSONs, stats, ablation)..."
    rsync -av \
      --include 'baseline/' \
      --include 'baseline/**/' \
      --include 'baseline/**/eval_v2_*.json' \
      --include 'baseline/**/eval_artifact_*.json' \
      --include 'baseline/**/*.csv' \
      --include 'finetune/' \
      --include 'finetune/eu_post_ft/' \
      --include 'finetune/eu_post_ft/eval_*.json' \
      --include 'finetune/eu_post_ft/eval_artifact_*.json' \
      --include 'stats/' \
      --include 'stats/**' \
      --include 'ablation/' \
      --include 'ablation/**' \
      --exclude '*' \
      "${REMOTE}:${REMOTE_BASE}/results/" \
      "${PROJECT_ROOT}/results/"
    echo "Pull-artifact complete."
    ;;
  *)
    usage
    exit 1
    ;;
esac
