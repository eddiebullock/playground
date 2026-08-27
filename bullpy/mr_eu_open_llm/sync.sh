#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_USER="${REMOTE_USER:-eb2007}"
REMOTE_HOST="${REMOTE_HOST:-login.hpc.cam.ac.uk}"
# Absolute path: quoted '~' breaks remote mkdir; study3 is the active experiment root.
# Override: REMOTE_BASE=/home/eb2007/rds/hpc-work/study2 ./sync.sh push
REMOTE_BASE="${REMOTE_BASE:-/home/${REMOTE_USER}/rds/hpc-work/study3}"
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
  --exclude 'data/eu_emotions/'
  --exclude 'data/cache/'
  --exclude 'conda_envs/'
  --exclude 'hf_cache/'
  --exclude '.venv/'
  --exclude 'venv/'
  --exclude 'venvs/'
)

usage() {
  echo "Usage: $0 [push|pull|pull-artifact|pull-study3]"
  echo "  push: sync local code/configs to CSD3 (one SSH session)"
  echo "  pull: sync all results/ from CSD3 (slow; includes finetune checkpoints)"
  echo "  pull-artifact: sync only eval JSONs, stats, ablation (Workstream A)"
  echo "  pull-study3: sync probes, RSA, patching, SAE, eval summaries (no checkpoints)"
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
  pull-study3)
    echo "Pulling Study 3 summary results (no finetune checkpoints / activation npy)..."
    rsync -av \
      --include 'baseline/' \
      --include 'baseline/eu_emotions/' \
      --include 'baseline/eu_emotions/*/' \
      --include 'baseline/eu_emotions/*/*.json' \
      --include 'baseline/eu_emotions/*/*.csv' \
      --include 'finetune/eu_post_ft/' \
      --include 'finetune/eu_post_ft/*.json' \
      --include 'finetune/eu_post_ft/*.csv' \
      --include 'activations/' \
      --include 'activations/*/' \
      --include 'activations/*/*/' \
      --include 'activations/*/*/extract_meta.json' \
      --include 'activations/*/*/layer*_trial_ids.json' \
      --include 'probes/' \
      --include 'probes/**' \
      --include 'rsa/' \
      --include 'rsa/**' \
      --include 'patching/' \
      --include 'patching/**' \
      --include 'sae/' \
      --include 'sae/**' \
      --include 'mech/' \
      --include 'mech/**' \
      --include 'stats/' \
      --include 'stats/**' \
      --exclude '*' \
      "${REMOTE}:${REMOTE_BASE}/results/" \
      "${PROJECT_ROOT}/results/"
    echo "Pull-study3 complete."
    ;;
  *)
    usage
    exit 1
    ;;
esac
