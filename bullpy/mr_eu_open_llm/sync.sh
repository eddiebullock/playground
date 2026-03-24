#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_USER="eb2007"
REMOTE_HOST="login.hpc.cam.ac.uk"
REMOTE_BASE="~/rds/hpc-work/study2"

usage() {
  echo "Usage: $0 [push|pull]"
  echo "  push: sync local scripts/configs to CSD3"
  echo "  pull: sync results from CSD3 back to local"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

CMD="$1"

case "$CMD" in
  push)
    echo "Pushing scripts and configs to CSD3..."
    rsync -av \
      --exclude '__pycache__' \
      --exclude '.ipynb_checkpoints' \
      --exclude '.git' \
      --exclude 'results/' \
      "${PROJECT_ROOT}/scripts/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/scripts/"

    rsync -av \
      "${PROJECT_ROOT}/config.py" \
      "${PROJECT_ROOT}/requirements.txt" \
      "${PROJECT_ROOT}/environment.yml" \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"
    ;;
  pull)
    echo "Pulling results from CSD3..."
    rsync -av \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/results/" \
      "${PROJECT_ROOT}/results/"
    ;;
  *)
    usage
    exit 1
    ;;
esac

