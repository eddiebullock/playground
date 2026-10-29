#!/bin/bash
# Copy this checkout onto the cluster. Run on the laptop.
# Keeps the Python data package. Does not copy participant files, reports, or local/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
REMOTE="${SITE_USER}@${SITE_HOST}:rds/hpc-work/social_ai/repo/"
ssh -o ControlMaster=no -o ControlPath="${SITE_CONTROL_PATH}" -o BatchMode=yes \
  "${SITE_USER}@${SITE_HOST}" "mkdir -p rds/hpc-work/social_ai/repo"
rsync -a \
  -e "ssh -o ControlMaster=no -o ControlPath=${SITE_CONTROL_PATH} -o BatchMode=yes" \
  --exclude '/data/' \
  --exclude '/reports/' \
  --exclude '/local/' \
  --exclude '.venv/' \
  --exclude 'node_modules/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.git/' \
  --exclude 'cue/' \
  "${ROOT}/" "${REMOTE}"
echo "code_pushed"
