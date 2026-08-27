#!/usr/bin/env bash
# study4_rmet-only sync. Does NOT use or modify the repo-root sync.sh (study3).
set -euo pipefail

STUDY4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY4_ROOT}/.." && pwd)"
REMOTE_USER="${REMOTE_USER:-eb2007}"
REMOTE_HOST="${REMOTE_HOST:-login.hpc.cam.ac.uk}"
# Dedicated HPC tree — sibling of study3, never study3 itself.
REMOTE_BASE="${REMOTE_BASE:-/home/${REMOTE_USER}/rds/hpc-work/study4_rmet}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '.ipynb_checkpoints/'
  --exclude '.tmp_rmet/'
  --exclude 'venv/'
  --exclude '.venv/'
  --exclude 'venvs/'
  --exclude 'hf_cache/'
  --exclude 'conda_envs/'
  --exclude 'models/'
  # Fullpage debug renders contain 4AFC labels — do not push to HPC by default.
  --exclude 'data/rmet/_debug/'
)

usage() {
  echo "Usage: $0 [push|pull|push-repo-readonly]"
  echo "  push:              sync study4_rmet/ only -> ${REMOTE_BASE}/study4_rmet/"
  echo "  pull:              sync study4 results from HPC"
  echo "  push-repo-readonly: also mirror parent scripts/config for import-only reuse"
  echo "                      into ${REMOTE_BASE}/ (never writes back to study3)"
  echo ""
  echo "Isolation: this script never targets study3. Root sync.sh is untouched."
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

CMD="$1"

# Open a persistent master connection to the remote (one password/TOTP per run)
SSHCTL="ssh -o ControlMaster=auto -o ControlPath=/tmp/study4rmet-ssh-%r@%h:%p -o ControlPersist=600"

start_ssh_master() {
  $SSHCTL -N -f "${REMOTE}" || true
}

stop_ssh_master() {
  $SSHCTL -O exit "${REMOTE}" || true
}

case "${CMD}" in
  push)
    echo "Pushing study4_rmet/ -> ${REMOTE}:${REMOTE_BASE}/study4_rmet/"
    start_ssh_master
    $SSHCTL "${REMOTE}" "mkdir -p '${REMOTE_BASE}/study4_rmet'"
    rsync -av -e "$SSHCTL" "${RSYNC_EXCLUDES[@]}" \
      "${STUDY4_ROOT}/" "${REMOTE}:${REMOTE_BASE}/study4_rmet/"
    stop_ssh_master
    echo "Push complete."
    ;;
  push-repo-readonly)
    echo "Mirroring parent scripts/config (read-only reuse) + study4_rmet/ -> ${REMOTE}:${REMOTE_BASE}/"
    start_ssh_master
    $SSHCTL "${REMOTE}" "mkdir -p '${REMOTE_BASE}/study4_rmet' '${REMOTE_BASE}/scripts' '${REMOTE_BASE}/slurm_jobs'"
    # Parent loaders only — explicit paths, never results/ or study3 HPC tree.
    rsync -av -e "$SSHCTL" \
      --exclude '__pycache__/' \
      "${REPO_ROOT}/scripts/" "${REMOTE}:${REMOTE_BASE}/scripts/"
    rsync -av -e "$SSHCTL" "${REPO_ROOT}/config.py" "${REMOTE}:${REMOTE_BASE}/config.py"
    rsync -av -e "$SSHCTL" "${RSYNC_EXCLUDES[@]}" \
      "${STUDY4_ROOT}/" "${REMOTE}:${REMOTE_BASE}/study4_rmet/"
    stop_ssh_master
    echo "Push-repo-readonly complete. HPC project root: ${REMOTE_BASE}"
    echo "NOTE: this does not write to ~/rds/hpc-work/study3"
    ;;
  pull)
    echo "Pulling study4 results from ${REMOTE}:${REMOTE_BASE}/study4_rmet/results/"
    start_ssh_master
    mkdir -p "${STUDY4_ROOT}/results"
    rsync -av -e "$SSHCTL" \
      "${REMOTE}:${REMOTE_BASE}/study4_rmet/results/" \
      "${STUDY4_ROOT}/results/"
    stop_ssh_master
    echo "Pull complete."
    ;;
  *)
    usage
    exit 1
    ;;
esac
