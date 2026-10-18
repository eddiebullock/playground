#!/usr/bin/env bash
# status | push | pull-results
# Reuses an existing ControlMaster socket. Does not open one.
# macOS rsync is 2.6.9: every option is --opt=value; globs are single-quoted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="${ROOT}/local/site.sh"
if [ ! -f "${SITE}" ]; then
  echo "ERROR: missing ${SITE}. Copy local/site.example.sh." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${SITE}"

REMOTE="${HPC_USER}@${HPC_HOST}"
SSH=(ssh
  -o "ControlPath=${HPC_SSH_CONTROL_PATH}"
  -o "ControlMaster=no"
  -o "BatchMode=yes"
)

rsync_ssh() {
  printf '%s ' "${SSH[@]}"
}

usage() {
  echo "Usage: $0 status|push|pull-results" >&2
}

need_socket() {
  if ! "${SSH[@]}" "${REMOTE}" 'true' >/dev/null 2>&1; then
    echo "ERROR: no live ControlMaster socket at ${HPC_SSH_CONTROL_PATH}." >&2
    echo "Open it by hand (password + TOTP), then retry." >&2
    exit 2
  fi
}

check_squeue() {
  echo "Would run: squeue -u ${HPC_USER} --name=esvlm --states=PENDING,RUNNING"
  echo "Ask before every cluster command. Not running squeue unless this script"
  echo "was invoked after an explicit yes for that check."
  if [ "${ESVLM_ALLOW_SQUEUE:-0}" != "1" ]; then
    echo "ERROR: set ESVLM_ALLOW_SQUEUE=1 only after the user approved squeue." >&2
    exit 3
  fi
  "${SSH[@]}" "${REMOTE}" "squeue -u '${HPC_USER}' --states=PENDING,RUNNING -h -o '%j %T'" \
    | { grep -E 'esvlm|qwen|gemma|molmo' && echo "ERROR: jobs from this project are queued or running. Not pushing." && exit 4 || true; }
}

write_version() {
  local dirty="CLEAN"
  if [ -n "$(git -C "${ROOT}/.." status --porcelain -- es-vlm)" ]; then
    dirty="DIRTY"
  fi
  {
    echo "commit=$(git -C "${ROOT}/.." rev-parse HEAD)"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "dirty=${dirty}"
  } > "${ROOT}/VERSION"
}

EXCLUDES=(
  --exclude='/.git/'
  --exclude='/data/'
  --exclude='/local/'
  --exclude='.env'
  --exclude='/models/'
  --exclude='/cache/'
  --exclude='/outputs/'
  --exclude='/.venv/'
  --exclude='/envs/main/.venv/'
  --exclude='/envs/molmo/.venv/'
  --exclude='/results/'
  --exclude='/logs/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='wandb/'
  --exclude='*.npy'
  --exclude='*.npz'
  --exclude='*.pt'
  --exclude='*.safetensors'
)

CMD="${1:-}"
case "${CMD}" in
  status)
    need_socket
    write_version
    rsync -an --out-format='%n' -e "$(rsync_ssh)" "${EXCLUDES[@]}" \
      "${ROOT}/" "${REMOTE}:${HPC_ROOT}/"
    ;;
  push)
    need_socket
    check_squeue
    write_version
    rsync -a --out-format='%n' -e "$(rsync_ssh)" "${EXCLUDES[@]}" \
      "${ROOT}/" "${REMOTE}:${HPC_ROOT}/"
    ;;
  pull-results)
    need_socket
    rsync -a --out-format='%n' -e "$(rsync_ssh)" \
      --include='*/' \
      --include='*.json' \
      --include='*.csv' \
      --include='*.parquet' \
      --include='*.png' \
      --include='*.pdf' \
      --include='*.svg' \
      --exclude='*' \
      "${REMOTE}:${HPC_ROOT}/results/aggregate/" \
      "${ROOT}/results/aggregate/"
    ;;
  *)
    usage
    exit 2
    ;;
esac
