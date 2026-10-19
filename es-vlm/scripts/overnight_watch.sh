#!/usr/bin/env bash
# Wait for a SLURM job, then submit PACK=e_panel on the GPU QOS.
# Reuses the ControlMaster socket. Does not open one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"

WAIT_JOB="${1:?usage: overnight_watch.sh JOBID}"
REMOTE="${HPC_USER}@${HPC_HOST}"
SSH=(ssh
  -o "ControlPath=${HPC_SSH_CONTROL_PATH}"
  -o "ControlMaster=no"
  -o "BatchMode=yes"
)

state_of() {
  "${SSH[@]}" "${REMOTE}" "sacct -j '$1' -n -X -o State" | awk '{print $1; exit}'
}

echo "watching ${WAIT_JOB}"
while true; do
  st="$(state_of "${WAIT_JOB}")"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${WAIT_JOB} ${st:-unknown}"
  case "${st}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
      break
      ;;
  esac
  sleep 90
done

if [ "${st}" != "COMPLETED" ]; then
  echo "ERROR: ${WAIT_JOB} ended ${st}; not submitting the chain" >&2
  exit 3
fi

echo "submitting PACK=e_panel"
"${SSH[@]}" "${REMOTE}" "cd '${HPC_ROOT}' && export PACK=e_panel MODE=baseline SHARD=0 && scripts/submit.sh --script slurm/chain_baselines.sbatch -J esvlm_e_panel -q intr --time=00:59:00 --cpus-per-task=32 --mem=250G"
