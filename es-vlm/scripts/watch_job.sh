#!/usr/bin/env bash
# Poll one SLURM job over the ControlMaster socket. Resubmit PACK=e_panel
# on TIMEOUT/FAILED. Does not open an SSH socket.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"

JOB="${1:?usage: watch_job.sh JOBID}"
REMOTE="${HPC_USER}@${HPC_HOST}"
SSH=(ssh
  -o "ControlPath=${HPC_SSH_CONTROL_PATH}"
  -o "ControlMaster=no"
  -o "BatchMode=yes"
  -o "ConnectTimeout=20"
)
LOG="${2:-/tmp/esvlm/watch_job.log}"
mkdir -p "$(dirname "${LOG}")"

state_of() {
  "${SSH[@]}" "${REMOTE}" "sacct -j '$1' -n -X -o State" 2>/dev/null | awk '{print $1; exit}'
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watching ${JOB}" | tee -a "${LOG}"
while true; do
  if ! "${SSH[@]}" "${REMOTE}" true >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ssh_down" | tee -a "${LOG}"
    sleep 30
    continue
  fi
  st="$(state_of "${JOB}")"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${JOB} ${st:-unknown}" | tee -a "${LOG}"
  case "${st}" in
    COMPLETED)
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done ${JOB}" | tee -a "${LOG}"
      exit 0
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY)
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${st}; resubmitting e_panel" | tee -a "${LOG}"
      "${SSH[@]}" "${REMOTE}" "cd '${HPC_ROOT}' && export PACK=e_panel MODE=baseline SHARD=0 && scripts/submit.sh --script slurm/chain_baselines.sbatch -J esvlm_e_panel -q intr --time=00:59:00 --cpus-per-task=32 --mem=250G" | tee -a "${LOG}"
      exit 0
      ;;
    PENDING|RUNNING|"")
      sleep 90
      ;;
    *)
      sleep 90
      ;;
  esac
done
