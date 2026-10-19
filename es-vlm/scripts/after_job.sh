#!/usr/bin/env bash
# Wait for JOBID then run a remote command. ControlMaster only.
set -u
ROOT="${ESVLM_LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
JOB="${1:?jobid}"
shift
REMOTE="${HPC_USER}@${HPC_HOST}"
SSH=(ssh -o "ControlPath=${HPC_SSH_CONTROL_PATH}" -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=20)
LOG="${AFTER_LOG:-/tmp/esvlm/after_${JOB}.log}"
mkdir -p "$(dirname "${LOG}")"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) wait ${JOB}" | tee -a "${LOG}"
while true; do
  if ! "${SSH[@]}" "${REMOTE}" true >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ssh_down" | tee -a "${LOG}"
    sleep 30
    continue
  fi
  st="$("${SSH[@]}" "${REMOTE}" "sacct -j '${JOB}' -n -X -o State" 2>/dev/null | awk '{print $1; exit}')"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${JOB} ${st:-unknown}" | tee -a "${LOG}"
  case "${st}" in
    COMPLETED)
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run $*" | tee -a "${LOG}"
      "${SSH[@]}" "${REMOTE}" "$@" | tee -a "${LOG}"
      exit 0
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY)
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) stop ${st}" | tee -a "${LOG}"
      exit 3
      ;;
    *)
      sleep 90
      ;;
  esac
done
