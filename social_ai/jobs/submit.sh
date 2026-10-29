#!/bin/bash
# Submit a slurm script with the account and partition from local/site.sh.
# Usage: jobs/submit.sh cpu|gpu path/to/job.sh [extra sbatch args]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
CLASS="${1:?cpu or gpu}"
shift
# Callers that must backfill on other nodes pass --nice=0 after this default.
NICE="${SITE_NICE:-10000}"

batch_qos() {
  local requested="$1"
  local fallback="$2"
  if [[ -z "${requested}" ]]; then
    printf '%s\n' "${fallback}"
    return
  fi
  case "${requested}" in
    intr|como_intr)
      echo "refusing the interactive service class" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "${requested}"
}

# Nodes reserved for this user's interactive jobs. Empty if there are none.
intr_node_list() {
  local id line node listed=""
  local ids
  ids="$(squeue -h -u "${USER}" -o "%A %q" | awk '$2=="intr" {print $1}')"
  [[ -z "${ids}" ]] && return 0
  for id in ${ids}; do
    line="$(timeout 20 scontrol -o show job "${id}" 2>/dev/null || true)"
    if [[ -z "${line}" ]]; then
      echo "interactive job ${id} has no readable node" >&2
      return 1
    fi
    while IFS= read -r node; do
      [[ -z "${node}" || "${node}" == "(null)" ]] && continue
      listed="${listed},${node}"
    done < <(printf '%s\n' "${line}" | tr ' ' '\n' | sed -n 's/^NodeList=//p; s/^SchedNodeList=//p')
  done
  listed="$(printf '%s\n' "${listed}" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)"
  if [[ -z "${listed}" ]]; then
    echo "an interactive job is queued but its node is not assigned yet" >&2
    return 1
  fi
  printf '%s\n' "${listed}"
}

case "${CLASS}" in
  cpu)
    ACCOUNT="${SITE_CPU_ACCOUNT}"
    PARTITION="${SITE_CPU_PARTITION}"
    MEM="${SITE_CPU_MEM_PER_CPU}"
    QOS="$(batch_qos "${SITE_CPU_QOS:-}" cpu2)"
    args=(-A "${ACCOUNT}" -p "${PARTITION}" --qos="${QOS}" --mem-per-cpu="${MEM}" --nice="${NICE}")
    ;;
  gpu)
    ACCOUNT="${SITE_GPU_ACCOUNT}"
    PARTITION="${SITE_GPU_PARTITION}"
    QOS="$(batch_qos "${SITE_GPU_QOS:-}" gpu2)"
    # One GPU and a fixed memory total, so the job stays on one node.
    # Never the interactive service class, and never that job's node.
    args=(-A "${ACCOUNT}" -p "${PARTITION}" --qos="${QOS}" --nodes=1 --gres=gpu:1 --mem=64G --nice="${NICE}")
    EXCLUDE="$(intr_node_list)"
    if [[ -n "${EXCLUDE}" ]]; then
      args+=(--exclude="${EXCLUDE}")
    fi
    ;;
  *)
    echo "first argument must be cpu or gpu" >&2
    exit 1
    ;;
esac
exec sbatch "${args[@]}" "$@"
