#!/bin/bash
# Submit a slurm script with the account and partition from local/site.sh.
# Usage: jobs/submit.sh cpu|gpu path/to/job.sh [extra sbatch args]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
CLASS="${1:?cpu or gpu}"
shift
# Lower priority than interactive work already waiting on the same account.
NICE="${SITE_NICE:-10000}"
case "${CLASS}" in
  cpu)
    ACCOUNT="${SITE_CPU_ACCOUNT}"
    PARTITION="${SITE_CPU_PARTITION}"
    MEM="${SITE_CPU_MEM_PER_CPU}"
    args=(-A "${ACCOUNT}" -p "${PARTITION}" --mem-per-cpu="${MEM}" --nice="${NICE}")
    if [[ -n "${SITE_CPU_QOS:-}" ]]; then
      args+=(--qos="${SITE_CPU_QOS}")
    fi
    ;;
  gpu)
    ACCOUNT="${SITE_GPU_ACCOUNT}"
    PARTITION="${SITE_GPU_PARTITION}"
    # One GPU, one node. A per-CPU memory request at the CPUs bundled with a
    # GPU spills onto a second node. Do not use the interactive QOS.
    args=(-A "${ACCOUNT}" -p "${PARTITION}" --nodes=1 --gres=gpu:1 --mem=64G --nice="${NICE}")
    if [[ -n "${SITE_GPU_QOS:-}" ]]; then
      args+=(--qos="${SITE_GPU_QOS}")
    fi
    ;;
  *)
    echo "first argument must be cpu or gpu" >&2
    exit 1
    ;;
esac
exec sbatch "${args[@]}" "$@"
