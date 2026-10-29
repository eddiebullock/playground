#!/bin/bash
# Submit a slurm script with the account and partition from local/site.sh.
# Usage: jobs/submit.sh cpu|gpu path/to/job.sh [extra sbatch args]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
CLASS="${1:?cpu or gpu}"
shift
case "${CLASS}" in
  cpu)
    ACCOUNT="${SITE_CPU_ACCOUNT}"
    PARTITION="${SITE_CPU_PARTITION}"
    MEM="${SITE_CPU_MEM_PER_CPU}"
    ;;
  gpu)
    ACCOUNT="${SITE_GPU_ACCOUNT}"
    PARTITION="${SITE_GPU_PARTITION}"
    # A per-CPU memory request at the CPUs that come with one GPU spills onto
    # a second node. Ask for one node and a fixed memory total instead.
    exec sbatch -A "${ACCOUNT}" -p "${PARTITION}" --nodes=1 --mem=64G "$@"
    ;;
  *)
    echo "first argument must be cpu or gpu" >&2
    exit 1
    ;;
esac
exec sbatch -A "${ACCOUNT}" -p "${PARTITION}" --mem-per-cpu="${MEM}" "$@"
