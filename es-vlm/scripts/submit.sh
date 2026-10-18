#!/usr/bin/env bash
# Wrapper around sbatch. Reads local/site.sh. Refuses over-cap memory.
# Never call sbatch directly from docs or habit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="${ROOT}/local/site.sh"
if [ ! -f "${SITE}" ]; then
  echo "ERROR: missing ${SITE}. Copy local/site.example.sh and fill values." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${SITE}"

usage() {
  echo "Usage: $0 --script slurm/gpu1.sbatch [--mem-per-cpu N] [--partition PARTITION] [sbatch args...]" >&2
}

SCRIPT=""
MEM_PER_CPU=""
PARTITION=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --script)
      SCRIPT="$2"
      shift 2
      ;;
    --mem-per-cpu)
      MEM_PER_CPU="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [ -z "${SCRIPT}" ]; then
  usage
  exit 2
fi

PART="${PARTITION:-${HPC_GPU_PARTITION}}"
MAX_MB="${HPC_GPU_MEM_PER_CPU_MAX_MB:-8000}"
ACCOUNT="${HPC_GPU_ACCOUNT}"
case "${PART}" in
  "${HPC_CPU_PARTITION}")
    MAX_MB="${HPC_CPU_MEM_PER_CPU_MAX_MB:-3370}"
    ACCOUNT="${HPC_CPU_ACCOUNT}"
    ;;
esac

if [ -n "${MEM_PER_CPU}" ]; then
  MEM_NUM="${MEM_PER_CPU%M}"
  MEM_NUM="${MEM_NUM%m}"
  if [ "${MEM_NUM}" -gt "${MAX_MB}" ]; then
    echo "ERROR: --mem-per-cpu ${MEM_PER_CPU} exceeds cap ${MAX_MB}M on ${PART}." >&2
    echo "SLURM would silently inflate the CPU count." >&2
    exit 2
  fi
fi

SBATCH_ARGS=(
  -A "${ACCOUNT}"
  -p "${PART}"
)
if [ -n "${MEM_PER_CPU}" ]; then
  SBATCH_ARGS+=(--mem-per-cpu="${MEM_PER_CPU}")
fi

export ESVLM_ROOT="${ESVLM_ROOT:-${HPC_ROOT}}"
echo "submit: ESVLM_ROOT=${ESVLM_ROOT} sbatch ${SBATCH_ARGS[*]} ${EXTRA[*]} ${SCRIPT}"
sbatch "${SBATCH_ARGS[@]}" "${EXTRA[@]}" "${SCRIPT}"
