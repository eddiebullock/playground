#!/usr/bin/env bash
# Sourced by every job. Bash only: login python is 3.6.
# Expects ESVLM_ROOT and local/site.sh already resolvable.
set -euo pipefail

if [ -z "${ESVLM_ROOT:-}" ]; then
  echo "ERROR: ESVLM_ROOT is unset" >&2
  exit 2
fi
SITE="${ESVLM_ROOT}/local/site.sh"
if [ ! -f "${SITE}" ]; then
  echo "ERROR: missing ${SITE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${SITE}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# Strip inherited cuda/11.4 from CPATH if present.
if [ -n "${CPATH:-}" ]; then
  NEW_CPATH=""
  OLDIFS="${IFS}"
  IFS=':'
  for part in ${CPATH}; do
    case "${part}" in
      *cuda/11.4*) ;;
      *)
        if [ -z "${NEW_CPATH}" ]; then
          NEW_CPATH="${part}"
        else
          NEW_CPATH="${NEW_CPATH}:${part}"
        fi
        ;;
    esac
  done
  IFS="${OLDIFS}"
  export CPATH="${NEW_CPATH}"
fi

CACHE="${ESVLM_CACHE_ROOT:-${HPC_ROOT}/cache}"
mkdir -p "${CACHE}/hf" "${CACHE}/torch" "${CACHE}/xdg" "${CACHE}/pip" "${CACHE}/uv" "${CACHE}/mpl"
export HF_HOME="${CACHE}/hf"
export HF_MODULES_CACHE="${CACHE}/hf/modules"
export TORCH_HOME="${CACHE}/torch"
export XDG_CACHE_HOME="${CACHE}/xdg"
export PIP_CACHE_DIR="${CACHE}/pip"
export UV_CACHE_DIR="${CACHE}/uv"
export MPLCONFIGDIR="${CACHE}/mpl"
export PYTHONNOUSERSITE=1

VERSION_FILE="${ESVLM_ROOT}/VERSION"
GIT_COMMIT="unknown"
if [ -f "${VERSION_FILE}" ]; then
  GIT_COMMIT="$(grep '^commit=' "${VERSION_FILE}" | cut -d= -f2)"
fi

echo "===== es-vlm job banner ====="
echo "job=${SLURM_JOB_ID:-local} name=${SLURM_JOB_NAME:-none}"
echo "node=${SLURMD_NODENAME:-$(hostname)} account=${SLURM_JOB_ACCOUNT:-unset} partition=${SLURM_JOB_PARTITION:-unset}"
echo "cpus=${SLURM_CPUS_ON_NODE:-unset}"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) local=$(date +%Y-%m-%dT%H:%M:%S)"
echo "commit=${GIT_COMMIT}"
echo "============================="

if [ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]; then
  set +e
  nvidia-smi
  set -e
fi

# Echo every run argument. A previous job silently ran smoke because MODE was ignored.
echo "args: MODEL=${MODEL:-} MODE=${MODE:-} TASK=${TASK:-} LAYER=${LAYER:-} LAYERS=${LAYERS:-} ALPHA=${ALPHA:-} ALPHAS=${ALPHAS:-} FRAMES=${FRAMES:-} N_SAMPLES=${N_SAMPLES:-} SHARD=${SHARD:-} VECTOR=${VECTOR:-} TRIAL_OFFSET=${TRIAL_OFFSET:-} TRIAL_LIMIT=${TRIAL_LIMIT:-}"

# Misspelled / lowercase variants of known vars.
for pair in model:MODEL mode:MODE layer:LAYER layers:LAYERS alpha:ALPHA alphas:ALPHAS frames:FRAMES n_samples:N_SAMPLES shard:SHARD task:TASK vector:VECTOR trial_offset:TRIAL_OFFSET trial_limit:TRIAL_LIMIT; do
  low="${pair%%:*}"
  up="${pair##*:}"
  # ${!low} is bash (not 3.6 python). Fine here.
  if [ -n "${!low:-}" ] && [ -z "${!up:-}" ]; then
    echo "ERROR: ${low} is set but ${up} is not. Refusing to guess." >&2
    exit 2
  fi
done

if [ "${ESVLM_OFFLINE:-1}" = "1" ] && [ "${MODE:-}" != "fetch" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
