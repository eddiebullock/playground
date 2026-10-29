#!/bin/bash
# Shared job preamble. Source this after local/site.sh.
# Never write to the real home directory. Caches go under the project root on RDS.
set -euo pipefail

if [[ -z "${SITE_PROJECT_ROOT:-}" ]]; then
  echo "SITE_PROJECT_ROOT is unset. Source local/site.sh first." >&2
  exit 1
fi

export REAL_HOME="${HOME}"
export HOME="${SITE_PROJECT_ROOT}/home"
mkdir -p \
  "${HOME}" \
  "${SITE_PROJECT_ROOT}/cache/conda-pkgs" \
  "${SITE_PROJECT_ROOT}/cache/pip" \
  "${SITE_PROJECT_ROOT}/cache/xdg" \
  "${SITE_PROJECT_ROOT}/cache/hf" \
  "${SITE_PROJECT_ROOT}/cache/torch" \
  "${SITE_PROJECT_ROOT}/cache/mpl"
export CONDA_PKGS_DIRS="${SITE_PROJECT_ROOT}/cache/conda-pkgs"
export CONDA_ENVS_DIRS="${SITE_PROJECT_ROOT}/conda/envs"
export PIP_CACHE_DIR="${SITE_PROJECT_ROOT}/cache/pip"
export XDG_CACHE_HOME="${SITE_PROJECT_ROOT}/cache/xdg"
export XDG_CONFIG_HOME="${SITE_PROJECT_ROOT}/cache/xdg"
export MPLCONFIGDIR="${SITE_PROJECT_ROOT}/cache/mpl"
export HF_HOME="${SITE_PROJECT_ROOT}/cache/hf"
export TORCH_HOME="${SITE_PROJECT_ROOT}/cache/torch"
export PYTHONNOUSERSITE=1
unset VIRTUAL_ENV || true

if [[ "${SCOG_EVAL_JOB:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

echo "job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local} account=${SLURM_JOB_ACCOUNT:-unset} partition=${SLURM_JOB_PARTITION:-unset} cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) local=$(date +%Y-%m-%dT%H:%M:%S%z)"
