#!/usr/bin/env bash
# Activate env-main or env-molmo after unsetting any inherited venv/conda.
# Safe to source from a job after slurm/_preamble.sh. Bash only (login python is 3.6).
set -euo pipefail

ESVLM_PROFILE="${1:-${ESVLM_PROFILE:-main}}"
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi
if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_SHLVL:-0}" != "0" ]; then
  # Do not `conda deactivate` in a loop; just drop CONDA_PREFIX from PATH.
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${CONDA_PREFIX}/bin" | paste -sd: -)"
  export PATH
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
fi

ENV_DIR="${ESVLM_ROOT}/envs/${ESVLM_PROFILE}"
if [ ! -d "${ENV_DIR}" ]; then
  echo "ERROR: missing env profile ${ENV_DIR}" >&2
  exit 2
fi

if [ -x "${ENV_DIR}/.venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/.venv/bin/activate"
  return 0 2>/dev/null || exit 0
fi

echo "ERROR: ${ENV_DIR}/.venv not built. On the cluster, submit the CPU env-build job." >&2
exit 2
