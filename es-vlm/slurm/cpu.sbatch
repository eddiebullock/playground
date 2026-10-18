#!/usr/bin/env bash
#SBATCH -J esvlm_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# CPU env build / analysis. Submit via scripts/submit.sh --partition "$HPC_CPU_PARTITION".

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"

cd "${ESVLM_ROOT}"
mkdir -p logs envs/main envs/molmo

MODE="${MODE:-env}"
if [ "${MODE}" = "env" ] || [ "${MODE}" = "env-main" ] || [ "${MODE}" = "env-molmo" ]; then
  # Prefer a filesystem with >=8 GiB free. Shared /tmp on icelake is a
  # 16 GiB tmpfs and was 100% full on cpu-q-138 (job 38749877).
  pick_uv_cache() {
    local cand avail
    for cand in \
      ${SLURM_TMPDIR:+"${SLURM_TMPDIR}/esvlm-uv-cache"} \
      "/local/tmp/${USER}/esvlm-uv-cache" \
      "/local/${USER}/esvlm-uv-cache" \
      "${ESVLM_CACHE_ROOT:-${HPC_ROOT}/cache}/uv"
    do
      mkdir -p "${cand}" 2>/dev/null || continue
      avail="$(df -Pk "${cand}" 2>/dev/null | awk 'NR==2 {print $4}')"
      echo "scratch candidate ${cand} avail_kb=${avail:-none}" >&2
      if [ -n "${avail}" ] && [ "${avail}" -gt 8388608 ]; then
        printf '%s\n' "${cand}"
        return 0
      fi
    done
    return 1
  }
  echo "===== scratch probe ====="
  df -h /tmp /local /local/tmp "${ESVLM_CACHE_ROOT:-${HPC_ROOT}/cache}" 2>/dev/null || true
  LOCAL_UV="$(pick_uv_cache)" || {
    echo "ERROR: no scratch with 8 GiB free for uv" >&2
    exit 2
  }
  export UV_CACHE_DIR="${LOCAL_UV}"
  export UV_LINK_MODE=copy
  export UV_CONCURRENT_DOWNLOADS=1
  export UV_CONCURRENT_EXTRACTS=1
  export TMPDIR="${LOCAL_UV}"
  echo "uv cache -> ${UV_CACHE_DIR}"
  df -h "${UV_CACHE_DIR}" || true
fi
if [ "${MODE}" = "env" ] || [ "${MODE}" = "env-main" ] || [ "${MODE}" = "env-molmo" ]; then
  # System /usr/bin/python3.12 has no Python.h. Molmo/torch 2.14 then
  # fails compiling a Triton driver (job 38750974). Managed CPython
  # ships headers. Keep the interpreter off $HOME (home quota is tight).
  export UV_PYTHON_INSTALL_DIR="${ESVLM_VENV_ROOT:-${HPC_ROOT}}/uv-python"
  mkdir -p "${UV_PYTHON_INSTALL_DIR}"
  uv python install 3.12
fi
if [ "${MODE}" = "env" ] || [ "${MODE}" = "env-main" ]; then
  unset VIRTUAL_ENV || true
  VENV_MAIN="${ESVLM_VENV_ROOT:-${ESVLM_ROOT}/envs}/main/.venv"
  mkdir -p "$(dirname "${VENV_MAIN}")"
  export UV_PROJECT_ENVIRONMENT="${VENV_MAIN}"
  echo "uv env-main -> ${UV_PROJECT_ENVIRONMENT}"
  uv sync --python 3.12 --project envs/main --extra dev
fi
if [ "${MODE}" = "env" ] || [ "${MODE}" = "env-molmo" ]; then
  unset VIRTUAL_ENV || true
  VENV_MOLMO="${ESVLM_VENV_ROOT:-${ESVLM_ROOT}/envs}/molmo/.venv"
  mkdir -p "$(dirname "${VENV_MOLMO}")"
  export UV_PROJECT_ENVIRONMENT="${VENV_MOLMO}"
  echo "uv env-molmo -> ${UV_PROJECT_ENVIRONMENT}"
  rm -rf "${VENV_MOLMO}"
  uv sync --python 3.12 --project envs/molmo
fi
if [ "${MODE}" = "analysis" ]; then
  # shellcheck disable=SC1091
  source "${ESVLM_ROOT}/scripts/activate_env.sh" main
  python -c "import esvlm; print(esvlm.__version__)"
fi
