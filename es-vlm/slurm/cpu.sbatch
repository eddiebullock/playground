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
if [ "${MODE}" = "env" ] || [ "${MODE}" = "env-main" ]; then
  unset VIRTUAL_ENV || true
  uv sync --project envs/main --extra dev
fi
if [ "${MODE}" = "env" ] || [ "${MODE}" = "env-molmo" ]; then
  unset VIRTUAL_ENV || true
  uv sync --project envs/molmo
fi
if [ "${MODE}" = "analysis" ]; then
  # shellcheck disable=SC1091
  source "${ESVLM_ROOT}/scripts/activate_env.sh" main
  python -c "import esvlm; print(esvlm.__version__)"
fi
