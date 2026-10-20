#!/usr/bin/env bash
#SBATCH -J esvlm_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# CPU partition. Paper 1/2 annotations only (NExT-QA CSVs, MMToM
# questions). Never the 257GB MMToM video zip. Submit via
# scripts/submit.sh --partition "$HPC_CPU_PARTITION".

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE=fetch
DATASET="${DATASET:-all}"
export DATASET
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/scripts/activate_env.sh" "${ESVLM_PROFILE:-main}"

cd "${ESVLM_ROOT}"
if [ -z "${ESVLM_PUBLIC_DIR:-}" ]; then
  echo "ERROR: ESVLM_PUBLIC_DIR is unset" >&2
  exit 2
fi
mkdir -p "${ESVLM_PUBLIC_DIR}" logs
python -m esvlm.cli.fetch_public --dataset "${DATASET}" --root "${ESVLM_PUBLIC_DIR}"
echo "ok dataset=${DATASET} root=${ESVLM_PUBLIC_DIR}"
