#!/usr/bin/env bash
#SBATCH -J esvlm_sdomain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# CPU partition. Public S-domain media go to $ESVLM_PUBLIC_DIR, never
# into a licensed stimuli tree. Submit via scripts/submit.sh
# --partition "$HPC_CPU_PARTITION".

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE=fetch
DATASET="${DATASET:-piqa}"
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
ROOT="${ESVLM_PUBLIC_DIR}"
if [ "${DATASET}" = "clevrer" ] && [ -n "${ESVLM_CLEVRER_DIR:-}" ]; then
  mkdir -p "${ESVLM_CLEVRER_DIR}"
  ROOT="$(cd "$(dirname "${ESVLM_CLEVRER_DIR}")/.." && pwd)"
fi
mkdir -p "${ROOT}" logs
python -m esvlm.cli.fetch_sdomain --dataset "${DATASET}" --root "${ROOT}"
echo "ok dataset=${DATASET} root=${ROOT}"
