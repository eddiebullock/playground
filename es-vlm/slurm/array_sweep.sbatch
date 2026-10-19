#!/usr/bin/env bash
#SBATCH -J esvlm_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# Chunked layer x alpha sweep. Confirm OSF prereg before submitting.
# Submit via scripts/submit.sh. Chain with --dependency=afterany.

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/scripts/activate_env.sh" "${ESVLM_PROFILE:-main}"

cd "${ESVLM_ROOT}"
mkdir -p logs outputs

if [ "${ESVLM_PREREG_POSTED:-0}" != "1" ]; then
  echo "ERROR: confirmatory sweep blocked until ESVLM_PREREG_POSTED=1" >&2
  exit 2
fi

python -m esvlm.cli.steer \
  --model "${MODEL}" \
  --task "${TASK}" \
  --layers "${LAYERS}" \
  --alphas "${ALPHAS}"
