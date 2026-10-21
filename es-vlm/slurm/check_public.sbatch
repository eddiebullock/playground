#!/usr/bin/env bash
#SBATCH -J esvlm_check_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# CPU partition. Load Paper 1/2 public items and run unit tests.
# No GPU. No MMToM video zip. Submit via scripts/submit.sh
# --partition "$HPC_CPU_PARTITION".

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE=check
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/scripts/activate_env.sh" "${ESVLM_PROFILE:-main}"

cd "${ESVLM_ROOT}"
mkdir -p logs results/aggregate

if [ ! -f "${ESVLM_NEXTQA_MANIFEST:-}" ] && [ -f "${ESVLM_PUBLIC_DIR}/nextqa/pairs_val.json" ]; then
  export ESVLM_NEXTQA_MANIFEST="${ESVLM_PUBLIC_DIR}/nextqa/pairs_val.json"
  export ESVLM_NEXTQA_DIR="${ESVLM_PUBLIC_DIR}/nextqa/videos"
fi
if [ ! -f "${ESVLM_MMTOM_MANIFEST:-}" ] && [ -f "${ESVLM_PUBLIC_DIR}/mmtom/manifest.json" ]; then
  export ESVLM_MMTOM_MANIFEST="${ESVLM_PUBLIC_DIR}/mmtom/manifest.json"
fi
echo "nextqa_manifest=${ESVLM_NEXTQA_MANIFEST:-} mmtom_manifest=${ESVLM_MMTOM_MANIFEST:-} piqa_manifest=${ESVLM_PIQA_MANIFEST:-}"

python -m esvlm.cli.check_public --out results/aggregate/public_item_lock.json
python -m pytest tests/test_nextqa.py tests/test_mmtom.py tests/test_fetch_public.py tests/test_score_cli.py tests/test_extract_task.py -q
echo "ok check_public"
