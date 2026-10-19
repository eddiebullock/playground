#!/usr/bin/env bash
#SBATCH -J esvlm_fetch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%j.out
#
# CPU partition only. New weights go to $HPC_ROOT/models, never another tree.
# Submit via scripts/submit.sh --partition "$HPC_CPU_PARTITION".

set -euo pipefail
ESVLM_ROOT="${ESVLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE=fetch
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/slurm/_preamble.sh"
# shellcheck disable=SC1091
source "${ESVLM_ROOT}/scripts/activate_env.sh" "${ESVLM_PROFILE:-main}"

cd "${ESVLM_ROOT}"
DEST="${ESVLM_NEW_MODELS_DIR}"
mkdir -p "${DEST}"

MODEL="${MODEL:-qwen3vl_2b}"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from esvlm.models.registry import get_spec

key = os.environ["MODEL"]
spec = get_spec(key)
dest = Path(os.environ["ESVLM_NEW_MODELS_DIR"]) / spec.key
print("fetch", spec.hf_id, spec.revision, "->", dest, flush=True)
snapshot_download(
    spec.hf_id,
    revision=spec.revision,
    local_dir=str(dest),
)
print("ok", dest / "config.json", (dest / "config.json").is_file(), flush=True)
PY
