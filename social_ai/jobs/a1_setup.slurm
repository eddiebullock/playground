#!/bin/bash
#SBATCH --job-name=a1-setup
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
# Open weights only. Downloads the public Qwen3-8B checkpoint. No hosted API.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"

free_gb="$(quota -s | tr '\r' '\n' | awk '$3+0 > 100 && $3+0 < 5000 {free=$3-$2} END {if (free=="") exit 1; printf "%.1f", free}')"
echo "free_gb=${free_gb}"
python3 - << PY
free = float("${free_gb}")
if free < 40:
    raise SystemExit("free project space is %.1f GB; refusing the 8B download" % free)
print("free project space %.1f GB" % free)
PY

PREFIX="${SITE_PROJECT_ROOT}/conda"
if [[ ! -x "${PREFIX}/bin/conda" ]]; then
  INSTALLER="${SITE_PROJECT_ROOT}/cache/miniforge.sh"
  mkdir -p "${SITE_PROJECT_ROOT}/cache"
  curl -fsSL -o "${INSTALLER}" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "${INSTALLER}" -b -p "${PREFIX}"
fi
# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
if [[ ! -x "${SITE_PROJECT_ROOT}/conda/envs/scog/bin/python" ]]; then
  conda create -y -p "${SITE_PROJECT_ROOT}/conda/envs/scog" python=3.11
fi
conda activate "${SITE_PROJECT_ROOT}/conda/envs/scog"
pip install -r "${ROOT}/requirements.lock"
pip install "vllm==0.8.5" "huggingface_hub==0.31.1"

WEIGHTS="${SITE_PROJECT_ROOT}/weights/Qwen3-8B"
if [[ ! -f "${WEIGHTS}/config.json" ]]; then
  mkdir -p "${SITE_PROJECT_ROOT}/weights"
  huggingface-cli download Qwen/Qwen3-8B --local-dir "${WEIGHTS}"
fi
if [[ ! -f "${WEIGHTS}/config.json" ]]; then
  echo "Qwen3-8B download did not produce a config" >&2
  exit 1
fi

export SCOG_C4_PATH="${SITE_PROJECT_ROOT}/data/data_c4_raw.csv"
export SCOG_SPLIT_LOCK="${ROOT}/configs/splits.lock"
export SCOG_MANIFEST="${SITE_PROJECT_ROOT}/runs/a1/explore_manifest.jsonl"
export SCOG_N_PER_GROUP=200
export SCOG_N_CONDITION1=20
export SCOG_N_CONDITION4=20
export PYTHONPATH="${ROOT}/src"
python -m scog.a1.prepare
echo "setup done"
