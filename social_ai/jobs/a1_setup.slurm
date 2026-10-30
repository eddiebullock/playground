#!/bin/bash
#SBATCH --job-name=a1-setup
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
# Open weights only. Downloads the public Qwen3-8B checkpoint. No hosted API.
set -euo pipefail
# sbatch copies this script into a spool directory, so $0 is not the repo.
ROOT="${SLURM_SUBMIT_DIR:?submit the job from the repo root}"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"

free_gb="$(quota -s | tr '\r' '\n' | awk '$3+0 > 100 && $3+0 < 5000 {free=$3-$2} END {if (free=="") exit 1; printf "%.1f", free}')"
echo "free_gb=${free_gb}"
python3 - << PY
free = float("${free_gb}")
# One 16 GB checkpoint plus about 8 GB for this env, and a little left under the quota.
if free < 28:
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
# One setup at a time. Two jobs sharing the package cache delete each other's files.
mkdir -p "${SITE_PROJECT_ROOT}/cache"
exec 9>"${SITE_PROJECT_ROOT}/cache/a1-setup.lock"
flock 9
export CONDA_PKGS_DIRS="${SITE_PROJECT_ROOT}/cache/conda-pkgs-a1"
# Own env. Study A2 uses conda/envs/scog at Python 3.11, and the lock needs 3.12.
ENV="${SITE_PROJECT_ROOT}/conda/envs/scog-a1"
if ! "${ENV}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
  rm -rf "${ENV}"
  conda create -y -p "${ENV}" python=3.12
fi
conda activate "${ENV}"
if ! python -c 'import importlib.util, sys; ok = importlib.util.find_spec("vllm") and importlib.util.find_spec("pandas"); sys.exit(0 if ok else 1)'; then
  pip install --no-cache-dir -r "${ROOT}/requirements.lock"
  pip install --no-cache-dir "vllm==0.8.5" "huggingface_hub==0.31.1"
  rm -rf "${SITE_PROJECT_ROOT}/cache/conda-pkgs-a1"
fi

free_gb="$(quota -s | tr '\r' '\n' | awk '$3+0 > 100 && $3+0 < 5000 {free=$3-$2} END {if (free=="") exit 1; printf "%.1f", free}')"
echo "free_gb_after_env=${free_gb}"
python3 - << PY
free = float("${free_gb}")
if free < 17:
    raise SystemExit("free project space is %.1f GB after the env install; refusing the 8B download" % free)
print("free project space after the env install %.1f GB" % free)
PY

WEIGHTS="${SITE_PROJECT_ROOT}/weights/Qwen3-8B"
if [[ ! -f "${WEIGHTS}/config.json" ]]; then
  mkdir -p "${SITE_PROJECT_ROOT}/weights"
  huggingface-cli download Qwen/Qwen3-8B --local-dir "${WEIGHTS}"
fi
if [[ ! -f "${WEIGHTS}/config.json" ]]; then
  echo "Qwen3-8B download did not produce a config" >&2
  exit 1
fi
# Drop a second copy if the hub cache kept one. Do not touch other studies' files.
rm -rf "${SITE_PROJECT_ROOT}/cache/hf/hub/models--Qwen--Qwen3-8B"

export SCOG_C4_PATH="${SITE_PROJECT_ROOT}/data/data_c4_raw.csv"
export SCOG_SPLIT_LOCK="${ROOT}/configs/splits.lock"
export SCOG_MANIFEST="${SITE_PROJECT_ROOT}/runs/a1/explore_manifest.jsonl"
export SCOG_N_PER_GROUP=200
export SCOG_N_CONDITION1=20
export SCOG_N_CONDITION4=20
export PYTHONPATH="${ROOT}/src"
python -m scog.a1.prepare
echo "setup done"
