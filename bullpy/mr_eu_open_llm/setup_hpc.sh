#!/usr/bin/env bash
set -euo pipefail

# Run on CSD3 after SSH:
#   cd ~/rds/hpc-work/study2
#   bash setup_hpc.sh
#
# Model downloads are large; use SKIP_MODEL_DOWNLOAD=1 if weights already exist.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

echo "Creating directory structure under ${PROJECT_ROOT}..."
mkdir -p "${PROJECT_ROOT}/data/eu_emotions"
mkdir -p "${PROJECT_ROOT}/data/mindreading"
mkdir -p "${PROJECT_ROOT}/data/rmet"
mkdir -p "${PROJECT_ROOT}/data/cache"
mkdir -p "${PROJECT_ROOT}/models/qwen2vl"
mkdir -p "${PROJECT_ROOT}/models/llavanext"
mkdir -p "${PROJECT_ROOT}/models/gemma4"
mkdir -p "${PROJECT_ROOT}/results"
mkdir -p "${PROJECT_ROOT}/scripts"
mkdir -p "${PROJECT_ROOT}/slurm_jobs"
mkdir -p "${PROJECT_ROOT}/logs"

echo "Loading miniconda module..."
module load miniconda || module load miniconda3 || true

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found after loading miniconda module. Please check module names on CSD3."
  exit 1
fi

export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"

ENV_NAME="mr_eu_open_llm"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda environment ${ENV_NAME} already exists. Skipping creation."
else
  echo "Creating conda environment ${ENV_NAME} from environment.yml..."
  conda env create -f "${PROJECT_ROOT}/environment.yml" || {
    echo "Falling back to manual environment creation..."
    conda create -y -n "${ENV_NAME}" python=3.10
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    pip install -r "${PROJECT_ROOT}/requirements.txt"
  }
fi

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# transformers 4.57.x requires huggingface_hub<1.0 (do not pip install --upgrade huggingface_hub)
echo "Pinning huggingface_hub for transformers compatibility..."
python -m pip install "huggingface_hub>=0.34.0,<1.0"

echo "Installing protocol v2 dependency (semantic entropy)..."
python -m pip install sentence-transformers

echo "Installing audio deps for Gemma multimodal (transformers load_audio_librosa)..."
python -m pip install librosa soundfile

echo "Ensuring conda-forge ffmpeg (avoid broken system module on compute nodes)..."
conda install -y -c conda-forge ffmpeg || echo "ffmpeg conda install failed; run manually."

hf_download() {
  local repo_id="$1"
  local local_dir="$2"
  if [[ -f "${local_dir}/config.json" ]] || [[ -f "${local_dir}/model.safetensors.index.json" ]]; then
    echo "Already present: ${local_dir} (skipping ${repo_id})"
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo_id}" --local-dir "${local_dir}" || return 1
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" --local-dir "${local_dir}" --local-dir-use-symlinks False || return 1
  else
    python -m huggingface_hub.cli download "${repo_id}" --local-dir "${local_dir}" || return 1
  fi
}

MODEL_DIR="${PROJECT_ROOT}/models"

if [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]]; then
  echo "SKIP_MODEL_DOWNLOAD=1: skipping model downloads."
else
  echo "Downloading models (hf download; may take a long time)..."
  hf_download "Qwen/Qwen2-VL-7B-Instruct" "${MODEL_DIR}/qwen2vl" \
    || echo "Qwen2-VL-7B-Instruct download failed or partially cached."
  hf_download "llava-hf/llava-interleave-qwen-7b-hf" "${MODEL_DIR}/llavanext" \
    || echo "LLaVA-NeXT download failed or partially cached."
  hf_download "google/gemma-4-E4B-it" "${MODEL_DIR}/gemma4" \
    || echo "Gemma4 download failed or partially cached."
fi

echo "Verifying Python stack..."
python -c "import transformers; import huggingface_hub; print('transformers', transformers.__version__, 'hub', huggingface_hub.__version__)"

echo "HPC setup complete."
echo "Next: check data/eu_emotions_118_manifest.json, then: MAX_TRIALS=5 sbatch slurm_jobs/test_job.sh"
