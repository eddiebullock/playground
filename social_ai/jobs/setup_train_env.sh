#!/bin/bash
# Install the training env under the project root. Idempotent. Not a login-node task.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "${SITE_PROJECT_ROOT:-}" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/local/site.sh"
fi
PREFIX="${SITE_PROJECT_ROOT}/conda"
export PIP_CACHE_DIR="${SITE_PROJECT_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${SITE_PROJECT_ROOT}/cache/conda-pkgs"
mkdir -p "${SITE_PROJECT_ROOT}/cache" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"
if [[ ! -x "${PREFIX}/bin/conda" ]]; then
  INSTALLER="${SITE_PROJECT_ROOT}/cache/miniforge.sh"
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
if ! python -c "import torch, transformers, peft, datasets, pandas" >/dev/null 2>&1; then
  # The analysis lock pins contourpy 1.4.0, which needs Python 3.12.
  # Training only needs the dataframe stack plus the model libraries.
  pip install "numpy==2.2.6" "pandas==2.2.3" "PyYAML==6.0.2"
  pip install "torch==2.5.1" --index-url "https://download.pytorch.org/whl/cu124"
  pip install "transformers==4.51.3" "peft==0.15.2" "accelerate==1.6.0" "datasets==3.6.0"
fi
python -c "import torch; print('train_env_ok', torch.__version__)"
rm -rf "${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"
