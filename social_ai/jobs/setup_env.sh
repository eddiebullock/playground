#!/bin/bash
# Install Miniforge under the project root on RDS. Do not run this on a login node.
# Do not run it until there is inode and byte headroom.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/local/site.sh"
# shellcheck disable=SC1091
source "${ROOT}/jobs/preamble.sh"
PREFIX="${SITE_PROJECT_ROOT}/conda"
if [[ -x "${PREFIX}/bin/conda" ]]; then
  echo "Miniforge already present at ${PREFIX}"
  exit 0
fi
INSTALLER="${SITE_PROJECT_ROOT}/cache/miniforge.sh"
mkdir -p "${SITE_PROJECT_ROOT}/cache"
curl -fsSL -o "${INSTALLER}" \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash "${INSTALLER}" -b -p "${PREFIX}"
# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
conda create -y -p "${SITE_PROJECT_ROOT}/conda/envs/scog" python=3.11
conda activate "${SITE_PROJECT_ROOT}/conda/envs/scog"
pip install -r "${ROOT}/requirements.lock"
echo "env ready"
