#!/usr/bin/env bash
#SBATCH -J msr_interp
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o logs/interpretability_%j.out
#SBATCH -e logs/interpretability_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p results/probes results/rsa logs

echo "Running probing..."
python -m scripts.probing \
  --activations_dir "results/activations" \
  --output "results/probes/probes_summary.json"

echo "Running RSA..."
python -m scripts.rsa \
  --activations_dir "results/activations" \
  --human_rdm "data/human_rdm.npy" \
  --output "results/rsa/rsa_summary.json"

echo "Interpretability job complete."

