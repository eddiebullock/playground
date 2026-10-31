#!/usr/bin/env bash
#SBATCH -J s3_stim_feat
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/study3_stim_%x_%j.out
#SBATCH -e logs/study3_stim_%x_%j.err
#
# Duration / luminance / motion-energy features for the Phase 2 stimulus baseline.
#
#   sbatch slurm_jobs/study3_stimulus_features.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs research/reproduction
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

python -m paper_v2.stimulus \
  --manifest data/eu_emotions_full_manifest.json \
  --data_root data/eu_emotions \
  --out research/reproduction/stimulus_features.json

echo "Done."
