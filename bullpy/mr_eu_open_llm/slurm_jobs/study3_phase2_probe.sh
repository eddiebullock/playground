#!/usr/bin/env bash
#SBATCH -J s3_phase2
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/study3_phase2_%x_%j.out
#SBATCH -e logs/study3_phase2_%x_%j.err
#
# Confirmatory Phase 2 probes (CPU). Layer set is whatever is on disk.
#
#   LAYER_SET=pilot_3layer_last_token sbatch slurm_jobs/study3_phase2_probe.sh
#   LAYER_SET=all_layers sbatch slurm_jobs/study3_phase2_probe.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"
LAYER_SET="${LAYER_SET:-pilot_3layer_last_token}"
N_PERM="${N_PERM:-1000}"
N_BOOT="${N_BOOT:-2000}"

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

EXTRA_STIM=""
if [[ -f research/reproduction/stimulus_features.json ]]; then
  EXTRA_STIM="--extra_stim research/reproduction/stimulus_features.json"
fi

python -m paper_v2.probe \
  --layer_set "${LAYER_SET}" \
  --n_perm "${N_PERM}" \
  --n_boot "${N_BOOT}" \
  --n_jobs 8 \
  --out "research/reproduction/phase2_gate_${LAYER_SET}.json" \
  ${EXTRA_STIM}

echo "Done."
