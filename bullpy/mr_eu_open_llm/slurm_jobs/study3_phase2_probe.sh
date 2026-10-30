#!/usr/bin/env bash
#SBATCH -J s3_phase2
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/study3_phase2_%x_%j.out
#SBATCH -e logs/study3_phase2_%x_%j.err
#
# Confirmatory Phase 2 probes (CPU). Layer set is whatever is on disk.
#
#   MODEL=qwen3vl LAYER_SET=pilot_3layer_last_token sbatch slurm_jobs/study3_phase2_probe.sh
#   MODEL=qwen3vl LAYER_SET=all_layers CONDITION=baseline_qwen3vl_6afc_all_layers_f4 \
#     sbatch --dependency=afterok:JOBID slurm_jobs/study3_phase2_probe.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"
MODEL="${MODEL:-qwen3vl}"
LAYER_SET="${LAYER_SET:-pilot_3layer_last_token}"
CONDITION="${CONDITION:-baseline_${MODEL}_6afc}"
N_PERM="${N_PERM:-1000}"
N_BOOT="${N_BOOT:-2000}"
ACT_DIR="${ACT_DIR:-results/activations/${CONDITION}/${MODEL}}"

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

if [[ ! -d "${ACT_DIR}" ]]; then
  echo "ERROR: missing activations at ${ACT_DIR}" >&2
  exit 2
fi

EXTRA_STIM=()
if [[ -f research/reproduction/stimulus_features.json ]]; then
  EXTRA_STIM=(--extra_stim research/reproduction/stimulus_features.json)
fi

python -m paper_v2.probe \
  --models "${MODEL}" \
  --act_dir "${ACT_DIR}" \
  --layer_set "${LAYER_SET}" \
  --n_perm "${N_PERM}" \
  --n_boot "${N_BOOT}" \
  --n_jobs 8 \
  --out "research/reproduction/phase2_gate_${LAYER_SET}_${MODEL}.json" \
  "${EXTRA_STIM[@]}"

echo "Done."
