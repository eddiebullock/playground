#!/usr/bin/env bash
#SBATCH -J msr_artifact_post
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o logs/artifact_post_%j.out
#SBATCH -e logs/artifact_post_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/stats/figures

bash slurm_jobs/augment_all_baselines.sh

python -m scripts.artifact_results_table \
  --output-json results/stats/artifact_master_table.json \
  --output-md results/stats/artifact_master_table.md

shopt -s nullglob
ARTIFACTS=(results/baseline/eu_emotions/*/eval_artifact_*.json results/finetune/eu_post_ft/eval_artifact_*.json)
if ((${#ARTIFACTS[@]} > 0)); then
  CAL_ARGS=()
  for f in "${ARTIFACTS[@]}"; do
    CAL_ARGS+=(--input "${f}")
  done
  python -m scripts.plot_calibration "${CAL_ARGS[@]}"
fi

echo "Artifact post-processing complete."
