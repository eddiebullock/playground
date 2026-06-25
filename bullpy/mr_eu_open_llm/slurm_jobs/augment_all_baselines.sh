#!/usr/bin/env bash
#SBATCH -J msr_augment_all
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o logs/augment_all_%j.out
#SBATCH -e logs/augment_all_%j.err
#
# Augment all baseline + post-FT eval_v2 JSONs. Prefer sbatch from study2 root.
# Full pipeline: sbatch slurm_jobs/artifact_postprocess.sh

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
mkdir -p logs

_augment_one() {
  local in_path="$1"
  local base
  base="$(basename "${in_path}")"
  if [[ "${base}" != eval_v2_* ]]; then
    return 0
  fi
  local out_dir
  out_dir="$(dirname "${in_path}")"
  local out_path="${out_dir}/${base/eval_v2_/eval_artifact_}"
  echo "Augmenting ${in_path} -> ${out_path}"
  python -m scripts.augment_eval_artifact --input "${in_path}" --output "${out_path}"
}

shopt -s nullglob
for model_dir in results/baseline/eu_emotions/*/; do
  for f in "${model_dir}"eval_v2_*.json; do
    _augment_one "${f}"
  done
done

for f in results/finetune/eu_post_ft/eval_v2_*.json; do
  _augment_one "${f}"
done

for f in results/ablation/eval_v2_*.json; do
  _augment_one "${f}"
done

echo "Augment all baselines complete."
