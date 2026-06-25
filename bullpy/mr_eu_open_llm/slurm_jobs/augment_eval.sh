#!/usr/bin/env bash
#SBATCH -J msr_augment
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o logs/augment_%j.out
#SBATCH -e logs/augment_%j.err

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

BASELINE_IN="${BASELINE_IN:-results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_multimodal_fps1_cap16_two_stage_seed42.json}"
BASELINE_OUT="${BASELINE_OUT:-results/baseline/eu_emotions/gemma4/eval_artifact_gemma4_multimodal_seed42.json}"
FT_IN="${FT_IN:-results/finetune/eu_post_ft/eval_v2_eu_emotions_gemma4_multimodal_finetuned_seed42.json}"
FT_OUT="${FT_OUT:-results/finetune/eu_post_ft/eval_artifact_gemma4_multimodal_finetuned_seed42.json}"

python -m scripts.augment_eval_artifact --input "${BASELINE_IN}" --output "${BASELINE_OUT}"
python -m scripts.augment_eval_artifact --input "${FT_IN}" --output "${FT_OUT}"

echo "Augment eval complete."
