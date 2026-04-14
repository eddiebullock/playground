#!/usr/bin/env bash
set -euo pipefail

# Submit one baseline job per model on EU-Emotions 118-trial manifest.
# Usage:
#   bash slurm_jobs/submit_baselines.sh
# Optional overrides:
#   MAX_TRIALS=118 N_FRAMES=4 MAX_NEW_TOKENS=96 bash slurm_jobs/submit_baselines.sh

MODELS=("qwen2vl" "internvl2" "llavanext")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_JOB_SCRIPT="${SCRIPT_DIR}/baseline_eval.sh"
DATASET="${DATASET:-eu_emotions}"
N_FRAMES="${N_FRAMES:-4}"
MAX_TRIALS="${MAX_TRIALS:-118}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.1}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"

for model in "${MODELS[@]}"; do
  echo "Submitting baseline for ${model}..."
  sbatch \
    --job-name="baseline_${model}" \
    --export=MODEL="${model}",DATASET="${DATASET}",N_FRAMES="${N_FRAMES}",MAX_TRIALS="${MAX_TRIALS}",MAX_NEW_TOKENS="${MAX_NEW_TOKENS}",TEMPERATURE="${TEMPERATURE}",DATA_ROOT="${DATA_ROOT}",MANIFEST="${MANIFEST}" \
    "${BASELINE_JOB_SCRIPT}"
done

echo "All baseline jobs submitted."
