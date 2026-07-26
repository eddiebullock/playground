#!/usr/bin/env bash
#SBATCH -J msr_s3_sae
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o logs/study3_sae_%j.out
#SBATCH -e logs/study3_sae_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MODEL="${MODEL:-gemma4}"

if [[ "${MODEL}" == "gemma4" ]]; then
  MODALITY=multimodal
else
  MODALITY=video_only
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/sae

EVAL_JSON="results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_${MODALITY}_fps1_cap16_two_stage_seed42.json"
BASE_ACT="results/activations/baseline_${MODEL}_4afc/${MODEL}"
FT_ACT="results/activations/finetuned_${MODEL}_4afc/${MODEL}"
PROBE_DIR="results/probes/baseline_${MODEL}_4afc"

if [[ "${MODEL}" == "gemma4" ]]; then
  ADAPTER="results/finetune/full_runs/gemma4/run_30364652/adapter_final"
elif [[ "${MODEL}" == "qwen2vl" ]]; then
  ADAPTER="results/finetune/full_runs/qwen2vl/run_30956896/adapter_final"
else
  ADAPTER="results/finetune/full_runs/llavanext/run_30994225/adapter_final"
fi

python -m scripts.activation_sae \
  --model "${MODEL}" \
  --baseline_act_dir "${BASE_ACT}" \
  --finetuned_act_dir "${FT_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --probe_dir "${PROBE_DIR}" \
  --output "results/sae/${MODEL}_peak_nmf.json"

echo "SAE/NMF analysis complete for ${MODEL}."
