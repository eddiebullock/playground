#!/usr/bin/env bash
#SBATCH -J msr_s3_path
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/study3_path_patch_%j.out
#SBATCH -e logs/study3_path_patch_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MODEL="${MODEL:-gemma4}"
MAX_TRIALS="${MAX_TRIALS:-30}"

if [[ "${MODEL}" == "gemma4" ]]; then
  MODALITY=multimodal
  ADAPTER="results/finetune/full_runs/gemma4/run_30364652/adapter_final"
elif [[ "${MODEL}" == "qwen2vl" ]]; then
  MODALITY=video_only
  ADAPTER="results/finetune/full_runs/qwen2vl/run_30956896/adapter_final"
else
  MODALITY=video_only
  ADAPTER="results/finetune/full_runs/llavanext/run_30994225/adapter_final"
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results/patching

BASE_EVAL="results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_${MODALITY}_fps1_cap16_two_stage_seed42.json"
FT_EVAL="results/finetune/eu_post_ft/eval_v2_eu_emotions_${MODEL}_${MODALITY}_finetuned_seed42.json"
BASE_ACT="results/activations/baseline_${MODEL}_4afc/${MODEL}"
PEAK_JSON="results/probes/baseline_${MODEL}_4afc/${MODEL}/peak_layer.json"

python -m scripts.path_patching \
  --model "${MODEL}" \
  --baseline_eval "${BASE_EVAL}" \
  --finetuned_eval "${FT_EVAL}" \
  --baseline_activations_dir "${BASE_ACT}" \
  --peak_layer_json "${PEAK_JSON}" \
  --checkpoint "${ADAPTER}" \
  --modality "${MODALITY}" \
  --max_trials "${MAX_TRIALS}" \
  --output "results/patching/path_patching_${MODEL}_4afc.json"

echo "Path patching complete for ${MODEL}."
