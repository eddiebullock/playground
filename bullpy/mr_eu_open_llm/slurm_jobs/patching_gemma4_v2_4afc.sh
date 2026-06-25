#!/usr/bin/env bash
#SBATCH -J msr_patch_v2_4afc
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/patching_v2_4afc_%j.out
#SBATCH -e logs/patching_v2_4afc_%j.err
#
# Patching with 4AFC-aligned baseline activations + peak layer from 4AFC probes.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-gemma4}"
if [[ "${MODEL}" == "gemma4" ]]; then
  MODALITY="${MODALITY:-multimodal}"
else
  MODALITY="${MODALITY:-video_only}"
fi

BASELINE_EVAL="${BASELINE_EVAL:-results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_${MODALITY}_fps1_cap16_two_stage_seed42.json}"
FINETUNED_EVAL="${FINETUNED_EVAL:-results/finetune/eu_post_ft/eval_v2_eu_emotions_${MODEL}_${MODALITY}_finetuned_seed42.json}"
BASELINE_ACT="${BASELINE_ACT:-results/activations/baseline_${MODEL}_4afc/${MODEL}}"
PEAK_JSON="${PEAK_JSON:-results/probes/baseline_${MODEL}_4afc/${MODEL}/peak_layer.json}"
LORA_ADAPTER="${LORA_ADAPTER:-results/finetune/full_runs/gemma4/run_30364652/adapter_final}"
MAX_TRIALS="${MAX_TRIALS:-0}"
PATCH_MODE="${PATCH_MODE:-last_token}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/patching

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"

ARGS=(
  --model "${MODEL}"
  --baseline_eval "${BASELINE_EVAL}"
  --finetuned_eval "${FINETUNED_EVAL}"
  --baseline_activations_dir "${BASELINE_ACT}"
  --peak_layer_json "${PEAK_JSON}"
  --checkpoint "${LORA_ADAPTER}"
  --modality "${MODALITY}"
  --max_trials "${MAX_TRIALS}"
  --patch_mode "${PATCH_MODE}"
  --output "results/patching/patching_results_${MODEL}_v2_4afc.json"
)

python -m scripts.activation_patching "${ARGS[@]}"

echo "Patching v2 (4AFC activations) complete."
