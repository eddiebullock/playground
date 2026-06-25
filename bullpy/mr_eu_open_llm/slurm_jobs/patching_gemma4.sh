#!/usr/bin/env bash
#SBATCH -J msr_patch
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/patching_%j.out
#SBATCH -e logs/patching_%j.err
#
# Inject baseline activations at peak probe layer into finetuned model (5 pairs).
# Ref: ~2 forwards/pair; 5 pairs should fit in ~20-30 min. Request 45m for queue fit.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-gemma4}"
BASELINE_EVAL="${BASELINE_EVAL:-results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_multimodal_fps1_cap16_two_stage_seed42.json}"
FINETUNED_EVAL="${FINETUNED_EVAL:-results/finetune/eu_post_ft/eval_v2_eu_emotions_gemma4_multimodal_finetuned_seed42.json}"
BASELINE_ACT="${BASELINE_ACT:-results/activations/baseline_gemma4/gemma4}"
PEAK_JSON="${PEAK_JSON:-results/probes/baseline_gemma4/gemma4/peak_layer.json}"
CONFUSED="${CONFUSED:-results/stats/confused_pairs_gemma4.json}"
LORA_ADAPTER="${LORA_ADAPTER:-results/finetune/full_runs/gemma4/run_30364652/adapter_final}"
MAX_PAIRS="${MAX_PAIRS:-5}"

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

python -m scripts.activation_patching \
  --model "${MODEL}" \
  --baseline_eval "${BASELINE_EVAL}" \
  --finetuned_eval "${FINETUNED_EVAL}" \
  --baseline_activations_dir "${BASELINE_ACT}" \
  --peak_layer_json "${PEAK_JSON}" \
  --confused_pairs "${CONFUSED}" \
  --checkpoint "${LORA_ADAPTER}" \
  --modality multimodal \
  --max_pairs "${MAX_PAIRS}" \
  --output results/patching/patching_results_gemma4.json

echo "Patching complete."
