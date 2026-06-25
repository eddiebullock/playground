#!/usr/bin/env bash
#SBATCH -J msr_patch_v2
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/patching_v2_%j.out
#SBATCH -e logs/patching_v2_%j.err
#
# v2 patching: same-trial baseline->FT, last-token L25, FT-incorrect trials only.
# Default MAX_TRIALS=0 → all FT-wrong (~115, ~45-90m). Smoke: MAX_TRIALS=30 (~15-20m).

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-gemma4}"
BASELINE_EVAL="${BASELINE_EVAL:-results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_multimodal_fps1_cap16_two_stage_seed42.json}"
FINETUNED_EVAL="${FINETUNED_EVAL:-results/finetune/eu_post_ft/eval_v2_eu_emotions_gemma4_multimodal_finetuned_seed42.json}"
BASELINE_ACT="${BASELINE_ACT:-results/activations/baseline_gemma4/gemma4}"
PEAK_JSON="${PEAK_JSON:-results/probes/baseline_gemma4/gemma4/peak_layer.json}"
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
  --modality multimodal
  --max_trials "${MAX_TRIALS}"
  --patch_mode "${PATCH_MODE}"
  --output results/patching/patching_results_gemma4_v2.json
)

python -m scripts.activation_patching "${ARGS[@]}"

echo "Patching v2 complete."
