#!/usr/bin/env bash
#SBATCH -J msr_ft_prompt
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/eval_finetune_prompt_%j.out
#SBATCH -e logs/eval_finetune_prompt_%j.err
#
# EU eval with finetune-style LABEL: prompt (stage2 only, no entropy).
# Ref: full two-stage post-FT eval ~12 min; stage2-only ~6-8 min. Request 45m for queue fit.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MODEL="${MODEL:-gemma4}"
CONDITION="${CONDITION:-multimodal}"
LORA_ADAPTER="${LORA_ADAPTER:-results/finetune/full_runs/gemma4/run_30364652/adapter_final}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/finetune/eu_post_ft

if [[ ! -d "${LORA_ADAPTER}" ]]; then
  echo "ERROR: LoRA adapter not found: ${LORA_ADAPTER}" >&2
  exit 1
fi

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset eu_emotions \
  --condition "${CONDITION}" \
  --stage stage2 \
  --stage2_prompt_mode finetune_label \
  --skip_entropy \
  --data_root data/eu_emotions_118 \
  --manifest data/eu_emotions_118_manifest.json \
  --lora_adapter "${LORA_ADAPTER}" \
  --output "results/finetune/eu_post_ft/eval_v2_eu_emotions_${MODEL}_${CONDITION}_finetune_prompt_seed42.json"

echo "Finetune-prompt EU eval complete."
