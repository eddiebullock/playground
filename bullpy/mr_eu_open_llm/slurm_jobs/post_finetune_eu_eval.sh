#!/usr/bin/env bash
#SBATCH -J msr_ft_eu_eval
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/post_finetune_eu_%j.out
#SBATCH -e logs/post_finetune_eu_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MODEL="${MODEL:-gemma4}"
if [[ -z "${CONDITION:-}" ]]; then
  if [[ "${MODEL}" == "gemma4" ]]; then
    CONDITION=multimodal
  else
    CONDITION=video_only
  fi
else
  CONDITION="${CONDITION}"
fi
LORA_ADAPTER="${LORA_ADAPTER:?Set LORA_ADAPTER to adapter_final path}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset eu_emotions \
  --condition "${CONDITION}" \
  --stage both \
  --data_root data/eu_emotions_118 \
  --manifest data/eu_emotions_118_manifest.json \
  --lora_adapter "${LORA_ADAPTER}" \
  --output "results/finetune/eu_post_ft/eval_v2_eu_emotions_${MODEL}_${CONDITION}_finetuned_seed42.json"

echo "Post-FT EU eval complete."
