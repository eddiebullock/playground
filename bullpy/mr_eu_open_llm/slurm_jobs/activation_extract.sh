#!/usr/bin/env bash
#SBATCH -J msr_acts
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/acts_%j.out
#SBATCH -e logs/acts_%j.err
#
# Full 118-trial extraction. EU eval ~12-25 min; forward-only similar. Request 30m for queue fit.
# Smoke: use slurm_jobs/activation_extract_smoke.sh (20m, 5 trials).

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

# --- configure via sbatch --export or env before sbatch ---
MODEL="${MODEL:-gemma4}"
CONDITION="${CONDITION:-baseline_gemma4}"
MODALITY="${MODALITY:-multimodal}"
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
MAX_TRIALS="${MAX_TRIALS:-}"          # empty = all 118; set 5 for smoke
CHECKPOINT="${CHECKPOINT:-}"            # PEFT path for finetuned runs
PROMPT_MODE="${PROMPT_MODE:-free_response}"   # free_response | 4afc
POOLING="${POOLING:-}"                # mean | last_token; empty = auto from prompt_mode

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/activations

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"

ARGS=(
  --model "${MODEL}"
  --condition "${CONDITION}"
  --manifest "${MANIFEST}"
  --data_root "${DATA_ROOT}"
  --modality "${MODALITY}"
)
if [[ -n "${MAX_TRIALS}" ]]; then
  ARGS+=(--max_trials "${MAX_TRIALS}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
ARGS+=(--prompt_mode "${PROMPT_MODE}")
if [[ -n "${POOLING}" ]]; then
  ARGS+=(--pooling "${POOLING}")
fi

echo "extract_activations ${ARGS[*]}"
python -m scripts.extract_activations "${ARGS[@]}"

echo "Activation extraction complete. See results/activations/${CONDITION}/${MODEL}/extract_meta.json"
