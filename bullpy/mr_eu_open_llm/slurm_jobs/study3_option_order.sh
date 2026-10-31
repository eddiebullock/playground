#!/usr/bin/env bash
#SBATCH -J s3_opt_order
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_optorder_%x_%j.out
#SBATCH -e logs/study3_optorder_%x_%j.err
#
# Option-order robustness: one shuffled human option order, one model.
# Do not submit while all-layer extracts / full ablation are still queued.
#
#   MODEL=qwen3vl ORDER_SEED=43 sbatch slurm_jobs/study3_option_order.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"
ORDER_SEED="${ORDER_SEED:-43}"
SEED="${SEED:-42}"
N_OPTIONS="${N_OPTIONS:-6}"
MAX_FRAMES="${MAX_FRAMES:-16}"
FC_SAMPLES="${FC_SAMPLES:-20}"

MOLMO2_VENV="${MOLMO2_VENV:-${PROJECT_ROOT}/venvs/molmo2}"
USE_MOLMO2_VENV=0
if [ "${MODEL}" = "molmo2" ] && [ -d "${MOLMO2_VENV}" ]; then
  USE_MOLMO2_VENV=1
fi
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
if [ "${USE_MOLMO2_VENV}" = "1" ]; then
  source "${MOLMO2_VENV}/bin/activate"
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs results/baseline/eu_emotions/"${MODEL}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"

LOOKUP="data/eu_emotions_human_entropy_order${ORDER_SEED}.json"
python -m paper_v2.option_score --write_lookups --seeds "${ORDER_SEED}"

FC_SEQUENTIAL="${FC_SEQUENTIAL:-0}"
EVAL_EXTRA=()
if [ "${FC_SEQUENTIAL}" = "1" ] || [ "${MODEL}" = "qwen3vl" ]; then
  EVAL_EXTRA+=(--fc_sequential)
fi

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset eu_emotions \
  --condition video_only \
  --stage stage2 \
  --max_frames "${MAX_FRAMES}" \
  --fps 1 \
  --seed "${SEED}" \
  --n_options "${N_OPTIONS}" \
  --data_root data/eu_emotions \
  --manifest data/eu_emotions_full_manifest.json \
  --human_options "${LOOKUP}" \
  --fc_samples "${FC_SAMPLES}" \
  "${EVAL_EXTRA[@]}" \
  --output "results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_video_only_seed${SEED}_order${ORDER_SEED}.json"

echo "Done."
