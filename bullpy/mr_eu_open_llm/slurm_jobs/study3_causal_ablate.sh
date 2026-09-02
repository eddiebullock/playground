#!/usr/bin/env bash
#SBATCH -J s3_ablate
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_ablate_%x_%j.out
#SBATCH -e logs/study3_ablate_%x_%j.err
#
# Study3 v2 primary causal: activation patching (mean-projection ablation).
# NOT additive steering — see study3_causal_steer.sh for exploratory steer.
#
#   MODE=smoke MODEL=qwen3vl LAYER=4 sbatch slurm_jobs/study3_causal_ablate.sh
#   MODE=medium MODEL=qwen3vl LAYER=4 sbatch --time=08:00:00 slurm_jobs/study3_causal_ablate.sh
#   MODE=full MODEL=qwen3vl LAYER=4 sbatch --time=12:00:00 slurm_jobs/study3_causal_ablate.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"
LAYER="${LAYER:-4}"
MODE="${MODE:-smoke}"
SEED="${SEED:-42}"
MAX_FRAMES="${MAX_FRAMES:-4}"
CONDITION="${CONDITION:-baseline_${MODEL}_6afc}"

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
mkdir -p logs results/mech

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"

ACT_DIR="results/activations/${CONDITION}/${MODEL}"
if [[ ! -d "${ACT_DIR}" ]]; then
  echo "ERROR: missing activations ${ACT_DIR}" >&2
  exit 2
fi

echo "Building EU causal axes (CPU)..."
python -m scripts.causal_eu_confusion_axes \
  --model "${MODEL}" \
  --layer "${LAYER}" \
  --activations_dir "${ACT_DIR}" \
  --seed "${SEED}"

ABLATE_ARGS=(
  --model "${MODEL}"
  --layer "${LAYER}"
  --activations_dir "${ACT_DIR}"
  --data_root data/eu_emotions
  --max_frames "${MAX_FRAMES}"
  --seed "${SEED}"
  --sample_temperature 0.0
)

if [[ "${MODE}" == "smoke" ]]; then
  ABLATE_ARGS+=(--smoke --top_pairs 1)
elif [[ "${MODE}" == "medium" ]]; then
  ABLATE_ARGS+=(
    --max_items 36
    --top_pairs 3
  )
elif [[ "${MODE}" == "full" ]]; then
  ABLATE_ARGS+=(--top_pairs 3)
else
  echo "ERROR: MODE must be smoke, medium, or full (got '${MODE}')" >&2
  exit 2
fi

echo "Running EU ablation / activation patching (${MODE})..."
echo "ABLATE_ARGS: ${ABLATE_ARGS[*]}"
python -m scripts.ablate_eu_confusion_axes "${ABLATE_ARGS[@]}"

echo "Done. See results/mech/ablate_*_${MODEL}_layer${LAYER}.*"
ls -lh "results/mech/ablate_summary_${MODEL}_layer${LAYER}.csv" 2>/dev/null || true
ls -lh "results/mech/${MODEL}_eu_ablation_layer${LAYER}.json" 2>/dev/null || true
