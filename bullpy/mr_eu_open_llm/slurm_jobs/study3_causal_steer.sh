#!/usr/bin/env bash
#SBATCH -J s3_steer
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_steer_%x_%j.out
#SBATCH -e logs/study3_steer_%x_%j.err
#
# EU confusability causal steer (study3). Builds axes (CPU) then GPU steer.
#
#   MODEL=qwen3vl LAYER=4 sbatch slurm_jobs/study3_causal_steer.sh
#   MODE=smoke MODEL=qwen3vl LAYER=4 sbatch slurm_jobs/study3_causal_steer.sh
#   MODE=medium MODEL=qwen3vl LAYER=4 sbatch --time=08:00:00 slurm_jobs/study3_causal_steer.sh
#   MODE=full MODEL=qwen3vl LAYER=4 sbatch --time=12:00:00 slurm_jobs/study3_causal_steer.sh

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

STEER_ARGS=(
  --model "${MODEL}"
  --layer "${LAYER}"
  --data_root data/eu_emotions
  --max_frames "${MAX_FRAMES}"
  --seed "${SEED}"
)

if [[ "${MODE}" == "smoke" ]]; then
  STEER_ARGS+=(--smoke --patch_modes last_token --alphas=-1,1)
elif [[ "${MODE}" == "medium" ]]; then
  STEER_ARGS+=(
    --max_items 36
    --n_samples 5
    --top_pairs 3
    --patch_modes last_token
    --alphas=-1,1
  )
elif [[ "${MODE}" == "full" ]]; then
  STEER_ARGS+=(
    --max_items 36
    --n_samples 10
    --patch_modes last_token,all_tokens
    --alphas=-2,-1,1,2
  )
else
  echo "ERROR: MODE must be smoke, medium, or full (got '${MODE}')" >&2
  exit 2
fi

echo "Running EU steer (${MODE})..."
echo "STEER_ARGS: ${STEER_ARGS[*]}"
python -m scripts.steer_eu_confusion_axes "${STEER_ARGS[@]}"

echo "Done. See results/mech/steer_*_${MODEL}_layer${LAYER}.*"
ls -lh "results/mech/steer_summary_${MODEL}_layer${LAYER}."* 2>/dev/null || ls -lh results/mech/steer_*"${MODEL}"* 2>/dev/null | head -20
