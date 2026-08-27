#!/usr/bin/env bash
#SBATCH -J s3_act
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_act_%x_%j.out
#SBATCH -e logs/study3_act_%x_%j.err
#
# Study3 full-EU activation extract (243 trials, 6AFC, human option sets).
# Stimulus videos: ~/rds/hpc-work/study3/data/eu_emotions (synced from OneDrive EU_Emotions).
#
#   MODEL=qwen3vl MAX_FRAMES=4 sbatch -J s3_act_qwen slurm_jobs/study3_activation_extract.sh
#   MODEL=gemma4  sbatch -J s3_act_gemma slurm_jobs/study3_activation_extract.sh
#   MODEL=molmo2  sbatch -J s3_act_molmo slurm_jobs/study3_activation_extract.sh
#
# Smoke (5 trials):
#   MAX_TRIALS=5 sbatch -J s3_act_smoke slurm_jobs/study3_activation_extract_smoke.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"
CONDITION="${CONDITION:-baseline_${MODEL}_6afc}"
MODALITY="${MODALITY:-video_only}"
MANIFEST="${MANIFEST:-data/eu_emotions_full_manifest.json}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions}"
HUMAN_OPTIONS="${HUMAN_OPTIONS:-data/eu_emotions_human_entropy.json}"
MAX_TRIALS="${MAX_TRIALS:-}"
MAX_FRAMES="${MAX_FRAMES:-16}"
N_OPTIONS="${N_OPTIONS:-6}"
SEED="${SEED:-42}"
PROMPT_MODE="${PROMPT_MODE:-4afc}"
POOLING="${POOLING:-last_token}"

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
OUT_DIR="results/activations/${CONDITION}/${MODEL}"
mkdir -p "${OUT_DIR}" logs

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"

ARGS=(
  --model "${MODEL}"
  --condition "${CONDITION}"
  --manifest "${MANIFEST}"
  --data_root "${DATA_ROOT}"
  --modality "${MODALITY}"
  --prompt_mode "${PROMPT_MODE}"
  --pooling "${POOLING}"
  --max_frames "${MAX_FRAMES}"
  --fps 1
  --seed "${SEED}"
  --n_options "${N_OPTIONS}"
  --human_options "${HUMAN_OPTIONS}"
  --output_dir "${OUT_DIR}"
)
if [[ -n "${MAX_TRIALS}" ]]; then
  ARGS+=(--max_trials "${MAX_TRIALS}")
fi

echo "study3 activation extract: MODEL=${MODEL} CONDITION=${CONDITION} MAX_FRAMES=${MAX_FRAMES}"
python -m scripts.extract_activations "${ARGS[@]}"
echo "Done: ${OUT_DIR}/extract_meta.json"
