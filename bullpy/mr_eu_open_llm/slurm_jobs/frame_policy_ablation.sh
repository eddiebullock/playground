#!/usr/bin/env bash
#SBATCH -J msr_frame_policy
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/frame_policy_%j.out
#SBATCH -e logs/frame_policy_%j.err
#
# Frame policy ablation: composite_grid vs native_video (same fps/cap).
# Default: gemma4 multimodal, 30 trials. Full 118: MAX_TRIALS=118 --time=04:00:00

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-gemma4}"
CONDITION="${CONDITION:-multimodal}"
MAX_TRIALS="${MAX_TRIALS:-30}"
FPS="${FPS:-1}"
MAX_FRAMES="${MAX_FRAMES:-16}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"
TEMPERATURE="${TEMPERATURE:-0.1}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p results/ablation logs

HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}/modules" "${HF_CACHE_DIR}/transformers" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/torch"
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"

COMPOSITE_OUT="results/ablation/eval_v2_${MODEL}_${CONDITION}_composite_grid_fps${FPS}_cap${MAX_FRAMES}_n${MAX_TRIALS}_seed42.json"
NATIVE_OUT="results/ablation/eval_v2_${MODEL}_${CONDITION}_native_video_fps${FPS}_cap${MAX_FRAMES}_n${MAX_TRIALS}_seed42.json"

for MODE in composite_grid native_video; do
  OUT="${COMPOSITE_OUT}"
  if [[ "${MODE}" == "native_video" ]]; then
    OUT="${NATIVE_OUT}"
  fi
  echo "Running frame_mode=${MODE} -> ${OUT}"
  python -m scripts.evaluate \
    --model "${MODEL}" \
    --dataset eu_emotions \
    --condition "${CONDITION}" \
    --frame_mode "${MODE}" \
    --fps "${FPS}" \
    --max_frames "${MAX_FRAMES}" \
    --stage both \
    --data_root "${DATA_ROOT}" \
    --manifest "${MANIFEST}" \
    --max_trials "${MAX_TRIALS}" \
    --temperature "${TEMPERATURE}" \
    --output "${OUT}"
  python -m scripts.augment_eval_artifact --input "${OUT}" --output "${OUT/eval_v2_/eval_artifact_}"
done

python -m scripts.summarize_frame_policy_ablation \
  --composite "${COMPOSITE_OUT}" \
  --native "${NATIVE_OUT}" \
  --output-json results/ablation/frame_policy_summary.json \
  --output-md results/ablation/frame_policy_summary.md

echo "Frame policy ablation complete."
