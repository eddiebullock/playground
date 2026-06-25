#!/usr/bin/env bash
#SBATCH -J msr_frame_ablation
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/frame_ablation_%j.out
#SBATCH -e logs/frame_ablation_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MODEL="${MODEL:-qwen2vl}"
MAX_TRIALS="${MAX_TRIALS:-30}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u
cd "${PROJECT_ROOT}"
mkdir -p results/ablation logs

for FRAMES in 4 16; do
  python -m scripts.evaluate \
    --model "${MODEL}" \
    --dataset eu_emotions \
    --max_frames "${FRAMES}" \
    --fps 1 \
    --max_trials "${MAX_TRIALS}" \
    --data_root data/eu_emotions_118 \
    --manifest data/eu_emotions_118_manifest.json \
    --output "results/ablation/${MODEL}_fps1_cap${FRAMES}_n${MAX_TRIALS}.json"
done

echo "Frame ablation complete."
