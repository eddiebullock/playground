#!/usr/bin/env bash
#SBATCH -J msr_acts_4afc_smoke
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/acts_4afc_smoke_%j.out
#SBATCH -e logs/acts_4afc_smoke_%j.err
#
# 5-trial smoke with 4AFC prompts + last-token pooling (aligned with patching).

set -euo pipefail

export MAX_TRIALS="${MAX_TRIALS:-5}"
export PROMPT_MODE=4afc
export POOLING="${POOLING:-last_token}"
export MODEL="${MODEL:-gemma4}"
export MODALITY="${MODALITY:-multimodal}"
export CONDITION="${CONDITION:-baseline_${MODEL}_4afc}"

ROOT="${SLURM_SUBMIT_DIR:-${HOME}/rds/hpc-work/study2}"
bash "${ROOT}/slurm_jobs/activation_extract.sh"
