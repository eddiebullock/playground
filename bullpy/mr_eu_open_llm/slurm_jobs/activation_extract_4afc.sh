#!/usr/bin/env bash
#SBATCH -J msr_acts_4afc
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/acts_4afc_%j.out
#SBATCH -e logs/acts_4afc_%j.err
#
# Full 118-trial 4AFC-aligned extraction (~15-30 min; shorter walltime for queue fit).
#
# Baseline (no adapter):
#   sbatch slurm_jobs/activation_extract_4afc.sh
#
# Finetuned (set CHECKPOINT):
#   CHECKPOINT=results/finetune/full_runs/gemma4/run_XXXXX/adapter_final \
#   CONDITION=finetuned_gemma4_4afc \
#   sbatch slurm_jobs/activation_extract_4afc.sh

set -euo pipefail

export PROMPT_MODE=4afc
export POOLING="${POOLING:-last_token}"
export MODEL="${MODEL:-gemma4}"
export MODALITY="${MODALITY:-multimodal}"
export CONDITION="${CONDITION:-baseline_${MODEL}_4afc}"
export CHECKPOINT="${CHECKPOINT:-}"

ROOT="${SLURM_SUBMIT_DIR:-${HOME}/rds/hpc-work/study2}"
bash "${ROOT}/slurm_jobs/activation_extract.sh"
