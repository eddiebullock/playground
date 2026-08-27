#!/usr/bin/env bash
#SBATCH -J s3_act_smoke
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o logs/study3_act_smoke_%x_%j.out
#SBATCH -e logs/study3_act_smoke_%x_%j.err

set -euo pipefail

export MAX_TRIALS="${MAX_TRIALS:-5}"
ROOT="${SLURM_SUBMIT_DIR:-${HOME}/rds/hpc-work/study3}"
bash "${ROOT}/slurm_jobs/study3_activation_extract.sh"
