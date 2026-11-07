#!/usr/bin/env bash
#SBATCH -J s3_a4_render
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH -o logs/study3_a4_render_%A_%a.out
#SBATCH -e logs/study3_a4_render_%A_%a.err
#
# Renders the A4 manifest. CPU only, numpy, scipy and Pillow. No model is involved.
#
#   python research/scripts/a4_stimuli.py manifest --zip data/external/snoek_au/au_emotion_data.zip --out results/a4/manifest.csv
#   sbatch --array=0-39 slurm_jobs/study3_a4_render.sh
#
# The assets are one archive on RDS and hold 2,590 files, so they are unpacked to node-local
# storage for each task, never onto RDS. Images are written into tar files.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
SHARD="${SLURM_ARRAY_TASK_ID:?submit as an array}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-mr_eu_open_llm}"
set -u

cd "${PROJECT_ROOT}"
LOCAL_ASSETS="$(mktemp -d -p "${TMPDIR:-/tmp}" au_assets.XXXXXX)"
trap 'rm -rf "${LOCAL_ASSETS}"' EXIT
tar -xzf "${PROJECT_ROOT}/data/au_render_assets.tgz" -C "${LOCAL_ASSETS}"
export AU_RENDER_ASSETS="${LOCAL_ASSETS}"
python research/scripts/a4_stimuli.py render --manifest results/a4/manifest.csv \
  --out-dir results/a4/renders --tar "${SHARD}"
