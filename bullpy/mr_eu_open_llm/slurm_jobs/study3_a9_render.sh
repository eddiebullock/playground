#!/usr/bin/env bash
#SBATCH -J s3_a9_render
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH -o logs/study3_a9_render_%A_%a.out
#SBATCH -e logs/study3_a9_render_%A_%a.err
#
# Renders the pinned A9 manifest (design B, 5,992 unique images) into 40 tar files, as A4 does.
# CPU only. No model is involved. Run only after the freeze: the preflight below refuses if A9's
# code is not pinned in the draft, or if A4 files, A9 inputs or the manifest differ from the block.
#
#   python -I research/scripts/a9_manifest.py manifest --out results/a9/manifest.csv
#   sbatch --array=0-39 slurm_jobs/study3_a9_render.sh

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
python -c "from pathlib import Path; from scripts import a9_design as a; a.check_code_pins(require=True); a.check_pins(groups=('a4_pins',)); a.check_manifest(Path('results/a9/manifest.csv'))"
LOCAL_ASSETS="$(mktemp -d -p "${TMPDIR:-/tmp}" au_assets.XXXXXX)"
trap 'rm -rf "${LOCAL_ASSETS}"' EXIT
tar -xzf "${PROJECT_ROOT}/data/au_render_assets.tgz" -C "${LOCAL_ASSETS}" 2>/dev/null
export AU_RENDER_ASSETS="${LOCAL_ASSETS}/src"  # the archive has a src/ prefix
python research/scripts/a9_manifest.py render --manifest results/a9/manifest.csv \
  --out-dir results/a9/renders --tar "${SHARD}"
