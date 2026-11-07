#!/usr/bin/env bash
#SBATCH -J s3_a4_v1
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH -o logs/study3_a4_v1_%j.out
#SBATCH -e logs/study3_a4_v1_%j.err
#
# Gate V1: LibreFace on one-AU renders. CPU only, and no vision-language model. Assets are one
# archive with a src/ prefix, unpacked to node-local storage, never onto RDS. Only a CSV is kept.
#
#   OUT=results/a4/detector_v1_pilot.csv sbatch slurm_jobs/study3_a4_detector.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
TOOLS="${PROJECT_ROOT}/venvs/au_tools"
OUT="${OUT:-results/a4/detector_v1.csv}"

source "${TOOLS}/env/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p logs results/a4
export PYTHONPATH="${TOOLS}/src/lf:${TOOLS}/src/of:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="${TOOLS}/hf" TORCH_HOME="${TOOLS}/torch" HF_HUB_OFFLINE=1
export PYTHONWARNINGS="ignore::UserWarning"

LOCAL_ASSETS="$(mktemp -d -p "${TMPDIR:-/tmp}" au_assets.XXXXXX)"
trap 'rm -rf "${LOCAL_ASSETS}"' EXIT
tar -xzf "${PROJECT_ROOT}/data/au_render_assets.tgz" -C "${LOCAL_ASSETS}" 2>/dev/null
export AU_RENDER_ASSETS="${LOCAL_ASSETS}/src"
python research/scripts/a4_detector.py --manifest results/a4/manifest.csv \
  --weights "${TOOLS}/weights/libreface" --out "${OUT}"
