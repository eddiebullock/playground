#!/usr/bin/env bash
#SBATCH -J s3_family
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH -o logs/study3_family_%x_%j.out
#SBATCH -e logs/study3_family_%x_%j.err
#
# RQ1 family-level same-clip analysis. Numbers only: human response norms and clip-level
# AU aggregates go in, JSON and CSV come out. No clip and no rotation result is read.
#
#   sbatch --export=ALL,STEP=recovery,CUE_SET=families slurm_jobs/study3_family_recovery.sh
#   sbatch --export=ALL,STEP=recovery,CUE_SET=families_singles slurm_jobs/study3_family_recovery.sh
#   sbatch --time=02:00:00 --cpus-per-task=8 --mem=16G --export=ALL,STEP=human \
#     slurm_jobs/study3_family_recovery.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
STEP="${STEP:-recovery}"
CUE_SET="${CUE_SET:-families}"
GRAPHAU_CSV=research/reproduction/eu_au_measures_graphau.csv
source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARGS=(--au-csv research/reproduction/eu_au_measures.csv --workers "${SLURM_CPUS_PER_TASK:-8}")
if [ -f "${GRAPHAU_CSV}" ]; then
  ARGS+=(--graphau-csv "${GRAPHAU_CSV}")
fi

case "${STEP}" in
  recovery)
    python research/scripts/same_clip_family_analysis.py --recovery --cue-set "${CUE_SET}" "${ARGS[@]}"
    ;;
  human)
    python research/scripts/same_clip_family_analysis.py "${ARGS[@]}"
    ;;
  *)
    echo "Unknown STEP=${STEP}" >&2
    exit 1
    ;;
esac

echo "Done."
