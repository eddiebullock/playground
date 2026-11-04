#!/usr/bin/env bash
#SBATCH -J s3_au_graphau
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH -o logs/study3_au_graphau_%x_%A_%a.out
#SBATCH -e logs/study3_au_graphau_%x_%A_%a.err
#
# RQ1: ME-GraphAU as a second detector for AU07, 10, 14, 15, 17, 23 and 24, which only
# LibreFace measured. Measurement only. Per-frame numbers stay in
# results/au/frames_graphau on the cluster; research/reproduction/eu_au_measures_graphau.csv
# holds clip-level aggregates, in a separate file from the existing detectors. No frame or
# face image is written. Model frames come from the frame policy; no eval file is read.
#
# Smoke (strict checkpoint load and three frames, then two clips):
#   sbatch --time=00:20:00 --export=ALL,STEP=verify slurm_jobs/study3_au_extract_graphau.sh
#   sbatch --time=00:20:00 --export=ALL,MAX_CLIPS=2,FRAMES_DIR=results/au/frames_graphau_smoke \
#     slurm_jobs/study3_au_extract_graphau.sh
# Full run, then aggregates:
#   JOB=$(sbatch --parsable --array=0-7 slurm_jobs/study3_au_extract_graphau.sh)
#   sbatch --dependency=afterok:${JOB} --time=00:20:00 --export=ALL,STEP=aggregate \
#     slurm_jobs/study3_au_extract_graphau.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
TOOLS="${PROJECT_ROOT}/venvs/au_tools"
STEP="${STEP:-extract}"
DEVICE="${DEVICE:-cpu}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-8}"
MAX_CLIPS="${MAX_CLIPS:-0}"
FRAMES_DIR="${FRAMES_DIR:-results/au/frames_graphau}"

# Set up once on a login node, in the existing au_tools env:
#   mkdir -p "${TOOLS}/src/graphau/model" "${TOOLS}/weights/graphau"
#   upstream CVI-SZU/ME-GraphAU, MIT: model/*.py and LICENSE into src/graphau, commit
#     recorded in src/graphau/UPSTREAM_COMMIT
#   gdown 1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9 (ResNet-50 BP4D, 354 MB, three folds), in
#     /tmp; keep only MEFARG_resnet50_BP4D_fold1.pth (127 MB), in weights/graphau
#   upstream's GNN.forward works on CUDA only. For CPU, in src/graphau/model/MEFL.py:
#     sed -i -e 's/dev = x.get_device()/dev = x.device/' -e 's/if dev >= 0:/if True:/'
#     (device handling only; recorded in src/graphau/LOCAL_CHANGES)
if [ ! -f "${TOOLS}/weights/graphau/MEFARG_resnet50_BP4D_fold1.pth" ]; then
  echo "Missing ME-GraphAU checkpoint; see the setup note above." >&2
  exit 1
fi
source "${TOOLS}/env/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs research/reproduction "${FRAMES_DIR}"
export PYTHONPATH="${TOOLS}/src/lf:${TOOLS}/src/of${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="${TOOLS}/hf" TORCH_HOME="${TOOLS}/torch" HF_HUB_OFFLINE=1
export TMPDIR="${TMPDIR:-/tmp}"

COMMON=(
  --manifest data/eu_emotions_full_manifest.json
  --data-root data/eu_emotions
  --frames-dir results/au/frames
  --graphau-frames-dir "${FRAMES_DIR}"
  --openface-weights "${TOOLS}/weights/openface"
  --graphau-repo "${TOOLS}/src/graphau"
  --graphau-checkpoint "${TOOLS}/weights/graphau/MEFARG_resnet50_BP4D_fold1.pth"
)

case "${STEP}" in
  verify)
    python research/scripts/extract_au_measures.py verify-graphau "${COMMON[@]}" --device "${DEVICE}"
    ;;
  extract)
    python research/scripts/extract_au_measures.py extract-graphau "${COMMON[@]}" \
      --device "${DEVICE}" --shard "${SHARD}" --n-shards "${N_SHARDS}" --max-clips "${MAX_CLIPS}"
    ;;
  aggregate)
    python research/scripts/extract_au_measures.py aggregate-graphau "${COMMON[@]}" \
      --out-csv research/reproduction/eu_au_measures.csv \
      --out-summary research/reproduction/eu_au_extraction_graphau_summary.json
    ;;
  *)
    echo "Unknown STEP=${STEP}" >&2
    exit 1
    ;;
esac

echo "Done."
