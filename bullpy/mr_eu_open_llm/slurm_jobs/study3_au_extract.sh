#!/usr/bin/env bash
#SBATCH -J s3_au_extract
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH -o logs/study3_au_extract_%x_%A_%a.out
#SBATCH -e logs/study3_au_extract_%x_%A_%a.err
#
# RQ1: LibreFace and OpenFace 3.0 action units on the 243 EU-Emotion clips.
# Measurement only. Per-frame numbers stay in results/au/frames on the cluster;
# research/reproduction/eu_au_measures.csv holds clip-level aggregates. No frame or
# face image is written, apart from three probe frames in node-local $TMPDIR during
# STEP=verify, which are deleted at once.
#
# Smoke (upstream equivalence, then two clips):
#   sbatch --time=00:20:00 --export=ALL,STEP=verify slurm_jobs/study3_au_extract.sh
#   sbatch --time=00:20:00 --export=ALL,MAX_CLIPS=2 slurm_jobs/study3_au_extract.sh
# Full run, then aggregates once every shard is done:
#   JOB=$(sbatch --parsable --array=0-7 slurm_jobs/study3_au_extract.sh)
#   sbatch --dependency=afterok:${JOB} --time=00:20:00 --export=ALL,STEP=aggregate \
#     slurm_jobs/study3_au_extract.sh
# GPU instead of CPU, one shard about an hour:
#   sbatch -A BARON-COHEN-SL3-GPU -p ampere --gres=gpu:1 --cpus-per-task=3 --mem=30G \
#     --export=ALL,DEVICE=cuda,N_SHARDS=1 slurm_jobs/study3_au_extract.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
TOOLS="${PROJECT_ROOT}/venvs/au_tools"
STEP="${STEP:-extract}"
DEVICE="${DEVICE:-cpu}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-8}"
MAX_CLIPS="${MAX_CLIPS:-0}"
FRAMES_DIR="${FRAMES_DIR:-results/au/frames}"

# Overlay venv on the shared study2 conda env (torch and friends); it adds MediaPipe
# and numpy 1.26 only. Environment and weights together are under 1 GB. Set up once on
# a login node, because compute jobs run offline:
#   python -m venv --system-site-packages "${TOOLS}/env"   # study2 env python
#   pip install numpy==1.26.4 mediapipe==0.10.14 gdown
#   pip download --no-deps libreface==0.2.0 openface-test==0.1.26; unzip to src/lf, src/of
#   weights/openface: Alignment_RetinaFace.pth, MTL_backbone.pth,
#     mobilenetV1X0.25_pretrain.tar (nutPace/openface_weights)
#   weights/libreface: AU_Recognition/weights/resnet.pt, AU_Detection/weights/repvgg.pt
#     (Drive ids in libreface/AU_*/inference.py)
#     and link AU_Recognition/weights/repvgg.pt to resnet.pt: upstream's image path
#     uses that name for the same Drive file, and STEP=verify takes that path
#   then build both LibreFace models once with HF_HOME, TORCH_HOME set as below, since
#   their constructors fetch ImageNet weights before the checkpoint replaces them.
if [ ! -x "${TOOLS}/env/bin/python" ]; then
  echo "Missing ${TOOLS}/env; see the setup note above." >&2
  exit 1
fi
source "${TOOLS}/env/bin/activate"

cd "${PROJECT_ROOT}"
mkdir -p logs research/reproduction "${FRAMES_DIR}"
export PYTHONPATH="${TOOLS}/src/lf:${TOOLS}/src/of${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="${TOOLS}/hf" TORCH_HOME="${TOOLS}/torch" HF_HUB_OFFLINE=1
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONWARNINGS="ignore::UserWarning"  # protobuf warns once per frame

COMMON=(
  --manifest data/eu_emotions_full_manifest.json
  --data-root data/eu_emotions
  --frames-dir "${FRAMES_DIR}"
  --libreface-weights "${TOOLS}/weights/libreface"
  --openface-weights "${TOOLS}/weights/openface"
)

case "${STEP}" in
  verify)
    python research/scripts/extract_au_measures.py verify "${COMMON[@]}" --device "${DEVICE}"
    ;;
  extract)
    python research/scripts/extract_au_measures.py extract "${COMMON[@]}" \
      --device "${DEVICE}" --shard "${SHARD}" --n-shards "${N_SHARDS}" --max-clips "${MAX_CLIPS}"
    ;;
  aggregate)
    python research/scripts/extract_au_measures.py aggregate "${COMMON[@]}" \
      --out-csv research/reproduction/eu_au_measures.csv \
      --out-summary research/reproduction/eu_au_extraction_summary.json
    ;;
  *)
    echo "Unknown STEP=${STEP}" >&2
    exit 1
    ;;
esac

echo "Done."
