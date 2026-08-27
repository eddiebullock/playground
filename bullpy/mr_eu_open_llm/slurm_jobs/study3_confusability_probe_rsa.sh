#!/usr/bin/env bash
#SBATCH -J s3_mech_cpu
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/study3_mech_%x_%j.out
#SBATCH -e logs/study3_mech_%x_%j.err
#
# Build human confusion RDM (if missing) + confusability RSA/probes for one model.
#
#   MODEL=qwen3vl sbatch slurm_jobs/study3_confusability_probe_rsa.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"
MODEL="${MODEL:-qwen3vl}"
CONDITION="${CONDITION:-baseline_${MODEL}_6afc}"
N_PERM="${N_PERM:-2000}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs data results/mech

if [[ ! -f data/human_confusion_rdm.npy ]]; then
  echo "Building human confusion artifacts..."
  python -m scripts.build_human_confusion \
    --manifest data/eu_emotions_full_manifest.json \
    --human data/eu_emotions_human_entropy.json \
    --out-dir data
fi

ACT_DIR="results/activations/${CONDITION}/${MODEL}"
if [[ ! -d "${ACT_DIR}" ]]; then
  echo "ERROR: missing activations at ${ACT_DIR}" >&2
  echo "Submit GPU extract first: MODEL=${MODEL} sbatch slurm_jobs/study3_activation_extract.sh" >&2
  exit 2
fi

python -m scripts.confusability_probe_rsa \
  --model "${MODEL}" \
  --activations_dir "${ACT_DIR}" \
  --out "results/mech/${MODEL}_confusability_probe_rsa.json" \
  --n_perm "${N_PERM}"

echo "Done."
