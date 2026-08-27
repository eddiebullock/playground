#!/usr/bin/env bash
#SBATCH -J s4_rmet_align
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o logs/study4_rmet_align_%j.out
#SBATCH -e logs/study4_rmet_align_%j.err
#
# Steps 4–6 (A1/A2/A3) on icelake CPU — do NOT run on the login node.
#
#   sbatch -J s4_rmet_align study4_rmet/slurm_jobs/rmet_alignment_cpu.sh
#
# Requires full model eval JSONs under study4_rmet/results/model/*/rmet_eval_*_full_*.json

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study4_rmet
ENV_ROOT=~/rds/hpc-work/study2
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"
N_PERM="${N_PERM:-5000}"
SEED="${SEED:-42}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "note: dropping inherited VIRTUAL_ENV=${VIRTUAL_ENV}"
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs study4_rmet/results/alignment

QWEN_JSON="study4_rmet/results/model/qwen3vl/rmet_eval_qwen3vl_full_seed${SEED}.json"
GEMMA_JSON="study4_rmet/results/model/gemma4/rmet_eval_gemma4_full_seed${SEED}.json"
MOLMO_JSON="study4_rmet/results/model/molmo2/rmet_eval_molmo2_full_seed${SEED}.json"

for f in "${QWEN_JSON}" "${GEMMA_JSON}" "${MOLMO_JSON}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: missing ${f}" >&2
    exit 2
  fi
done

echo "study4 RMET alignment (CPU): n_perm=${N_PERM} seed=${SEED}"
echo "python $(command -v python)"

python study4_rmet/scripts/alignment_analyses.py \
  --human_csv study4_rmet/results/human/item_trait_sensitivity.csv \
  --card_csv study4_rmet/data/processed/card_rmet_item_level.csv \
  --answer_key study4_rmet/data/rmet/answer_key/rmet_adult_answer_key.json \
  --outdir study4_rmet/results/alignment \
  --n_perm "${N_PERM}" \
  --seed "${SEED}" \
  --model_eval "qwen3vl=${QWEN_JSON}" \
  --model_eval "gemma4=${GEMMA_JSON}" \
  --model_eval "molmo2=${MOLMO_JSON}"

echo "study4 RMET alignment complete."
ls -lh study4_rmet/results/alignment/
