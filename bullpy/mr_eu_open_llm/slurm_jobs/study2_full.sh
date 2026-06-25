#!/usr/bin/env bash
#SBATCH -J msr_study2
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o logs/study2_%j.out
#SBATCH -e logs/study2_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
EVAL_JSON="${EVAL_JSON:-results/stats/best_model_eval.json}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u
cd "${PROJECT_ROOT}"

for MODEL in qwen2vl llavanext gemma4; do
  python -m scripts.extract_activations \
    --model "${MODEL}" \
    --condition "baseline_${MODEL}" \
    --manifest "${MANIFEST}" \
    --data_root "${DATA_ROOT}"
done

BEST=$(python -c "import json; print(json.load(open('results/stats/best_model.json'))['model_key'])")
python -m scripts.extract_activations --model "${BEST}" --condition "baseline_${BEST}" --manifest "${MANIFEST}" --data_root "${DATA_ROOT}"
python -m scripts.extract_activations --model "${BEST}" --condition "finetuned_${BEST}" --manifest "${MANIFEST}" --data_root "${DATA_ROOT}"

python -m scripts.probing \
  --activations_dir "results/activations/baseline_${BEST}/${BEST}" \
  --eval_json "${EVAL_JSON}"

python -m scripts.rsa \
  --activations "results/activations/baseline_${BEST}/${BEST}/layer0_eu_emotions_seed42.npy" \
  --output "results/rsa/baseline_${BEST}/rsa_summary.json"

python -m scripts.activation_patching \
  --model "${BEST}" \
  --confused_pairs "results/stats/confused_pairs_${BEST}.json"

echo "Study 2 pipeline finished (verify outputs under results/)."
