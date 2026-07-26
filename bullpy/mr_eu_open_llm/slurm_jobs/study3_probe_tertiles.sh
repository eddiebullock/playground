#!/usr/bin/env bash
#SBATCH -J msr_s3_probe_tert
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o logs/study3_probe_tertiles_%j.out
#SBATCH -e logs/study3_probe_tertiles_%j.err
#
# Re-run 4AFC-aligned probes for all models/conditions with entropy tertile breakdown.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs

_probe_one() {
  local model="$1"
  local condition="$2"
  local act_dir="$3"
  local modality="$4"
  local eval_json="results/baseline/eu_emotions/${model}/eval_v2_eu_emotions_${model}_${modality}_fps1_cap16_two_stage_seed42.json"
  local out_dir="results/probes/${condition}/${model}"
  mkdir -p "${out_dir}"
  echo "=== Probes: ${condition} / ${model} ==="
  python -m scripts.probing \
    --activations_dir "${act_dir}" \
    --eval_json "${eval_json}" \
    --output "${out_dir}/probes_summary.json"
}

for MODEL in qwen2vl llavanext gemma4; do
  if [[ "${MODEL}" == "gemma4" ]]; then
    MODALITY=multimodal
  else
    MODALITY=video_only
  fi
  _probe_one "${MODEL}" "baseline_${MODEL}_4afc" "results/activations/baseline_${MODEL}_4afc/${MODEL}" "${MODALITY}"
  _probe_one "${MODEL}" "finetuned_${MODEL}_4afc" "results/activations/finetuned_${MODEL}_4afc/${MODEL}" "${MODALITY}"
done

echo "Study 3 entropy-tertile probes complete."
