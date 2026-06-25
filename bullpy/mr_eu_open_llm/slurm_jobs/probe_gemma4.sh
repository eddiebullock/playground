#!/usr/bin/env bash
#SBATCH -J msr_probe
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/probe_%j.out
#SBATCH -e logs/probe_%j.err
#
# Linear probes on baseline vs finetuned Gemma4 EU activations (CPU compute node).

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

EVAL_JSON="${EVAL_JSON:-results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_multimodal_fps1_cap16_two_stage_seed42.json}"
BASELINE_ACT="${BASELINE_ACT:-results/activations/baseline_gemma4/gemma4}"
FINETUNED_ACT="${FINETUNED_ACT:-results/activations/finetuned_gemma4/gemma4}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs \
  results/probes/baseline_gemma4/gemma4 \
  results/probes/finetuned_gemma4/gemma4

echo "=== Baseline probes (activations: ${BASELINE_ACT}) ==="
python -m scripts.probing \
  --activations_dir "${BASELINE_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --output results/probes/baseline_gemma4/gemma4/probes_summary.json

echo "=== Finetuned probes (activations: ${FINETUNED_ACT}) ==="
python -m scripts.probing \
  --activations_dir "${FINETUNED_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --output results/probes/finetuned_gemma4/gemma4/probes_summary.json

echo "=== Summary ==="
python -c "
import json
from pathlib import Path

def summarize(name, path):
    p = json.loads(Path(path).read_text())
    if 'layers' in p:
        peak = p.get('peak_layer'), p.get('peak_accuracy')
        print(name, 'peak_layer', peak[0], 'peak_acc', f'{peak[1]:.3f}' if peak[1] else peak[1])
        for r in p['layers']:
            print(f'  layer {r[\"layer_index\"]}: {r[\"cv_accuracy\"]:.3f}')
    else:
        print(name, 'overall_accuracy', p.get('overall_accuracy'),
              'low_amb', p.get('low_ambiguity_accuracy'),
              'high_amb', p.get('high_ambiguity_accuracy'))

summarize('baseline', 'results/probes/baseline_gemma4/gemma4/probes_summary.json')
summarize('finetuned', 'results/probes/finetuned_gemma4/gemma4/probes_summary.json')
"

echo "Probe job complete."
