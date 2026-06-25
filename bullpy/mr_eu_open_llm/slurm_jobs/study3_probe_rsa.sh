#!/usr/bin/env bash
#SBATCH -J msr_s3_probe_rsa
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/study3_probe_rsa_%j.out
#SBATCH -e logs/study3_probe_rsa_%j.err
#
# Study 3: multi-layer linear probes (layers 4,12,25) + per-layer RDMs + baseline vs FT RSA.

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
  results/probes/finetuned_gemma4/gemma4 \
  results/rsa/baseline_gemma4/gemma4 \
  results/rsa/finetuned_gemma4/gemma4 \
  results/stats

echo "=== Baseline multi-layer probes ==="
python -m scripts.probing \
  --activations_dir "${BASELINE_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --output results/probes/baseline_gemma4/gemma4/probes_summary.json

echo "=== Finetuned multi-layer probes ==="
python -m scripts.probing \
  --activations_dir "${FINETUNED_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --output results/probes/finetuned_gemma4/gemma4/probes_summary.json

echo "=== Baseline per-layer RDMs ==="
python -m scripts.rsa \
  --activations_dir "${BASELINE_ACT}" \
  --output_dir results/rsa/baseline_gemma4/gemma4

echo "=== Finetuned per-layer RDMs ==="
python -m scripts.rsa \
  --activations_dir "${FINETUNED_ACT}" \
  --output_dir results/rsa/finetuned_gemma4/gemma4

echo "=== Baseline vs finetuned RDM correlation (per layer) ==="
python -m scripts.rsa \
  --activations_dir "${BASELINE_ACT}" \
  --compare_ft_dir "${FINETUNED_ACT}" \
  --compare_output results/rsa/baseline_vs_finetuned.json

echo "=== Confused pairs for patching (from baseline eval) ==="
python -m scripts.error_analysis \
  --results "${EVAL_JSON}" \
  --output_pairs results/stats/confused_pairs_gemma4.json \
  --top_k 5

echo "=== Layer-wise probe comparison ==="
python -c "
import json
from pathlib import Path

def load_layers(path):
    p = json.loads(Path(path).read_text())
    return {r['layer_index']: r['cv_accuracy'] for r in p.get('layers', [])}

base = load_layers('results/probes/baseline_gemma4/gemma4/probes_summary.json')
ft = load_layers('results/probes/finetuned_gemma4/gemma4/probes_summary.json')
peak_b = json.loads(Path('results/probes/baseline_gemma4/gemma4/peak_layer.json').read_text())
peak_f = json.loads(Path('results/probes/finetuned_gemma4/gemma4/peak_layer.json').read_text())
print('baseline peak layer', peak_b['peak_layer'], 'acc', peak_b['peak_accuracy'])
print('finetuned peak layer', peak_f['peak_layer'], 'acc', peak_f['peak_accuracy'])
for layer in sorted(set(base) | set(ft)):
    b, f = base.get(layer), ft.get(layer)
    drop = (b - f) if b is not None and f is not None else None
    print(f'layer {layer}: baseline={b:.3f} ft={f:.3f} drop={drop:.3f}' if drop is not None else f'layer {layer}: missing')

rsa = json.loads(Path('results/rsa/baseline_vs_finetuned.json').read_text())
print('RSA mean rho baseline vs FT:', rsa.get('mean_rho'))
for row in rsa.get('layers', []):
    print('  layer', row['layer_index'], 'rho', row['spearman_rho_baseline_vs_finetuned'])
"

echo "Study 3 probe+RSA complete."
