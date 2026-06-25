#!/usr/bin/env bash
#SBATCH -J msr_s3_probe_rsa_4afc
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/study3_probe_rsa_4afc_%j.out
#SBATCH -e logs/study3_probe_rsa_4afc_%j.err
#
# Study 3 probes + RSA on 4AFC-aligned activations (aligned with patching).

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

MODEL="${MODEL:-gemma4}"
if [[ "${MODEL}" == "gemma4" ]]; then
  MODALITY="${MODALITY:-multimodal}"
else
  MODALITY="${MODALITY:-video_only}"
fi

DEFAULT_BASELINE_EVAL="results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_${MODALITY}_fps1_cap16_two_stage_seed42.json"
DEFAULT_FT_EVAL="results/finetune/eu_post_ft/eval_v2_eu_emotions_${MODEL}_${MODALITY}_finetuned_seed42.json"

EVAL_JSON="${EVAL_JSON:-${DEFAULT_BASELINE_EVAL}}"
BASELINE_ACT="${BASELINE_ACT:-results/activations/baseline_${MODEL}_4afc/${MODEL}}"
FINETUNED_ACT="${FINETUNED_ACT:-results/activations/finetuned_${MODEL}_4afc/${MODEL}}"

PROBE_BASE="results/probes/baseline_${MODEL}_4afc/${MODEL}"
PROBE_FT="results/probes/finetuned_${MODEL}_4afc/${MODEL}"
RSA_BASE="results/rsa/baseline_${MODEL}_4afc/${MODEL}"
RSA_FT="results/rsa/finetuned_${MODEL}_4afc/${MODEL}"
RSA_COMPARE="results/rsa/baseline_vs_finetuned_${MODEL}_4afc.json"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p logs "${PROBE_BASE}" "${PROBE_FT}" "${RSA_BASE}" "${RSA_FT}" results/stats

echo "=== Baseline multi-layer probes (4AFC activations) ==="
python -m scripts.probing \
  --activations_dir "${BASELINE_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --output "${PROBE_BASE}/probes_summary.json"

echo "=== Finetuned multi-layer probes (4AFC activations) ==="
python -m scripts.probing \
  --activations_dir "${FINETUNED_ACT}" \
  --eval_json "${EVAL_JSON}" \
  --output "${PROBE_FT}/probes_summary.json"

echo "=== Baseline per-layer RDMs ==="
python -m scripts.rsa \
  --activations_dir "${BASELINE_ACT}" \
  --output_dir "${RSA_BASE}"

echo "=== Finetuned per-layer RDMs ==="
python -m scripts.rsa \
  --activations_dir "${FINETUNED_ACT}" \
  --output_dir "${RSA_FT}"

echo "=== Baseline vs finetuned RDM correlation ==="
python -m scripts.rsa \
  --activations_dir "${BASELINE_ACT}" \
  --compare_ft_dir "${FINETUNED_ACT}" \
  --compare_output "${RSA_COMPARE}"

echo "=== Layer-wise probe comparison ==="
python -c "
import json
from pathlib import Path

def load_layers(path):
    p = json.loads(Path(path).read_text())
    return {r['layer_index']: r['cv_accuracy'] for r in p.get('layers', [])}

base = load_layers('${PROBE_BASE}/probes_summary.json')
ft = load_layers('${PROBE_FT}/probes_summary.json')
peak_b = json.loads(Path('${PROBE_BASE}/peak_layer.json').read_text())
peak_f = json.loads(Path('${PROBE_FT}/peak_layer.json').read_text())
print('baseline peak layer', peak_b['peak_layer'], 'acc', peak_b['peak_accuracy'])
print('finetuned peak layer', peak_f['peak_layer'], 'acc', peak_f['peak_accuracy'])
for layer in sorted(set(base) | set(ft)):
    b, f = base.get(layer), ft.get(layer)
    drop = (b - f) if b is not None and f is not None else None
    print(f'layer {layer}: baseline={b:.3f} ft={f:.3f} drop={drop:.3f}' if drop is not None else f'layer {layer}: missing')

rsa = json.loads(Path('${RSA_COMPARE}').read_text())
print('RSA mean rho baseline vs FT:', rsa.get('mean_rho'))
for row in rsa.get('layers', []):
    print('  layer', row['layer_index'], 'rho', row['spearman_rho_baseline_vs_finetuned'])
"

echo "Study 3 probe+RSA (4AFC) complete."
