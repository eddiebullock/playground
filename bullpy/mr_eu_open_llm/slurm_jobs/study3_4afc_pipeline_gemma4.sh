#!/usr/bin/env bash
# Submit Study 3 Tier-B pipeline: 4AFC extract -> probe/RSA -> patching (Gemma4).
#
# Usage (from study2 root on HPC):
#   LORA_ADAPTER=results/finetune/full_runs/gemma4/run_30364652/adapter_final \
#     bash slurm_jobs/study3_4afc_pipeline_gemma4.sh
#
# Optional: MAX_TRIALS=30 on patching for a faster causal check.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-gemma4}"
LORA_ADAPTER="${LORA_ADAPTER:?Set LORA_ADAPTER to adapter_final path}"
PATCH_MAX_TRIALS="${PATCH_MAX_TRIALS:-0}"

echo "Submitting 4AFC baseline extraction..."
BASE_JID=$(sbatch --parsable \
  --export=MODEL="${MODEL}",CONDITION="baseline_${MODEL}_4afc",MODALITY=multimodal,PROMPT_MODE=4afc,POOLING=last_token \
  "${SCRIPT_DIR}/activation_extract_4afc.sh")

echo "Submitting 4AFC finetuned extraction (after baseline starts; needs adapter only)..."
FT_ACT_JID=$(sbatch --parsable \
  --export=MODEL="${MODEL}",CONDITION="finetuned_${MODEL}_4afc",MODALITY=multimodal,PROMPT_MODE=4afc,POOLING=last_token,CHECKPOINT="${LORA_ADAPTER}" \
  "${SCRIPT_DIR}/activation_extract_4afc.sh")

echo "Submitting probe+RSA (after both extractions)..."
PROBE_JID=$(sbatch --parsable --dependency=afterok:"${BASE_JID}":"${FT_ACT_JID}" \
  --export=MODEL="${MODEL}",MODALITY=multimodal \
  "${SCRIPT_DIR}/study3_probe_rsa_4afc.sh")

echo "Submitting patching (after probes)..."
PATCH_JID=$(sbatch --parsable --dependency=afterok:"${PROBE_JID}" \
  --export=MODEL="${MODEL}",MODALITY=multimodal,LORA_ADAPTER="${LORA_ADAPTER}",MAX_TRIALS="${PATCH_MAX_TRIALS}" \
  "${SCRIPT_DIR}/patching_gemma4_v2_4afc.sh")

echo "Submitted jobs:"
echo "  baseline extract: ${BASE_JID}"
echo "  finetuned extract: ${FT_ACT_JID}"
echo "  probe+RSA: ${PROBE_JID} (depends on ${BASE_JID}, ${FT_ACT_JID})"
echo "  patching: ${PATCH_JID} (depends on ${PROBE_JID})"
