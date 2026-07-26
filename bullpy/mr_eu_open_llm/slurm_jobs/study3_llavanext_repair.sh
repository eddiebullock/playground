#!/usr/bin/env bash
# Resubmit LLaVA post-FT eval + full patching after PeftModel generation fix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=llavanext
MODALITY=video_only
ADAPTER="results/finetune/full_runs/llavanext/run_30994225/adapter_final"

echo "=== Re-eval LLaVA with PeftModel generate ==="
EVAL_JID=$(sbatch --parsable \
  --export=MODEL="${MODEL}",CONDITION="${MODALITY}",LORA_ADAPTER="${ADAPTER}" \
  "${SCRIPT_DIR}/post_finetune_eu_eval.sh")
echo "  post-FT eval: ${EVAL_JID}"

echo "=== Full LLaVA patching (after eval) ==="
PATCH_JID=$(sbatch --parsable --dependency=afterok:"${EVAL_JID}" \
  --time=03:00:00 \
  --export=MODEL="${MODEL}",MODALITY="${MODALITY}",LORA_ADAPTER="${ADAPTER}",MAX_TRIALS=0 \
  "${SCRIPT_DIR}/patching_gemma4_v2_4afc.sh")
echo "  patching: ${PATCH_JID}"
