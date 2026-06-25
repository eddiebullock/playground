#!/usr/bin/env bash
# Submit Tier-C replication: MR LoRA finetune + EU eval + 4AFC probes/patching.
#
# NOT just finetune_full — this chains the full Study 3 downstream pipeline.
#
# Usage (from study2 root on HPC):
#   MODEL=qwen2vl bash slurm_jobs/study3_tierC_submit.sh
#   MODEL=llavanext bash slurm_jobs/study3_tierC_submit.sh
#
# Prerequisites:
#   - Baseline EU eval JSON exists for MODEL (run submit_baselines.sh if missing)
#   - Model weights on HPC under models/${MODEL}
#
# After finetune (~2-3h Gemma4 multimodal; video-only VLMs often faster), downstream runs automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-qwen2vl}"

if [[ "${MODEL}" == "gemma4" ]]; then
  MODALITY=multimodal
else
  MODALITY=video_only
fi

BASELINE_EVAL="results/baseline/eu_emotions/${MODEL}/eval_v2_eu_emotions_${MODEL}_${MODALITY}_fps1_cap16_two_stage_seed42.json"
if [[ ! -f "${BASELINE_EVAL}" ]]; then
  echo "WARNING: baseline eval not found: ${BASELINE_EVAL}"
  echo "Submit baselines first: bash slurm_jobs/submit_baselines.sh"
fi

echo "=== Tier C: MODEL=${MODEL}, MODALITY=${MODALITY} ==="

echo "Submitting MR LoRA finetune..."
FT_JID=$(sbatch --parsable --export=MODEL="${MODEL}" "${SCRIPT_DIR}/finetune_full.sh")
ADAPTER="results/finetune/full_runs/${MODEL}/run_${FT_JID}/adapter_final"
echo "  finetune job: ${FT_JID}"
echo "  expected adapter: ${ADAPTER}"

echo "Submitting baseline 4AFC activation extract (parallel with finetune)..."
BASE_ACT_JID=$(sbatch --parsable \
  --export=MODEL="${MODEL}",CONDITION="baseline_${MODEL}_4afc",MODALITY="${MODALITY}",PROMPT_MODE=4afc,POOLING=last_token \
  "${SCRIPT_DIR}/activation_extract_4afc.sh")

echo "Submitting post-finetune EU eval (after finetune)..."
EVAL_JID=$(sbatch --parsable --dependency=afterok:"${FT_JID}" \
  --export=MODEL="${MODEL}",CONDITION="${MODALITY}",LORA_ADAPTER="${ADAPTER}" \
  "${SCRIPT_DIR}/post_finetune_eu_eval.sh")

echo "Submitting finetuned 4AFC activation extract (after finetune)..."
FT_ACT_JID=$(sbatch --parsable --dependency=afterok:"${FT_JID}" \
  --export=MODEL="${MODEL}",CONDITION="finetuned_${MODEL}_4afc",MODALITY="${MODALITY}",PROMPT_MODE=4afc,POOLING=last_token,CHECKPOINT="${ADAPTER}" \
  "${SCRIPT_DIR}/activation_extract_4afc.sh")

echo "Submitting probe+RSA (after both extractions)..."
PROBE_JID=$(sbatch --parsable --dependency=afterok:"${BASE_ACT_JID}":"${FT_ACT_JID}" \
  --export=MODEL="${MODEL}",MODALITY="${MODALITY}" \
  "${SCRIPT_DIR}/study3_probe_rsa_4afc.sh")

PATCH_MAX_TRIALS="${PATCH_MAX_TRIALS:-0}"
echo "Submitting patching (after probes + eval)..."
PATCH_JID=$(sbatch --parsable --dependency=afterok:"${PROBE_JID}":"${EVAL_JID}" \
  --export=MODEL="${MODEL}",MODALITY="${MODALITY}",LORA_ADAPTER="${ADAPTER}",MAX_TRIALS="${PATCH_MAX_TRIALS}" \
  "${SCRIPT_DIR}/patching_gemma4_v2_4afc.sh")

echo ""
echo "Tier C pipeline submitted."
echo "  finetune:        ${FT_JID}"
echo "  baseline acts:   ${BASE_ACT_JID}"
echo "  post-FT EU eval: ${EVAL_JID} (after ${FT_JID})"
echo "  finetuned acts:  ${FT_ACT_JID} (after ${FT_JID})"
echo "  probe+RSA:       ${PROBE_JID}"
echo "  patching:        ${PATCH_JID}"
echo ""
echo "Monitor: sacct -j ${FT_JID},${EVAL_JID},${PROBE_JID},${PATCH_JID}"
echo "After finetune: verify B0 with verify_finetune_eval.py using ${BASELINE_EVAL}"
