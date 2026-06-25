#!/usr/bin/env bash
set -euo pipefail

# Submit EU-Emotions baseline jobs (118 trials by default).
#
# Default profile (video + audio where supported):
#   qwen2vl, llavanext  -> video_only, stage both (vision-only VLMs)
#   gemma4              -> multimodal, stage both (stage-1 entropy + stage-2 4AFC)
#
# Only gemma4 has MODEL_AUDIO_CAPABILITIES=True; other models will fail if forced to
# audio_only/multimodal (see config.py).
#
# Usage:
#   bash slurm_jobs/submit_baselines.sh
#
# Optional overrides:
#   MAX_TRIALS=5 bash slurm_jobs/submit_baselines.sh
#   MULTIMODAL_ONLY=1 bash slurm_jobs/submit_baselines.sh   # gemma4 multimodal only
#   CONDITION=multimodal MODEL=gemma4 sbatch slurm_jobs/baseline_eval.sh

MODELS=("qwen2vl" "llavanext" "gemma4")
AUDIO_CAPABLE_MODELS=("gemma4")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_JOB_SCRIPT="${SCRIPT_DIR}/baseline_eval.sh"
DATASET="${DATASET:-eu_emotions}"
MAX_FRAMES="${MAX_FRAMES:-16}"
FPS="${FPS:-1}"
MAX_TRIALS="${MAX_TRIALS:-118}"
TEMPERATURE="${TEMPERATURE:-0.1}"
DATA_ROOT="${DATA_ROOT:-data/eu_emotions_118}"
MANIFEST="${MANIFEST:-data/eu_emotions_118_manifest.json}"
MULTIMODAL_ONLY="${MULTIMODAL_ONLY:-0}"
VISION_CONDITION="${VISION_CONDITION:-video_only}"
VISION_STAGE="${VISION_STAGE:-both}"
GEMMA_CONDITION="${GEMMA_CONDITION:-multimodal}"
GEMMA_STAGE="${GEMMA_STAGE:-both}"

_model_supports_audio() {
  local m="$1"
  local cap
  for cap in "${AUDIO_CAPABLE_MODELS[@]}"; do
    if [[ "${cap}" == "${m}" ]]; then
      return 0
    fi
  done
  return 1
}

_resolve_condition() {
  local model="$1"
  if _model_supports_audio "${model}"; then
    echo "${GEMMA_CONDITION}"
  else
    echo "${VISION_CONDITION}"
  fi
}

_resolve_stage() {
  local model="$1"
  if _model_supports_audio "${model}"; then
    echo "${GEMMA_STAGE}"
  else
    echo "${VISION_STAGE}"
  fi
}

_submit_one() {
  local model="$1"
  local condition stage
  condition="$(_resolve_condition "${model}")"
  stage="$(_resolve_stage "${model}")"
  echo "Submitting baseline for ${model} (condition=${condition}, stage=${stage})..."
  sbatch \
    --job-name="baseline_${model}" \
    --export=MODEL="${model}",DATASET="${DATASET}",CONDITION="${condition}",STAGE="${stage}",MAX_FRAMES="${MAX_FRAMES}",FPS="${FPS}",MAX_TRIALS="${MAX_TRIALS}",TEMPERATURE="${TEMPERATURE}",DATA_ROOT="${DATA_ROOT}",MANIFEST="${MANIFEST}" \
    "${BASELINE_JOB_SCRIPT}"
}

if [[ "${MULTIMODAL_ONLY}" == "1" ]]; then
  echo "MULTIMODAL_ONLY=1: submitting audio-capable models only (${AUDIO_CAPABLE_MODELS[*]})..."
  for model in "${AUDIO_CAPABLE_MODELS[@]}"; do
    _submit_one "${model}"
  done
else
  for model in "${MODELS[@]}"; do
    _submit_one "${model}"
  done
fi

echo "All baseline jobs submitted."
