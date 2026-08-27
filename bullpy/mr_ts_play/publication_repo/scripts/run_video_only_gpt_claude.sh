#!/usr/bin/env bash
# Video-only reruns for GPT-5, GPT-5 Mini, and Claude Opus 4.5 (publication_repo pipeline).
#
# Uses the same trial JSONs and cache/results layout as the Gemini Flash full_run.
# Run from repo root (mr_ts_play). Requires OPENAI_API_KEY and ANTHROPIC_API_KEY in .env.
#
# Parallel (recommended — one model per terminal, different providers):
#   Terminal 1: ./publication_repo/scripts/run_video_only_gpt_claude.sh --model gpt-5
#   Terminal 2: ./publication_repo/scripts/run_video_only_gpt_claude.sh --model gpt-5-mini
#   Terminal 3: ./publication_repo/scripts/run_video_only_gpt_claude.sh --model claude-opus-4-5
#
# Sequential (all six legs):
#   ./publication_repo/scripts/run_video_only_gpt_claude.sh --all

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR_EU="${DATA_DIR_EU:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions}"
DATA_DIR_MR="${DATA_DIR_MR:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions}"
TRIALS_EU="${TRIALS_EU:-data/trial_definitions/eu_emotion_test_final.json}"
TRIALS_MR="${TRIALS_MR:-data/trial_definitions/mindreading_emotions_test.json}"
CACHE_DIR="${CACHE_DIR:-publication_repo/cache/full_run}"
RESULTS_DIR="${RESULTS_DIR:-publication_repo/results/full_run}"
SEED="${SEED:-42}"

MODEL=""
ALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --all) ALL=true; shift ;;
    --data-dir-eu) DATA_DIR_EU="$2"; shift 2 ;;
    --data-dir-mr) DATA_DIR_MR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ "$ALL" == true ]]; then
  MODELS=(gpt-5 gpt-5-mini claude-opus-4-5)
elif [[ -n "$MODEL" ]]; then
  MODELS=("$MODEL")
else
  echo "Pass --model <name> or --all"
  exit 2
fi

RUNNER="$REPO_ROOT/publication_repo/scripts/run_evaluation_detached.sh"

for m in "${MODELS[@]}"; do
  echo ""
  echo "======== EU video_only | $m ========"
  "$RUNNER" \
    --model "$m" \
    --dataset eu_emotion \
    --condition video_only \
    --trials_file "$TRIALS_EU" \
    --data_dir "$DATA_DIR_EU" \
    --cache_dir "$CACHE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    --log "$RESULTS_DIR/eu_video_only_${m}.log"

  echo "Waiting for EU run to finish..."
  while pgrep -f "run_evaluation.py.*--model $m.*eu_emotion" >/dev/null 2>&1; do
    sleep 30
  done

  echo ""
  echo "======== Mindreading video_only | $m ========"
  "$RUNNER" \
    --model "$m" \
    --dataset mindreading \
    --condition video_only \
    --trials_file "$TRIALS_MR" \
    --data_dir "$DATA_DIR_MR" \
    --cache_dir "$CACHE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --seed "$SEED" \
    --log "$RESULTS_DIR/mindreading_video_only_${m}.log"

  echo "Waiting for Mindreading run to finish..."
  while pgrep -f "run_evaluation.py.*--model $m.*mindreading" >/dev/null 2>&1; do
    sleep 60
  done
done

echo ""
echo "Done. Check summaries in $RESULTS_DIR"
echo "Then: cd publication_repo && python analysis/run_study_analysis.py"
