#!/usr/bin/env bash
# Run EU emotion test on Gemini 3 Pro and GPT-5 (full).
# Same paths and trials as other EU runs. Run from project root: mr_ts_play
#
# Cost: ~$0.70 (Gemini 3 Pro) + ~$0.48 (GPT-5) ≈ $1.20 total for 118 trials.
# Progress: "Processing trial X/118" then "RESULTS" and "Accuracy: X.XX%".

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

EU_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
EU_AUDIO_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume"
TRIALS="data/trial_definitions/eu_emotion_test_final.json"

mkdir -p results

echo "Project root: $PROJECT_ROOT"
echo "Data root:    $EU_DATA_ROOT"
echo "Trials:       $TRIALS (118 trials)"
echo ""

# 1. Gemini 3 Pro (video + audio)
echo "=============================================="
echo "1. Gemini 3 Pro — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$EU_DATA_ROOT" \
  --audio-dir "$EU_AUDIO_DIR" \
  --output-dir results/eu_emotion_gemini3_pro \
  --provider google \
  --model gemini-3-pro-preview \
  --use-audio
echo "1. Gemini 3 Pro — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 2. GPT-5 (full; video-only for OpenAI unless using gpt-audio)
echo "=============================================="
echo "2. GPT-5 — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$EU_DATA_ROOT" \
  --audio-dir "$EU_AUDIO_DIR" \
  --output-dir results/eu_emotion_gpt5 \
  --provider openai \
  --model gpt-5 \
  --use-audio
echo "2. GPT-5 — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "Done. Results:"
echo "  Gemini 3 Pro: results/eu_emotion_gemini3_pro/summary.json"
echo "  GPT-5:       results/eu_emotion_gpt5/summary.json"
