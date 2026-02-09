#!/usr/bin/env bash
# Run EU emotion test on Gemini 3 Flash, GPT-5 Mini, and Anthropic Opus 4.5.
# Paths match the setup used for the Gemini 2.5 Flash EU run (same data root and audio dir).
# Run from project root: mr_ts_play
#
# Progress: each run logs "Processing trial X/118" then "RESULTS" and "Accuracy: X.XX%".
# To save a log: ./run_eu_three_models.sh 2>&1 | tee results/eu_three_models.log
# To run in background: nohup ./run_eu_three_models.sh > results/eu_three_models.log 2>&1 &
#   then: tail -f results/eu_three_models.log

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# EU emotion paths (from llm_config.yaml / MULTIMODAL_SETUP.md – same as 80% Gemini 2.5 Flash run)
EU_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
EU_AUDIO_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio/Fixed - amplified volume"
TRIALS="data/trial_definitions/eu_emotion_test_final.json"

mkdir -p results

echo "Project root: $PROJECT_ROOT"
echo "Data root:    $EU_DATA_ROOT"
echo "Audio dir:    $EU_AUDIO_DIR"
echo "Trials:       $TRIALS"
echo ""

# 1. Gemini 3 Flash (video + audio)
echo "=============================================="
echo "1. Gemini 3 Flash — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$EU_DATA_ROOT" \
  --audio-dir "$EU_AUDIO_DIR" \
  --output-dir results/eu_emotion_gemini3_flash \
  --provider google \
  --model gemini-3-flash-preview \
  --use-audio
echo "1. Gemini 3 Flash — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 2. GPT-5 Mini (video + audio)
echo "=============================================="
echo "2. GPT-5 Mini — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$EU_DATA_ROOT" \
  --audio-dir "$EU_AUDIO_DIR" \
  --output-dir results/eu_emotion_gpt5_mini \
  --provider openai \
  --model gpt-5-mini \
  --use-audio
echo "2. GPT-5 Mini — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 3. Anthropic Opus 4.5 (video only; Claude API has no audio input)
echo "=============================================="
echo "3. Anthropic Opus 4.5 — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$EU_DATA_ROOT" \
  --audio-dir "$EU_AUDIO_DIR" \
  --output-dir results/eu_emotion_opus_4_5 \
  --provider anthropic \
  --model claude-opus-4-5 \
  --use-audio
echo "3. Anthropic Opus 4.5 — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "Done. Results:"
echo "  Gemini 3 Flash:    results/eu_emotion_gemini3_flash/summary.json"
echo "  GPT-5 Mini:       results/eu_emotion_gpt5_mini/summary.json"
echo "  Anthropic Opus 4.5: results/eu_emotion_opus_4_5/summary.json"
