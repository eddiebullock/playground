#!/usr/bin/env bash
# Run MindReading emotion experiment with Anthropic Opus 4.5, GPT-5, and Gemini 3 Pro.
# Same trial set as video-only run (mindreading_emotions_test.json, 1263 trials; ~583 process successfully).
# Run from project root: mr_ts_play
#
# Paths: /Volumes/MindReading/Emotions (data + audio). Adjust if your mount differs.
# Cost (worst case): ~$17 total for all three (see MINDREADING_THREE_MODELS_COST.md).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

MR_DATA_ROOT="/Volumes/MindReading/Emotions"
MR_AUDIO_BASE="/Volumes/MindReading/Emotions/Audio"
TRIALS="data/trial_definitions/mindreading_emotions_test.json"

mkdir -p results

echo "Project root:    $PROJECT_ROOT"
echo "Data root:       $MR_DATA_ROOT"
echo "Audio base:      $MR_AUDIO_BASE"
echo "Trials:          $TRIALS (1263 trials; ~583 typically process due to decode)"
echo ""

# 1. Gemini 3 Pro (video + audio)
echo "=============================================="
echo "1. Gemini 3 Pro — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$MR_DATA_ROOT" \
  --audio-base-dir "$MR_AUDIO_BASE" \
  --audio-folder 1 \
  --output-dir results/mindreading_gemini3_pro \
  --provider google \
  --model gemini-3-pro-preview \
  --use-audio \
  --skip-failed
echo "1. Gemini 3 Pro — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 2. GPT-5 (video-only; no audio in this setup)
echo "=============================================="
echo "2. GPT-5 — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$MR_DATA_ROOT" \
  --audio-base-dir "$MR_AUDIO_BASE" \
  --audio-folder 1 \
  --output-dir results/mindreading_gpt5 \
  --provider openai \
  --model gpt-5 \
  --use-audio \
  --skip-failed
echo "2. GPT-5 — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 3. Anthropic Opus 4.5 (video-only; Claude API has no audio input)
echo "=============================================="
echo "3. Anthropic Opus 4.5 — started $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
python experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py \
  --trial-definitions "$TRIALS" \
  --data-root "$MR_DATA_ROOT" \
  --audio-base-dir "$MR_AUDIO_BASE" \
  --audio-folder 1 \
  --output-dir results/mindreading_opus_4_5 \
  --provider anthropic \
  --model claude-opus-4-5 \
  --use-audio \
  --skip-failed
echo "3. Anthropic Opus 4.5 — finished $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "Done. Results:"
echo "  Gemini 3 Pro:     results/mindreading_gemini3_pro/summary.json"
echo "  GPT-5:            results/mindreading_gpt5/summary.json"
echo "  Anthropic Opus 4.5: results/mindreading_opus_4_5/summary.json"
