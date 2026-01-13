#!/bin/bash
# Simple script to test all LLM providers - manual config editing required
#
# This script provides commands to test each provider.
# You'll need to manually edit the config file between runs.

set -e

CONFIG_FILE="experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml"
SCRIPT="experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py"

echo "============================================================"
echo "Testing All LLM Providers on EU-Emotion Dataset"
echo "============================================================"
echo ""
echo "This script will guide you through testing each provider."
echo "You'll need to edit the config file between runs."
echo ""
echo "Providers to test:"
echo "  1. OpenAI (GPT-4o-mini) - ~\$0.02"
echo "  2. Anthropic (Claude 3.5 Sonnet) - ~\$0.11"
echo "  3. Google (Gemini 1.5 Pro) - ~\$0.03"
echo ""
echo "Total estimated cost: ~\$0.16"
echo ""

# Test OpenAI
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1/3: OpenAI (GPT-4o-mini)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Make sure config has: provider: \"openai\""
echo "Running OpenAI test..."
echo ""
python "${SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --dataset eu_emotion \
    --use_cache || {
    echo "❌ OpenAI test failed"
    exit 1
}
echo "✅ OpenAI test completed"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PAUSE: Please edit ${CONFIG_FILE}"
echo "Change: provider: \"anthropic\""
echo "Press Enter when ready to continue..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read

# Test Anthropic
echo "Test 2/3: Anthropic (Claude 3.5 Sonnet)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Anthropic test..."
echo ""
python "${SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --dataset eu_emotion \
    --use_cache || {
    echo "❌ Anthropic test failed"
    exit 1
}
echo "✅ Anthropic test completed"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PAUSE: Please edit ${CONFIG_FILE}"
echo "Change: provider: \"google\""
echo "Press Enter when ready to continue..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read

# Test Google
echo "Test 3/3: Google (Gemini 1.5 Pro)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Google test..."
echo ""
python "${SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --dataset eu_emotion \
    --use_cache || {
    echo "❌ Google test failed"
    exit 1
}
echo "✅ Google test completed"
echo ""

echo "============================================================"
echo "All Provider Tests Completed!"
echo "============================================================"
echo ""
echo "Results saved to: results/llm_augmented_eu_emotion_weighted_average/"
echo ""
