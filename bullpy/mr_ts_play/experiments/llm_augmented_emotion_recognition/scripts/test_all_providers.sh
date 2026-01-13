#!/bin/bash
# Test all three LLM providers (OpenAI, Anthropic, Google) on EU-Emotion dataset
#
# This script runs the LLM-augmented experiment with each provider sequentially
# Results are saved separately for each provider

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

CONFIG_FILE="experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml"
SCRIPT="experiments/llm_augmented_emotion_recognition/scripts/run_llm_augmented_experiment.py"

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Testing All LLM Providers on EU-Emotion Dataset${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""
echo -e "${BLUE}Providers to test:${NC}"
echo "  1. OpenAI (GPT-4o-mini) - ~\$0.02"
echo "  2. Anthropic (Claude 3.5 Sonnet) - ~\$0.11"
echo "  3. Google (Gemini 1.5 Pro) - ~\$0.03"
echo ""
echo -e "${BLUE}Total estimated cost: ~\$0.16${NC}"
echo -e "${BLUE}Note: Subsequent runs use cache (free)${NC}"
echo ""

# Function to update provider in config
update_provider() {
    local provider=$1
    local model=$2
    
    # Use Python to update YAML (more reliable than sed)
    python3 << EOF
import yaml
import sys

config_file = '${CONFIG_FILE}'
new_provider = '${provider}'
new_model = '${model}'

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['llm']['provider'] = new_provider

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config: provider = {new_provider}, model = {new_model}")
EOF
}

# Test OpenAI
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 1/3: OpenAI (GPT-4o-mini)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
update_provider "openai" "gpt-4o-mini"
python "${SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --dataset eu_emotion \
    --use_cache || {
    echo -e "${RED}❌ OpenAI test failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ OpenAI test completed${NC}"
echo ""

# Test Anthropic
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 2/3: Anthropic (Claude 3.5 Sonnet)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
update_provider "anthropic" "claude-3-5-sonnet-20241022"
python "${SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --dataset eu_emotion \
    --use_cache || {
    echo -e "${RED}❌ Anthropic test failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Anthropic test completed${NC}"
echo ""

# Test Google
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 3/3: Google (Gemini 1.5 Pro)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
update_provider "google" "gemini-1.5-pro"
python "${SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --dataset eu_emotion \
    --use_cache || {
    echo -e "${RED}❌ Google test failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Google test completed${NC}"
echo ""

# Reset to OpenAI (default)
update_provider "openai" "gpt-4o-mini"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All Provider Tests Completed!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${BLUE}Results saved to:${NC}"
echo "  results/llm_augmented_eu_emotion_weighted_average/"
echo ""
echo -e "${BLUE}To compare results, check:${NC}"
echo "  - Individual model directories in results/"
echo "  - Comparison reports generated for each provider"
echo ""
