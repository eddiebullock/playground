#!/bin/bash
# Test LLM on full EU emotions dataset (train + val + test combined)
# Since LLMs are zero-shot, testing on full dataset is valid and gives more reliable results

set -e

echo "=========================================="
echo "Testing LLM on Full EU Emotions Dataset"
echo "=========================================="
echo ""

# Combine all splits into one test set
COMBINED_FILE="data/trial_definitions/eu_emotion_full.json"

echo "Combining train + val + test splits..."
python -c "
import json
from pathlib import Path

all_trials = []
splits = ['train', 'val', 'test']

for split in splits:
    file_path = Path(f'data/trial_definitions/eu_emotion_{split}.json')
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
            trials = data.get('trials', data) if isinstance(data, dict) else data
            all_trials.extend(trials)
            print(f'  Added {split}: {len(trials)} samples')

# Save combined set
output_path = Path('$COMBINED_FILE')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump({'trials': all_trials}, f, indent=2)

print(f'Total: {len(all_trials)} samples')
print(f'Saved to: {output_path}')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to combine splits"
    exit 1
fi

echo ""
echo "=========================================="
echo "COST ESTIMATION"
echo "=========================================="
python estimate_llm_cost.py

echo ""
read -p "Do you want to proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cancelled. No charges will be made."
    exit 0
fi

echo ""
echo "=========================================="
echo "Running LLM evaluation on full dataset"
echo "=========================================="
echo "Model: Google Gemini 2.5 Flash (64.81% on test set)"
echo "Dataset: 546 samples (train + val + test combined)"
echo "Estimated cost: ~\$1-2 (see estimate above)"
echo ""
echo "⚠️  This will make API calls. Monitor your Google Cloud billing."
echo ""

# Ensure config uses Google Gemini (best model)
echo "Setting config to use Google Gemini 2.5 Flash..."
python -c "
import yaml
from pathlib import Path

config_path = Path('experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Set to Google (best model)
config['provider'] = 'google'
config['google']['model'] = 'gemini-2.5-flash'
config['google']['vision_model'] = 'gemini-2.5-flash'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('✅ Config set to Google Gemini 2.5 Flash')
"

# Run LLM evaluation (only Google Gemini)
python experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials "$COMBINED_FILE" \
    --output_dir results/llm_only_eu_emotion_google_full

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Evaluation complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/llm_only_eu_emotion_google_full/"
    echo ""
    echo "To view results:"
    echo "  python -c \"import json; data=json.load(open('results/llm_only_eu_emotion_google_full/results.json')); print(f'Accuracy: {data.get(\\\"metrics\\\", {}).get(\\\"overall_accuracy\\\", 0)*100:.1f}%')\""
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi
