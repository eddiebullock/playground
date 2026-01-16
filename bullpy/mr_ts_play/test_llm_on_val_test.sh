#!/bin/bash
# Test LLM on val + test splits (172 samples total, ~$3.32)
# Safer and more cost-effective than full dataset

set -e

echo "=========================================="
echo "Testing LLM on Val + Test Splits"
echo "=========================================="
echo ""

# Combine val + test splits
COMBINED_FILE="data/trial_definitions/eu_emotion_val_test.json"

echo "Combining val + test splits..."
python -c "
import json
from pathlib import Path

all_trials = []
splits = ['val', 'test']

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
python -c "
import json
from pathlib import Path

# Load previous results
prev_results = Path('results/llm_only_eu_emotion_google/results.json')
if prev_results.exists():
    with open(prev_results) as f:
        data = json.load(f)
        prev_samples = len(data.get('predictions', []))
        cost_per_sample = 1.04 / prev_samples
else:
    cost_per_sample = 0.0193

# Calculate for val + test
with open('$COMBINED_FILE') as f:
    data = json.load(f)
    num_samples = len(data.get('trials', data) if isinstance(data, dict) else data)

estimated_cost = cost_per_sample * num_samples

print(f'Dataset size: {num_samples} samples (val + test)')
print(f'Estimated cost: \${estimated_cost:.2f}')
print()
print('⚠️  This is an ESTIMATE based on previous run')
print('   Actual cost may vary by ±20-30%')
"

echo ""
read -p "Do you want to proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cancelled. No charges will be made."
    exit 0
fi

echo ""
echo "=========================================="
echo "Running LLM evaluation on val + test"
echo "=========================================="
echo "Model: Google Gemini 2.5 Flash (64.81% on test set)"
echo "Dataset: 172 samples (val + test combined)"
echo "Estimated cost: ~\$3.32"
echo ""
echo "⚠️  This will make API calls. Monitor your Google Cloud billing."
echo ""

# Ensure config uses Google Gemini
echo "Setting config to use Google Gemini 2.5 Flash..."
python -c "
import yaml
from pathlib import Path

config_path = Path('experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Set to Google (best model)
config['llm']['provider'] = 'google'

# Ensure google section exists
if 'google' not in config['llm']:
    config['llm']['google'] = {}

config['llm']['google']['model'] = 'gemini-2.5-flash'
config['llm']['google']['vision_model'] = 'gemini-2.5-flash'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print('✅ Config set to Google Gemini 2.5 Flash')
"

# Run LLM evaluation using test_llm_only.py (direct approach)
python experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials "$COMBINED_FILE" \
    --output_dir results/llm_only_eu_emotion_google_val_test \
    --num_frames 4

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Evaluation complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/llm_only_eu_emotion_google_val_test/"
    echo ""
    echo "To view results:"
    echo "  python -c \"import json; data=json.load(open('results/llm_only_eu_emotion_google_val_test/results.json')); metrics=data.get('metrics', {}); print(f'Accuracy: {metrics.get(\\\"overall_accuracy\\\", 0)*100:.1f}%')\""
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi
