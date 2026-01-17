# Next Commands for Optimization

## Current Status

| Variation | Accuracy | Improvement | Status |
|-----------|----------|------------|--------|
| Baseline | 68.52% | - | ✅ Done |
| Explicit Distinctions | 70.37% | +1.85% | ✅ Done |
| **Intensity-Aware** | **74.07%** | **+5.55%** | ✅ Done |
| Temporal Analysis | TBD | TBD | ⏳ Next |
| Combined | TBD | TBD | ⏳ Pending |

**Best so far: Intensity-Aware (74.07%)** 🎉

## Next Commands

### 1. Test Temporal Analysis (Next)

```bash
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_temporal_analysis \
    --num_frames 4 \
    --prompt_variation temporal_analysis
```

**Expected time**: ~5-10 minutes  
**Expected cost**: ~$1.40

### 2. Test Combined (All Enhancements)

```bash
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_combined \
    --num_frames 4 \
    --prompt_variation combined
```

**Expected time**: ~5-10 minutes  
**Expected cost**: ~$1.40

### 3. Compare All Results

After both are done, run:

```bash
python3.13 -c "
import json
from pathlib import Path

variations = {
    'baseline': 'results/llm_optimization/baseline_validation/results.json',
    'explicit_distinctions': 'results/llm_optimization/variation_explicit_distinctions/results.json',
    'intensity_aware': 'results/llm_optimization/variation_intensity_aware/results.json',
    'temporal_analysis': 'results/llm_optimization/variation_temporal_analysis/results.json',
    'combined': 'results/llm_optimization/variation_combined/results.json'
}

print('='*80)
print('FINAL COMPARISON - VALIDATION SET')
print('='*80)
print()

results = {}
for name, path in variations.items():
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
            acc = data.get('metrics', {}).get('overall_accuracy', 0) * 100
            correct = data.get('metrics', {}).get('num_correct', 0)
            total = data.get('metrics', {}).get('num_total', 0)
            results[name] = acc
            print(f'{name:25} {acc:6.2f}% ({correct}/{total})')

print()
print('='*80)
print('BEST PERFORMING:')
print('='*80)
if results:
    best = max(results.items(), key=lambda x: x[1])
    print(f'{best[0]:25} {best[1]:6.2f}%')
    print()
    print(f'Use this for final test evaluation: --prompt_variation {best[0]}')
"
```

### 4. Final Test Evaluation

After selecting best prompt:

```bash
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_test_final.json \
    --output_dir results/llm_final_evaluation \
    --num_frames 4 \
    --prompt_variation [best_variation_from_step_3]
```

## Expected Timeline

- Temporal Analysis: ~5-10 minutes
- Combined: ~5-10 minutes
- Comparison: 1 minute
- Final test: ~10-15 minutes

**Total remaining**: ~20-35 minutes

## Expected Results

- Temporal Analysis: 72-74% (expected)
- Combined: 75-77% (expected - combines all enhancements)
- Final test: 73-75% (expected, slightly lower than validation)

## Cost Remaining

- Temporal Analysis: ~$1.40
- Combined: ~$1.40
- Final test: ~$3.04
- **Total remaining**: ~$5.84
