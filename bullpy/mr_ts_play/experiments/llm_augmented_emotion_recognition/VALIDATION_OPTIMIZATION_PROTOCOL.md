# Validation-Based Optimization Protocol (Option A)

## Scientific Rigor ✅

This approach is **scientifically sound** because:
- ✅ Based on **validation set** errors (used for optimization)
- ❌ NOT based on test set errors
- ✅ Will be tested on validation set first
- ✅ Only run on test set if validation improves

## Steps Completed

### Step 1: Analyze Validation Set Errors ✅

**Validation set error patterns:**
- Afraid → Surprised (2 errors)
- Interested → Neutral (2 errors)
- Afraid → Worried (1 error)
- Disgusted Low Intensity → Sneaky (1 error)
- Kind → Happy Low Intensity (1 error)
- Plus several other single errors

### Step 2: Create Validation-Optimized Prompt ✅

Created `validation_optimized` prompt variation that:
- Addresses validation set error patterns
- Includes intensity awareness (from best performing variation)
- Adds specific distinctions for confusing pairs
- Scientifically sound (based on validation, not test)

## Next Steps

### Step 3: Test on Validation Set

```bash
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_validation_optimized \
    --num_frames 4 \
    --prompt_variation validation_optimized
```

**Expected time**: ~5-10 minutes  
**Expected cost**: ~$1.40

### Step 4: Compare Results

Compare validation_optimized to intensity_aware (current best: 74.07%):

```bash
python3.13 -c "
import json
from pathlib import Path

intensity_file = Path('results/llm_optimization/variation_intensity_aware/results.json')
optimized_file = Path('results/llm_optimization/variation_validation_optimized/results.json')

with open(intensity_file) as f:
    intensity = json.load(f)
with open(optimized_file) as f:
    optimized = json.load(f)

intensity_acc = intensity['metrics']['overall_accuracy'] * 100
optimized_acc = optimized['metrics']['overall_accuracy'] * 100

print('Validation Set Comparison:')
print(f'Intensity-Aware:     {intensity_acc:.2f}%')
print(f'Validation-Optimized: {optimized_acc:.2f}%')
print(f'Improvement:         {optimized_acc - intensity_acc:+.2f}%')
"
```

### Step 5: Final Test Evaluation (If Improved)

**Only if validation_optimized performs better than intensity_aware:**

```bash
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_test_final.json \
    --output_dir results/llm_final_evaluation_optimized \
    --num_frames 4 \
    --prompt_variation validation_optimized
```

## Success Criteria

- **Validation accuracy**: Should be > 74.07% (current best)
- **Test accuracy**: Expected 72-74% (slightly lower than validation)
- **Improvement**: Should address validation set error patterns

## Scientific Reporting

**In paper, report:**

```
We optimized prompts based on validation set error patterns. The validation 
set (54 trials) was used to identify common confusions (e.g., afraid vs. 
surprised, interested vs. neutral). We created an enhanced prompt addressing 
these patterns and tested it on the validation set. The best-performing 
prompt was then evaluated on a held-out test set (118 trials), which was 
never used for optimization.
```

This is **scientifically sound** and **publishable**.
