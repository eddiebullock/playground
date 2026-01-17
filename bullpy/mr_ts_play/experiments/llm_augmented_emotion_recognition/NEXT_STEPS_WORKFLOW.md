# Next Steps: Optimization Workflow

## ✅ What We've Done

1. **Reorganized data splits** ✅
   - Validation set: 54 trials (for optimization)
   - Test set: 118 trials (for final evaluation)

2. **Established baseline** ✅
   - Baseline prompt: 68.52% on validation set
   - This is our starting point

3. **Created prompt variations** ✅
   - Enhanced prompts ready to test

## 🎯 What To Do Next

### Step 1: Optimize on VALIDATION Set (Do This First!)

**Goal**: Try different prompts on validation set, find the best one.

**Process**:
1. Test each prompt variation on **validation set** (54 trials)
2. Compare results
3. Select best performing prompt

**Commands**:

```bash
# 1. Baseline (already done - 68.52%)
# Results: results/llm_optimization/baseline_validation/

# 2. Explicit Distinctions (try this first - highest expected gain)
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_explicit_distinctions \
    --num_frames 4 \
    --prompt_variation explicit_distinctions

# 3. Intensity-Aware
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_intensity_aware \
    --num_frames 4 \
    --prompt_variation intensity_aware

# 4. Temporal Analysis
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_temporal_analysis \
    --num_frames 4 \
    --prompt_variation temporal_analysis

# 5. Combined (all enhancements)
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_combined \
    --num_frames 4 \
    --prompt_variation combined
```

**Expected Time**: ~2-3 hours (5 variations × 30-40 min each)

**Expected Results**:
- Baseline: 68.52%
- Explicit Distinctions: 73-75% (expected)
- Intensity-Aware: 71-73% (expected)
- Temporal Analysis: 70-72% (expected)
- Combined: 75-77% (expected)

### Step 2: Compare Results on Validation Set

**Goal**: Select best performing prompt.

**Process**:
1. Load results from each variation
2. Compare validation accuracy
3. Check per-emotion performance (especially "worried", "afraid")
4. Select best prompt

**Quick comparison script**:
```python
import json
from pathlib import Path

variations = [
    'baseline_validation',
    'variation_explicit_distinctions',
    'variation_intensity_aware',
    'variation_temporal_analysis',
    'variation_combined'
]

print("Validation Set Results:")
print("="*60)
for var in variations:
    results_file = Path(f"results/llm_optimization/{var}/results.json")
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            acc = data.get('metrics', {}).get('overall_accuracy', 0) * 100
            print(f"{var:40} {acc:6.2f}%")
```

### Step 3: Final Evaluation on TEST Set (Do This Last!)

**Goal**: Get final, publishable result.

**Process**:
1. Run **best prompt** on test set (118 trials)
2. This is your **final, publishable result**
3. Do NOT modify based on test results

**Command** (after selecting best prompt):
```bash
# Replace 'combined' with your best performing variation
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_test_final.json \
    --output_dir results/llm_final_evaluation \
    --num_frames 4 \
    --prompt_variation combined  # Use best variation here
```

**Expected Result**: 70-73% (slightly lower than validation, but honest)

## 📊 Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: OPTIMIZE ON VALIDATION SET                      │
│ (Can try multiple prompts, iterate)                      │
│                                                          │
│  Baseline: 68.52%                                        │
│  → Try explicit_distinctions                            │
│  → Try intensity_aware                                  │
│  → Try temporal_analysis                                │
│  → Try combined                                         │
│                                                          │
│  Compare all results                                    │
│  Select best: e.g., combined = 75%                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: FINAL EVALUATION ON TEST SET                    │
│ (Run ONCE, do not modify)                               │
│                                                          │
│  Run best prompt on test set (118 trials)              │
│  Result: e.g., 73% (honest, publishable)                │
└─────────────────────────────────────────────────────────┘
```

## ⚠️ Important Rules

1. **Validation Set**: Use for optimization (try different prompts)
2. **Test Set**: Use ONCE for final evaluation only
3. **Do NOT**: Optimize on test set
4. **Do NOT**: Modify prompts based on test results

## 🚀 Quick Start

**Start with explicit_distinctions** (highest expected gain):

```bash
python3.13 experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_explicit_distinctions \
    --num_frames 4 \
    --prompt_variation explicit_distinctions
```

This will:
- Use enhanced prompt with explicit distinctions
- Run on validation set (54 trials)
- Save results for comparison
- Take ~30-40 minutes

## 📝 Expected Timeline

- **Step 1 (Optimization)**: 2-3 hours
  - Baseline: ✅ Done (68.52%)
  - 4 more variations: ~2 hours
  
- **Step 2 (Comparison)**: 30 minutes
  - Compare results
  - Select best prompt
  
- **Step 3 (Final Test)**: 30 minutes
  - Run best on test set
  - Get final result

**Total**: ~3-4 hours to complete optimization

## ✅ Success Criteria

- Validation accuracy improves from 68.52%
- Test accuracy is within 2-3% of validation (honest estimate)
- Per-emotion performance improves for weak emotions
- Methodology is scientifically sound
