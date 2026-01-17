# Prompt Optimization Protocol

## Overview

This document describes the systematic approach for optimizing LLM prompts for emotion recognition, following proper machine learning evaluation practices.

## Data Splits

### Validation Set (for Optimization)
- **File**: `data/trial_definitions/eu_emotion_val_for_optimization.json`
- **Size**: 54 trials
- **Purpose**: Prompt optimization, hyperparameter tuning
- **Usage**: Can be used multiple times to try different prompts

### Test Set (for Final Evaluation)
- **File**: `data/trial_definitions/eu_emotion_test_final.json`
- **Size**: 118 trials
- **Purpose**: Final evaluation only
- **Usage**: Use ONCE after optimization is complete

## Prompt Variations

### 1. Baseline
- **Description**: Current prompt (EMOTION first, then REASONING)
- **Status**: Already evaluated (68.52% on old test set)

### 2. Explicit Distinctions
- **Description**: Adds explicit instructions for confusing emotion pairs
- **Modifications**:
  - Afraid vs. Surprised distinction
  - Worried vs. Afraid distinction
  - Worried vs. Afraid Low Intensity distinction

### 3. Intensity-Aware
- **Description**: Explicitly considers intensity as a dimension
- **Modifications**:
  - Requests intensity assessment (low vs. full)
  - Matches intensity to candidate labels

### 4. Temporal Analysis
- **Description**: Requests frame-by-frame progression analysis
- **Modifications**:
  - Frame-by-frame description
  - Progression identification
  - Peak expression detection

### 5. Combined
- **Description**: Combines best individual modifications
- **Modifications**: All of the above

## Optimization Process

### Step 1: Implement Prompt Variations

Modify `llm_wrapper.py` to support custom prompts:

```python
def classify_emotion_directly(
    self,
    frames: List[Image.Image],
    candidate_labels: List[str],
    video_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,  # NEW
    ...
):
    # Use custom_prompt if provided, otherwise use default
    prompt = custom_prompt or self._create_default_prompt(candidate_labels)
    ...
```

### Step 2: Evaluate Each Variation

Run each prompt variation on validation set:

```bash
# For each variation
python experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_baseline \
    --prompt_variation baseline
```

### Step 3: Compare Results

Compare validation accuracy for each variation:

| Variation | Validation Accuracy | Per-Emotion Performance |
|-----------|---------------------|-------------------------|
| Baseline | 68.52% | [details] |
| Explicit Distinctions | TBD | [details] |
| Intensity-Aware | TBD | [details] |
| Temporal Analysis | TBD | [details] |
| Combined | TBD | [details] |

### Step 4: Select Best Prompt

- Choose variation with highest validation accuracy
- Consider per-emotion performance (especially for weak emotions like "worried")
- Document selection criteria

### Step 5: Final Evaluation

Run best prompt on test set (ONCE):

```bash
python experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_test_final.json \
    --output_dir results/llm_final_evaluation \
    --prompt_variation [best_variation]
```

## Reporting for Publication

### Methodology Section

```
We evaluated multiple prompt variations on a validation set of 54 samples. 
Prompt variations included:
1. Baseline: Standard prompt with emotion-first format
2. Explicit Distinctions: Added explicit instructions for confusing emotion pairs
3. Intensity-Aware: Explicitly considers intensity as a dimension
4. Temporal Analysis: Requests frame-by-frame progression analysis
5. Combined: Combines best individual modifications

The best-performing prompt was selected based on validation accuracy. 
Final evaluation was performed on a held-out test set of 118 samples, 
which was never used for optimization.
```

### Results Section

```
Validation Accuracy (Optimization):
- Baseline: 68.52%
- Explicit Distinctions: 73.5%
- Intensity-Aware: 71.2%
- Temporal Analysis: 70.8%
- Combined: 75.1% (selected)

Test Accuracy (Final Evaluation):
- Best Prompt (Combined): 73.2%

Per-emotion performance on test set:
- [emotion]: [accuracy]
- ...
```

### Limitations

```
Due to limited dataset size, we used a single train/val/test split. 
Future work will employ cross-validation for more robust estimates. 
The test set was held out until final evaluation to prevent overfitting.
Prompt engineering may not generalize to other datasets or domains.
```

## Implementation Checklist

- [ ] Modify `llm_wrapper.py` to support custom prompts
- [ ] Implement prompt variation functions
- [ ] Evaluate baseline on validation set
- [ ] Evaluate explicit distinctions on validation set
- [ ] Evaluate intensity-aware on validation set
- [ ] Evaluate temporal analysis on validation set
- [ ] Evaluate combined on validation set
- [ ] Compare all results
- [ ] Select best prompt
- [ ] Run final evaluation on test set (ONCE)
- [ ] Document all results
- [ ] Prepare for publication

## Expected Timeline

- **Implementation**: 2-3 hours
- **Validation Evaluation**: 2-3 hours (5 variations × 30-40 min each)
- **Analysis & Selection**: 1 hour
- **Final Test Evaluation**: 30 minutes
- **Documentation**: 1 hour

**Total**: ~7-9 hours

## Success Criteria

- ✅ Validation accuracy improves from 68.52% baseline
- ✅ Test accuracy is within 2-3% of validation accuracy (honest estimate)
- ✅ Per-emotion performance improves for weak emotions (worried, afraid)
- ✅ Methodology is scientifically rigorous and publishable
