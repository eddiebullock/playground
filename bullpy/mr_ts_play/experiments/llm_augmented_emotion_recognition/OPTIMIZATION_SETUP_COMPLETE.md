# Optimization Setup Complete ✅

## What Was Done

### 1. Data Split Reorganization ✅

**Created proper train/val/test splits:**
- **Validation set** (for optimization): `data/trial_definitions/eu_emotion_val_for_optimization.json` (54 trials)
- **Test set** (for final evaluation): `data/trial_definitions/eu_emotion_test_final.json` (118 trials)
- **Backups created**: Original files backed up in `data/trial_definitions/backup_before_reorganization/`

**Documentation**: See `data/trial_definitions/SPLIT_REORGANIZATION.md`

### 2. LLM Wrapper Enhanced ✅

**Added custom prompt support:**
- Modified `llm_wrapper.py` to accept `custom_prompt` parameter
- All three providers (Google, OpenAI, Anthropic) now support custom prompts
- Backward compatible (default prompts still work)

### 3. Optimization Framework Created ✅

**Files created:**
- `scripts/optimize_prompts.py`: Systematic prompt optimization script
- `PROMPT_OPTIMIZATION_PROTOCOL.md`: Detailed protocol for optimization
- `OPTIMIZATION_SETUP_COMPLETE.md`: This file

## Next Steps

### Step 1: Implement Prompt Variations

You need to create prompt generation functions. See `PROMPT_OPTIMIZATION_PROTOCOL.md` for details.

**Quick start:**
```python
from experiments.llm_augmented_emotion_recognition.models.llm_wrapper import LLMWrapper

# Create wrapper
llm = LLMWrapper.from_config("configs/llm_config.yaml")

# Create custom prompt
custom_prompt = create_enhanced_prompt(candidate_labels, variation="explicit_distinctions")

# Use custom prompt
result = llm.classify_emotion_directly(
    frames=frames,
    candidate_labels=candidate_labels,
    custom_prompt=custom_prompt
)
```

### Step 2: Run Optimization on Validation Set

For each prompt variation:

```bash
# Example: Evaluate explicit distinctions variation
python experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_val_for_optimization.json \
    --output_dir results/llm_optimization/variation_explicit_distinctions \
    --num_frames 4
```

**Note**: You'll need to modify `test_llm_only.py` to accept and use custom prompts, or create a wrapper script.

### Step 3: Compare Results

After evaluating all variations:
1. Compare validation accuracy
2. Compare per-emotion performance
3. Select best performing prompt

### Step 4: Final Evaluation

Run best prompt on test set (ONCE):

```bash
python experiments/llm_augmented_emotion_recognition/scripts/test_llm_only.py \
    --config experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml \
    --test_trials data/trial_definitions/eu_emotion_test_final.json \
    --output_dir results/llm_final_evaluation \
    --num_frames 4
```

## Scientific Rigor ✅

**This setup ensures:**
- ✅ No test set optimization
- ✅ Proper validation-based optimization
- ✅ Honest performance estimates
- ✅ Publishable methodology

## Files Reference

- **Data splits**: `data/trial_definitions/`
- **Optimization protocol**: `PROMPT_OPTIMIZATION_PROTOCOL.md`
- **Split reorganization**: `data/trial_definitions/SPLIT_REORGANIZATION.md`
- **LLM wrapper**: `models/llm_wrapper.py` (now supports custom prompts)
- **Test script**: `scripts/test_llm_only.py`

## Expected Results

**Validation Set (Optimization):**
- Baseline: 68.52% (from previous test set)
- Expected improvements: 73-76% with enhanced prompts

**Test Set (Final Evaluation):**
- Expected: 70-73% (slightly lower than validation, but honest)
- This is your publishable result

## Timeline

- **Setup**: ✅ Complete (2 hours)
- **Implementation**: 2-3 hours (create prompt variations)
- **Validation Evaluation**: 2-3 hours (5 variations)
- **Analysis**: 1 hour
- **Final Test**: 30 minutes

**Total**: ~7-9 hours to complete optimization

## Questions?

See `PROMPT_OPTIMIZATION_PROTOCOL.md` for detailed instructions.
