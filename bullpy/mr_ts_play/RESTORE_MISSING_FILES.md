# Restore Missing Files

The following files were deleted during "undo all" and need to be restored:

1. `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py`
2. `experiments/llm_augmented_emotion_recognition/evaluation/three_way_comparison.py`

## Option 1: Restore from Git (If Files Were Committed)

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

# Try to restore from HEAD
git checkout HEAD -- experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py
git checkout HEAD -- experiments/llm_augmented_emotion_recognition/evaluation/three_way_comparison.py

# If that doesn't work, check git history
git log --all --full-history --oneline -- "**/llm_wrapper.py"
git log --all --full-history --oneline -- "**/three_way_comparison.py"
```

## Option 2: Use Working Script Instead

Since `run_llm_augmented_experiment.py` exists and worked before, you can:

1. Temporarily update config to point to val+test file
2. Run the three-way comparison (it will generate LLM-only results)
3. Extract LLM-only results

The script has been updated to do this automatically.

## Quick Fix

The `test_llm_on_val_test.sh` script has been updated to:
- Use the working `run_llm_augmented_experiment.py` script
- Extract LLM-only results from the output
- Save to the desired location

Try running it again - it should work now!
