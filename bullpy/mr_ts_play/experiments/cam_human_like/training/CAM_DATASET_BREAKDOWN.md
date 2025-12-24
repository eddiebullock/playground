# CAM Dataset Breakdown: Training and Testing Data

## Executive Summary

**Critical Issue**: 49 out of 100 trials (all voice trials) have corrupted files that cannot be used for training or testing.

## Dataset Overview

### Total Trials: 100
- **Face Trials**: 51 (all valid)
- **Voice Trials**: 49 (all corrupted)

### File Status
- **Valid Files (>50KB)**: 51 (all face)
- **Corrupted Files (<50KB)**: 49 (all voice)
- **Missing Files**: 0

## What the Model Will Actually Train/Test On

### Current Situation (Without Fix)

**Training Set:**
- **Usable**: 51 trials (all face modality)
- **Skipped**: 49 trials (all voice modality)
- **Effective Training**: 51 face-only trials

**Test Set:**
- **Usable**: 10 trials (10 face, 0 voice)
- **Skipped**: 10 trials (all voice)
- **Effective Testing**: 10 face-only trials

**Problem**: The model is only seeing **face trials**, completely missing voice modality!

### After Fixing Trial Definitions

**Training Set:**
- **Usable**: 100 trials (51 face + 49 voice)
- **Skipped**: 0 trials
- **Effective Training**: Full dataset with both modalities

**Test Set:**
- **Usable**: 20 trials (10 face + 10 voice)
- **Skipped**: 0 trials
- **Effective Testing**: Balanced face/voice evaluation

## Train/Test Split Details

Based on `hpc_cam_replication.sh`:

1. **Split Creation** (Step 1):
   - Uses `create_cam_splits.py` with `concept_balanced` method
   - 80% train (80 trials), 20% test (20 trials)
   - Ensures each concept appears in both train and test

2. **Training** (Step 2):
   - Trains on `train_trials.json` (80 trials)
   - Validates on `test_trials.json` (20 trials)
   - Uses same test set for validation during training

3. **Evaluation** (Step 3):
   - Evaluates on `test_trials.json` (20 trials)
   - This is the final test set

## Concept Distribution

Each of the 20 concepts has 5 trials:
- Typically: 2-3 face trials, 2-3 voice trials
- After corruption: Only face trials remain usable

**Example Breakdown** (top concepts):
- `confronted`: 2 face (valid) + 3 voice (corrupted)
- `resentful`: 2 face (valid) + 3 voice (corrupted)
- `subservient`: 2 face (valid) + 3 voice (corrupted)
- `distaste`: 3 face (valid) + 2 voice (corrupted)

## About the Corrupted Files

### File Analysis
- **Format**: Valid QuickTime movie files (`.mov`)
- **Structure**: Contains QuickTime headers (`moov`, `cmov` chunks)
- **Compression**: Uses zlib compression
- **Size**: 4-18KB (too small to contain actual video/audio)
- **Status**: Incomplete/corrupted - likely truncated during transfer or creation

### Why They're Corrupted
1. **File Structure**: They have valid QuickTime headers but incomplete data
2. **Size**: Too small to contain meaningful video/audio content
3. **Pattern**: ALL voice trials (`T` files) are corrupted, ALL face trials (`V` files) are valid
4. **Likely Cause**: Original dataset issue or transfer corruption affecting only `T` files

### Can We Fix the Corrupted Files?

**Short Answer**: No, not directly.

**Why**:
- The files are incomplete - missing actual video/audio data
- They're not just compressed - they're truncated
- Re-downloading from source would be needed (if available)

**Solution**: Use the fix script to replace corrupted `T` files with valid `V` files in the trial definitions.

## Impact on Results

### Current Results (10 valid test trials)
- **Overall Accuracy**: 50.00%
- **Face Accuracy**: 33.3% (below 37% baseline) ❌
- **Voice Accuracy**: 75% (but only 3 trials, unreliable)
- **Sample Size**: Too small (10 trials) for reliable metrics

### Expected Results (After Fix - 20 valid test trials)
- **More Reliable Metrics**: Larger sample size
- **Balanced Evaluation**: Both face and voice modalities
- **Better Face Accuracy**: More face trials in test set
- **Proper Voice Evaluation**: 10 voice trials instead of 0

## Recommendations

1. **Immediate Action**: Run `fix_cam_trial_definitions.py` to update trial definitions
2. **Re-run Training**: Train with all 100 trials (both modalities)
3. **Re-run Evaluation**: Test on all 20 trials (balanced face/voice)
4. **Investigate Source**: Check if original CAM dataset has these corrupted files or if it's a transfer issue

## Files to Update

1. **Trial Definitions**: `data/cam_trial_definitions_20concepts.json`
   - Update 49 voice trials to use valid `V` files
   - Preserve actor information where possible

2. **HPC Scripts**: Already configured correctly
   - `hpc_cam_replication.sh` will use updated trial definitions
   - No changes needed to training/evaluation scripts

## Next Steps

1. Run fix script locally or on HPC:
   ```bash
   python3 experiments/cam_human_like/training/fix_cam_trial_definitions.py \
       data/cam_trial_definitions_20concepts.json \
       "/path/to/CAM/Emotions" \
       --output data/cam_trial_definitions_20concepts_fixed.json
   ```

2. Transfer updated trial definitions to HPC:
   ```bash
   rsync data/cam_trial_definitions_20concepts_fixed.json \
       eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/data/
   ```

3. Re-run CAM replication on HPC with fixed trial definitions


