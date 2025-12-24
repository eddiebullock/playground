# Dataset Size Explanation: Why Only 40 Trials?

## Key Distinction: Trial Definitions vs Actual Dataset

### The Actual Dataset (Your Files)
- **Location**: `/Users/eb2007/.../Emotions/`
- **Total `.mov` files**: **~4,968 files**
- **Valid files (>50KB)**: **~2,496 files**
- **These are ALL available videos** in the CAM dataset

### The Trial Definitions (Experiment Design)
- **File**: `data/cam_trial_definitions_20concepts.json`
- **Total trials defined**: **100 trials**
- **These are SPECIFIC trials** selected for the CAM experiment

## Why Only 100 Trials?

The CAM experiment follows the **original Golan et al. (2006) methodology**:

1. **20 emotion concepts** (e.g., confronted, resentful, subservient)
2. **5 trials per concept** = 100 total trials
3. **Each trial is a forced-choice question**:
   - 1 video stimulus
   - 4 candidate emotion labels (1 correct + 3 foils)
   - Carefully selected foils to match human experimental design

**Why not use all 4,968 files?**
- The trial definitions have **carefully selected foils** (wrong answers)
- This matches the original experimental design
- Ensures fair comparison with published human performance
- Using random files would not be comparable to the original study

## The "40 Trials" Explained

### Current Situation (With Corrupted Files)

**Total defined trials**: 100
- Face trials: 51 (all valid ✅)
- Voice trials: 49 (all corrupted ❌)

**80/20 Train/Test Split**:
- **Train set**: 80 trials expected
  - Face: ~40 trials (usable) ✅
  - Voice: ~40 trials (corrupted, skipped) ❌
  - **Actual usable**: ~40 trials (50% of expected!)

- **Test set**: 20 trials expected
  - Face: ~11 trials (usable) ✅
  - Voice: ~9 trials (corrupted, skipped) ❌
  - **Actual usable**: ~11 trials (55% of expected!)

### Why "40 Trials"?
The **40** refers to **usable training trials**, not total files:
- Expected: 80 training trials
- Actual: ~40 usable trials (only face modality)
- **Missing 40 trials** due to corrupted voice files

## Why Test Set Becomes "Larger" After Fix

### Before Fix:
- **Expected**: 20 test trials
- **Actual usable**: ~11 trials (only face)
- **Missing**: 9 trials (corrupted voice files)

### After Fix (Replace Corrupted T Files with Valid V Files):
- **Expected**: 20 test trials
- **Actual usable**: **20 trials** (all valid!)
- **Gain**: +9 trials (from 11 to 20)

This is why the test set becomes "larger" - we go from **~11 usable trials to 20 usable trials**.

## Summary

| Item | Count | Explanation |
|------|-------|-------------|
| **Total video files in dataset** | ~4,968 | All available videos |
| **Valid video files** | ~2,496 | Files >50KB |
| **Trials in definitions** | 100 | Specific experiment design |
| **Usable training trials (current)** | ~40 | Only face, voice corrupted |
| **Usable test trials (current)** | ~11 | Only face, voice corrupted |
| **Usable training trials (after fix)** | 80 | All trials valid |
| **Usable test trials (after fix)** | 20 | All trials valid |

## The Real Issue

The problem is **NOT** that there aren't enough files in the dataset. The problem is:

1. **Only 100 specific trials** are used (by design, to match original study)
2. **49 of those 100 trials** have corrupted files (voice trials)
3. **Training only sees ~40 usable trials** instead of 80
4. **Test only sees ~11 usable trials** instead of 20

After fixing the trial definitions (replacing corrupted T files with valid V files), we get:
- **80 usable training trials** (full dataset)
- **20 usable test trials** (full dataset)

This is why accuracy is poor - the model is training on **half the expected data**!

