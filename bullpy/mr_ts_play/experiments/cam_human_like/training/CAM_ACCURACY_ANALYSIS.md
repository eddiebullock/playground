# CAM Accuracy Analysis: Why Performance is Poor

## Current Results

- **Overall Accuracy**: 50.00% (5/10 valid trials)
- **Face Accuracy**: 33.3% (below 37% baseline) ❌
- **Voice Accuracy**: 75% (but unreliable due to bug - see below)
- **Test Set**: Only 10 valid trials (should be 20)

## Root Causes of Poor Performance

### 1. **Severely Reduced Training Data** ⚠️ CRITICAL

**Expected**: 80 training trials (80% of 100)
**Actual**: ~40 training trials (only face modality)

**Why**:
- All 49 voice trials have corrupted files (<50KB)
- Training dataset skips corrupted files
- Model only trained on ~40 face trials instead of 80

**Impact**:
- **50% reduction in training data**
- Model underfits due to insufficient examples
- Cannot learn robust emotion representations

### 2. **Black Frames Fallback During Training** ⚠️

**Problem**: When corrupted files are encountered, `TaskSpecificTrialDataset` returns black frames:

```python
except Exception as e:
    print(f"Warning: Error loading video {video_path}: {e}")
    # Return black frames as fallback
    return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.num_frames)]
```

**Impact**:
- Model learns from **black frames** (no signal)
- These trials provide **no useful information**
- Effectively reduces training data further
- Model may learn to ignore certain patterns

**Better Solution**: Skip corrupted trials entirely or raise error to prevent training on invalid data.

### 3. **No Voice Modality Exposure** ⚠️

**Problem**: Model never saw voice trials during training
- All 49 voice trials were corrupted
- Model only trained on face modality
- Cannot generalize to voice trials

**Impact**:
- Model lacks multimodal understanding
- Cannot handle voice trials at test time
- Incomplete learning of emotion recognition

### 4. **Small Test Set** ⚠️

**Expected**: 20 test trials (balanced face/voice)
**Actual**: 10 valid test trials (all face)

**Impact**:
- **Unreliable metrics** (small sample size)
- Face accuracy of 33.3% based on only ~3-4 trials
- High variance in results
- Cannot draw reliable conclusions

### 5. **Metrics Calculation Bug** (Fixed) ✅

**Problem**: `compute_metrics` incorrectly zipped predictions with all trials
- Assumed `predictions[i]` corresponds to `trials[i]`
- When trials are skipped, indices don't align
- Face predictions matched to voice trials (incorrect)

**Impact**:
- Reported face/voice accuracies were **incorrect**
- Voice accuracy of 75% was artifact of bug
- After fix, voice accuracy should be 0% (no valid voice trials)

**Status**: ✅ Fixed in `evaluate_on_cam.py`

## Why Face Accuracy is Below Baseline (33% vs 37%)

### Possible Explanations:

1. **Insufficient Training Data**
   - Only ~40 trials vs expected 80
   - Model underfits
   - Cannot learn robust features

2. **Training on Black Frames**
   - Corrupted voice trials returned black frames
   - Model learned noise instead of signal
   - Degraded performance

3. **Small Test Set**
   - Only ~10-11 face trials in test
   - High variance
   - 33.3% could be statistical fluctuation

4. **Hyperparameters Not Optimized**
   - Learning rate: 1e-5 (may be too low)
   - Batch size: 4 (small, may cause instability)
   - Epochs: 10 (may need more with less data)

5. **Model Architecture Limitations**
   - CLIP may not be optimal for emotion recognition
   - Multi-frame aggregation (mean pooling) may lose information
   - No task-specific architecture modifications

## Expected Performance After Fixes

### After Fixing Trial Definitions:

**Training**:
- **100 trials** (51 face + 49 voice using V files)
- **80 training trials** (full dataset)
- **No black frames** (all files valid)

**Testing**:
- **20 test trials** (10 face + 10 voice)
- **Reliable metrics** (larger sample)

**Expected Accuracy**:
- **Face Accuracy**: 50-60% (above 37% baseline)
- **Voice Accuracy**: 40-50% (new metric)
- **Overall Accuracy**: 45-55% (balanced)

### After Additional Improvements:

1. **Better Frame Aggregation**
   - Attention mechanism instead of mean pooling
   - Temporal modeling

2. **Hyperparameter Tuning**
   - Learning rate: 5e-5 or 1e-4
   - Batch size: 8-16
   - More epochs: 15-20

3. **Data Augmentation**
   - Random crops, flips
   - Color jitter
   - Temporal augmentation

4. **Transfer Learning**
   - Pre-train on EU-Emotion dataset
   - Fine-tune on CAM

## Recommendations

### Immediate Actions:

1. ✅ **Fix Trial Definitions** (use `fix_cam_trial_definitions.py`)
   - Replace corrupted T files with valid V files
   - Ensures all 100 trials are usable

2. ✅ **Fix Metrics Bug** (already done)
   - Match predictions to trials by trial_id
   - Accurate face/voice accuracy reporting

3. ⚠️ **Fix Training Dataset** (skip corrupted files properly)
   - Don't return black frames
   - Skip trials that can't be loaded
   - Or raise error to prevent training on invalid data

4. **Re-run Training**
   - Train on full 100 trials
   - Validate on 20 trials
   - Should see improved accuracy

### Long-term Improvements:

1. **Hyperparameter Optimization**
   - Grid search for learning rate, batch size
   - Early stopping based on validation

2. **Architecture Improvements**
   - Attention-based frame aggregation
   - Temporal convolutions
   - Multi-scale features

3. **Data Quality**
   - Verify all files are valid before training
   - Pre-process and validate dataset
   - Check for corrupted files upfront

## Conclusion

The poor accuracy (33% face, below 37% baseline) is primarily due to:

1. **50% reduction in training data** (40 vs 80 trials)
2. **Training on black frames** (no signal from corrupted files)
3. **Small test set** (unreliable metrics)
4. **No voice modality** (incomplete learning)

After fixing trial definitions and re-running training with the full dataset, we should see:
- **Face accuracy**: 50-60% (above baseline)
- **Overall accuracy**: 45-55%
- **Reliable metrics** (larger test set)

The model is likely working correctly, but was severely limited by the corrupted data issue.




