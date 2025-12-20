# Why 37% Accuracy is Low and What's Wrong

## The Results Breakdown

**Overall**: 37.49% accuracy (vs random: 14.29%)

**But look at per-class performance:**

| Emotion | Recall | Support | What's Happening |
|---------|--------|---------|------------------|
| **Neutral** | 64% | 370 | Model predicts this a lot (most common) |
| **Happy** | 59% | 237 | Model predicts this often (2nd most common) |
| **Angry** | 12% | 145 | Model barely learns this |
| **Sad** | **2%** | 136 | Model almost never predicts this |
| **Fear** | **0%** | 95 | Model completely fails |
| **Surprise** | **3%** | 59 | Model almost never predicts this |
| **Disgust** | **0%** | 17 | Model completely fails |

## The Real Problem

**The model is essentially only learning to predict "Neutral" and "Happy"** - the two most common classes.

Together, Neutral + Happy = 607 out of 1059 test samples (57%)
- If model predicts Neutral/Happy for everything, it gets ~57% accuracy
- But it's getting 37%, meaning it's sometimes wrong even on these!

**The model is NOT learning the other 5 emotions at all.**

## Why This Is Happening

### 1. **Severe Class Imbalance** ⚠️

**Train set distribution:**
- Neutral: 206 samples (35%)
- Happy: 123 samples (21%)
- Angry: 87 samples (15%)
- Sad: 82 samples (14%)
- Fear: 50 samples (9%)
- Surprise: 23 samples (4%)
- **Disgust: 11 samples (2%)** ← TOO FEW!

**Problem**: Model learns to predict common classes and ignores rare ones.

### 2. **Overfitting** ⚠️

- **Training accuracy**: 71.3% (epoch 11)
- **Validation accuracy**: 36.9% (best)
- **Gap**: 34.4% = **SEVERE OVERFITTING**

**Problem**: Model memorizes training data instead of learning generalizable features.

### 3. **Actor-Independent Splits** ⚠️

- Different actors in train vs test
- Model can't rely on actor-specific cues
- **This is GOOD for generalization, but HARD for learning**

### 4. **Limited Training Data** ⚠️

- Only 582 training samples for 7 classes
- Some classes have <25 samples
- Not enough to learn robust features

## What Needs to Be Fixed

### Critical Issues:

1. **Class weights NOT being used** - The experiment has `--use_class_weights` flag but it wasn't used!
2. **No dropout** - Currently dropout=0.0, need regularization
3. **Overfitting** - 34% gap between train and val
4. **Rare classes failing** - Fear and Disgust have 0% recall

## Expected Improvements

### With Proper Fixes:

**If we add**:
- Class weights (handle imbalance)
- Dropout (reduce overfitting)
- Better augmentation
- More frames per video

**Expected**: 45-55% accuracy (still below 70%, but better)

**To reach 70%**, we likely need:
- Remove actor stratification (but then not generalizable)
- OR get more training data
- OR use much better architecture (video transformers, etc.)

## The Hard Truth

**With actor-independent splits and 582 training samples, 70% accuracy may not be achievable.**

**Your options**:
1. **Accept 50-60%** and frame as "challenging benchmark with actor generalization"
2. **Remove actor stratification** to reach 70% (but acknowledge limitation)
3. **Get more data** (collect more samples per emotion)
4. **Reframe question** (few-shot learning contribution, not "as good as humans")

Let me implement the fixes (class weights, dropout, better regularization) and see if we can get to 50-60%.



