# Final Methodology: Separate Experiments for High-Impact Mental State Recognition Study

## Research Question

**Primary**: Can computer vision identify mental states accurately?  
**Secondary**: Can it approach/match human performance levels?

## Study Design: Separate Experiments

### Core Principle: **Unified Methodology, Separate Experiments**

Both datasets follow the **same procedure** but are evaluated **separately**:

1. ✅ **Discover ALL available valid files** (maximize data usage)
2. ✅ **Generate trials from discovered files** (consistent approach)
3. ✅ **Train/test split** (80/20, concept-balanced)
4. ✅ **Train and evaluate separately** (no transfer learning)
5. ✅ **Compare to human baselines** (comprehensive evaluation)

## Experiment 1: EU-Emotion Mental State Recognition

### Procedure

1. **Discover All Files**
   - Scan EU-Emotion dataset directory
   - Find ALL valid face/voice files (>50KB)
   - Group by emotion (27 emotions)

2. **Generate Trials**
   - Create forced-choice trials (4 options each)
   - 10 trials per emotion (if enough files available)
   - Proper foil selection (semantically different emotions)
   - Result: ~270 trials total

3. **Train/Test Split** (80/20)
   - Train: ~216 trials
   - Test: ~54 trials
   - Concept-balanced (each emotion in both splits)

4. **Training**
   - Fine-tune CLIP on EU-Emotion train set
   - Task-specific training (4-option forced-choice)
   - 10 epochs
   - Expected: 60-75% accuracy

5. **Evaluation**
   - Test on EU-Emotion test set
   - Compare to random baseline (25%)
   - Report comprehensive metrics

### Expected Results
- **Accuracy**: 60-75%
- **Interpretation**: CV can identify EU-Emotion emotions accurately
- **Comparison**: Well above random (25%)

## Experiment 2: CAM Mental State Recognition

### Procedure

1. **Discover All Files**
   - Scan CAM dataset directory
   - Find ALL valid video files (>50KB)
   - Group by concept (20 concepts)
   - **Use ALL ~2,496 valid files** (not just 100)

2. **Generate Trials**
   - Create forced-choice trials (4 options each)
   - 10-15 trials per concept (if enough files available)
   - Proper foil selection (semantically different concepts)
   - Result: ~200-300 trials total

3. **Train/Test Split** (80/20)
   - Train: ~160-240 trials
   - Test: ~40-60 trials
   - Concept-balanced (each concept in both splits)

4. **Training**
   - Fine-tune CLIP on CAM train set
   - Task-specific training (4-option forced-choice)
   - 10 epochs
   - Expected: 70-85% accuracy

5. **Evaluation**
   - Test on CAM test set
   - Compare to:
     - Random baseline (25%)
     - Zero-shot CLIP (37%)
     - Human (AS): 70%
     - Human (Control): 88%
   - Report comprehensive metrics

### Expected Results
- **Accuracy**: 70-85%
- **Interpretation**: CV can identify CAM mental states accurately
- **Comparison**: Approaches human performance (70-88%)

## Key Advantages

### 1. Scientifically Robust
- Clear research questions
- No label mismatch issues
- Standard ML practice
- Reproducible methodology

### 2. High Impact
- Shows CV can identify mental states on both datasets
- Compares basic vs complex emotions
- Compares to human baselines
- Publication-ready results

### 3. Unified Methodology
- Same procedure for both datasets
- Fair comparison
- Easy to interpret
- Consistent evaluation

### 4. Maximum Data Usage
- Uses ALL available valid files
- Better learning, less underfitting
- More robust models

## Implementation

### Scripts Required

1. **EU-Emotion Trial Generator** (`create_eu_emotion_trials.py`) ✅ Already exists
2. **CAM Trial Generator** (`create_cam_trials_from_all_files.py`) ✅ Created
3. **Training Scripts** (updated for separate experiments)
4. **Evaluation Scripts** (comprehensive metrics)
5. **Test Scripts** (quick validation before full runs)

### Workflow

**EU-Emotion Experiment**:
```bash
# 1. Generate trials
python create_eu_emotion_trials.py --eu-emotion-dir ... --trials-per-emotion 10

# 2. Train
python finetune_clip_emotions.py --train_trials ... --val_trials ... --dataset_type eu_emotion

# 3. Evaluate
python evaluate_on_cam.py --model_path ... --trial_definitions ... --dataset_type eu_emotion
```

**CAM Experiment**:
```bash
# 1. Generate trials
python create_cam_trials_from_all_files.py --cam-dir ... --trials-per-concept 10

# 2. Train
python finetune_clip_emotions.py --train_trials ... --val_trials ... --dataset_type cam

# 3. Evaluate
python evaluate_on_cam.py --model_path ... --trial_definitions ... --dataset_type cam
```

## Comparison to Human Baselines

### EU-Emotion
- **Random baseline**: 25% (4-option forced-choice)
- **Target**: 60-75% (well above random)

### CAM
- **Random baseline**: 25%
- **Zero-shot CLIP**: 37%
- **Human (AS)**: 70%
- **Human (Control)**: 88%
- **Target**: 70-85% (approaching human performance)

## Success Criteria

### EU-Emotion
- ✅ Accuracy > 50% (well above random)
- ✅ Shows CV can identify basic emotions
- ✅ Demonstrates methodology works

### CAM
- ✅ Accuracy > 70% (approaching human)
- ✅ Shows CV can identify complex mental states
- ✅ Compares favorably to human baselines
- ✅ **High impact**: Answers research question

## Conclusion

This methodology:
- ✅ **Robust**: Scientifically sound, reproducible
- ✅ **High impact**: Addresses research question directly
- ✅ **Comprehensive**: Uses all data, proper evaluation
- ✅ **Publication-ready**: Clear methodology, proper baselines

Expected outcome: **70-85% accuracy on CAM**, approaching human performance levels.

