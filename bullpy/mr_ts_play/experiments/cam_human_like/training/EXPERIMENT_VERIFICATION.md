# CAM Replication Experiment Verification

## ✅ Forced-Choice Evaluation Format

### Original Golan/CAM Study Procedure
1. **Stimulus presentation**: Participant views/hears a video (face or voice)
2. **Four options shown**: Four numbered adjectives (1-4) are presented
3. **Forced choice**: Participant **must** select one of the four options (no "other" option)
4. **Response**: Press 1, 2, 3, or 4 to select answer
5. **No feedback**: No indication of correctness

### Our Computational Replication

**✅ CORRECTLY IMPLEMENTED**

The evaluation in `evaluate_on_cam.py` matches this procedure:

```python
# Step 1: Model processes stimulus and scores all 4 candidate labels
output = model.score_labels(
    stimulus_path=trial.stimulus_path,
    candidate_labels=trial.candidate_labels,  # Exactly 4 labels
    modality=trial.modality,
)

# Step 2: Decision restricted to the 4 trial options (forced-choice)
predicted_label = max(output.label_scores.items(), key=lambda x: x[1])[0]

# Step 3: Find index of predicted label in candidate_labels (0-3)
predicted_idx = trial.candidate_labels.index(predicted_label)
```

**Key Points:**
- ✅ Model only scores the 4 candidate labels (not all possible emotions)
- ✅ Selection is restricted to these 4 options (forced-choice)
- ✅ Highest-scoring option is selected (mimics human choice)
- ✅ Each trial has exactly 4 candidate labels (1 target + 3 foils)

### Trial Structure Verification

Example trial from `cam_trial_definitions_20concepts.json`:
```json
{
  "stimulus_path": "22/2200301/2200301S3Vdistaste.mov",
  "modality": "face",
  "correct_label": "distaste",
  "candidate_labels": [
    "distaste",      // Target (correct answer)
    "appalled",      // Foil 1
    "intimate",      // Foil 2
    "insincere"      // Foil 3
  ],
  "correct_idx": 0,
  "concept": "distaste",
  "trial_id": "trial_001"
}
```

**✅ Structure is correct:**
- 4 candidate labels (1 target + 3 foils)
- `correct_idx` indicates which option is correct (0-3)
- Model must choose from these 4 options only

## ⚠️ Label Mismatch Issue

### Problem Identified

**EU-Emotion emotions (27):**
- Basic emotions: afraid, angry, happy, sad, surprised, disgusted, etc.
- No overlap with CAM concepts

**CAM concepts (20):**
- Complex mental states: appalled, appealing, confronted, distaste, empathic, etc.
- Completely different from EU-Emotion

**Impact:**
- Model learns to recognize EU-Emotion emotions (afraid, angry, happy, etc.)
- But CAM tests completely different concepts (appalled, distaste, etc.)
- This explains poor performance: 21.43% vs 37% baseline

### Solution: Two-Stage Fine-Tuning

**Stage 1: EU-Emotion Fine-Tuning** (Current)
- Purpose: Learn general emotion recognition features
- Labels: 27 EU-Emotion emotions (afraid, angry, happy, etc.)
- Result: Model learns visual/audio features for emotions

**Stage 2: CAM Fine-Tuning** (Needed)
- Purpose: Adapt to CAM-specific concepts
- Labels: 20 CAM concepts (appalled, distaste, etc.)
- Result: Model learns the specific concepts needed for evaluation

**Why This Works:**
- Stage 1 provides general emotion recognition capabilities
- Stage 2 adapts these capabilities to CAM's specific concepts
- Similar to transfer learning: general → specific

## Experiment Organization Checklist

### ✅ Correctly Implemented

1. **Forced-choice format**: Model restricted to 4 options ✓
2. **Trial structure**: 100 trials, 5 per concept, 4 options each ✓
3. **Modality handling**: Face and voice trials properly handled ✓
4. **Evaluation procedure**: Matches Golan study format ✓
5. **Multi-frame processing**: Extracts multiple frames from videos ✓

### ⚠️ Needs Attention

1. **Label mismatch**: EU-Emotion → CAM transfer needs Stage 2 fine-tuning
2. **Training data**: Need CAM training split for Stage 2
3. **Evaluation split**: Currently using test split (should use separate eval split)

### 📋 Next Steps

1. **Create CAM train/test split** (if not already done)
2. **Implement Stage 2 fine-tuning** on CAM training data
3. **Evaluate on CAM test split** with Stage 2 model
4. **Compare to baseline**: Should improve from 21.43% → target >37%

## Summary

**Evaluation Format**: ✅ **CORRECT** - Properly implements 4-option forced-choice matching Golan study

**Label Mapping**: ⚠️ **ISSUE IDENTIFIED** - Zero overlap between EU-Emotion and CAM labels explains poor performance

**Solution**: Two-stage fine-tuning is essential:
1. EU-Emotion (general emotion features) ← **Current step**
2. CAM (specific concepts) ← **Next step**

The experiment is well-organized and correctly implements the forced-choice format. The poor performance is expected given the label mismatch, and will be addressed with Stage 2 fine-tuning.


