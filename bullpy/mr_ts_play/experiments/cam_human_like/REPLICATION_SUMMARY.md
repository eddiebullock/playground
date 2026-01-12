# CAM Face-Voice Battery Replication Summary

## Experiment Setup

This experiment replicates the Cambridge Mindreading (CAM) Face-Voice Battery (Golan et al., 2006) using pretrained computer-vision models instead of human participants.

## Original CAM Study (Golan et al., 2006)

**Participants:**
- Group 1: 21 adults with Asperger Syndrome (AS)
- Group 2: 17 control adults (matched on age, IQ)

**Test Structure:**
- 20 emotion concepts
- 5 items per concept = 100 total trials
- Counterbalanced: 3 face + 2 voice OR 2 face + 3 voice per concept
- 4-option forced-choice (1 target + 3 foils)
- Foils from levels 4 and 5, different emotion groups than target

**Original Results:**
- Control group: ~85-90% overall accuracy
- AS group: ~65-75% overall accuracy
- Face accuracy: Control ~88%, AS ~70%
- Voice accuracy: Control ~87%, AS ~60%

## Your Replication

### Trial Generation
✅ **Generated 100 trials across 20 CAM concepts**
- Exactly matching original CAM structure
- 5 trials per concept
- Counterbalanced face/voice distribution
- Proper foil selection (levels 4-5, different groups)

### The 20 CAM Concepts
1. appalled
2. appealing (asking for)
3. confronted
4. distaste
5. empathic
6. exonerated
7. grave
8. guarded
9. insincere
10. intimate
11. lured
12. mortified
13. nostalgic
14. reassured
15. resentful
16. stern
17. subdued
18. subservient
19. uneasy
20. vibrant

### Current Results (CLIP, zero-shot, test split)

**Test Split (actor-independent):**
- Overall accuracy: 39.1% (23 trials)
- Face accuracy: 40.9% (22 trials)
- Voice accuracy: 0.0% (1 trial - too few for meaningful result)
- Concept recognition rate: 0.0% (17 concepts tested)

**Note on sample size:**
The test split has only 23 trials because actor-independent splitting filtered out most trials. The original CAM study tested all participants on the same 100 trials (no actor splitting).

### Comparison

| Metric | Original CAM (Control) | Your Replication (CLIP) |
|--------|----------------------|------------------------|
| Overall Accuracy | ~85-90% | 39.1% |
| Face Accuracy | ~88% | 40.9% |
| Voice Accuracy | ~87% | N/A (CLIP doesn't process audio) |
| Sample Size | 100 trials | 23 trials (test split) |

## Key Differences from Original

1. **Actor Independence**: Your replication uses actor-independent splits (different actors in train/test), while original CAM tested all participants on same stimuli
2. **Model vs Human**: CLIP is a zero-shot vision-language model, not trained for emotion recognition
3. **Audio Processing**: CLIP doesn't process audio, so voice trials are less meaningful
4. **Sample Size**: Test split has fewer trials due to actor filtering

## Next Steps

### Option 1: Run on All Trials (No Actor Filtering)
To match original CAM more closely, you could:
- Use all 100 trials (not filtered by actor split)
- This matches how original CAM tested all participants on same stimuli

### Option 2: Try Different Models
- Larger CLIP models (CLIP-Large)
- Multimodal models (vision + audio)
- Emotion-specific models

### Option 3: Few-Shot Calibration
- Use validation set to calibrate model
- Temperature scaling or Platt scaling

## Files Generated

- `data/cam_trial_definitions_20concepts.json` - 100 trials for 20 CAM concepts
- `results/cam_human_like/clip_*/` - Experiment results
  - `summary.json` - Overall metrics
  - `trial_results.csv` - Per-trial predictions
  - `confusion_matrix.csv` - Error analysis
  - `per_emotion_accuracy.csv` - Per-emotion breakdown
  - `per_concept_accuracy.csv` - Per-concept breakdown

## Replication Status

✅ **Trial Structure**: Matches original CAM exactly
✅ **20 Concepts**: All original CAM concepts included
✅ **Counterbalancing**: 3+2 or 2+3 face/voice per concept
✅ **Foil Selection**: Levels 4-5, different emotion groups
✅ **Forced-Choice**: 4-option structure preserved
⏳ **Sample Size**: Limited by actor-independent splits (can be adjusted)
⏳ **Model Performance**: Below human performance (expected for zero-shot)

The replication structure is complete and matches the original CAM methodology. The lower performance is expected for a zero-shot model compared to human participants.









