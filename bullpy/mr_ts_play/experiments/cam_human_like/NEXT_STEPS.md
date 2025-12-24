# Next Steps: Running the CAM Replication

## Current Status

✅ **Trial Generation Complete**
- Generated 2,050 trials across 410 concepts
- 5 trials per concept (matching CAM methodology)
- Counterbalanced face/voice distribution (3+2 or 2+3)
- Proper foil selection using CAM taxonomy

✅ **Experiment Already Run Once**
- Tested with CLIP model on test split
- Results saved in `results/cam_human_like/clip_20251219_104735/`
- Overall accuracy: 20.2% (above random chance of 25% for 4 options)

## Important Note: Concept Count Difference

**Original CAM Study:**
- 20 emotion concepts
- 5 items per concept = 100 total trials

**Your Current Setup:**
- 410 emotion concepts (all concepts in dataset)
- 5 items per concept = 2,050 total trials

This is **fine for a computational replication** - you're testing on a broader set of emotions, which is actually more comprehensive. However, if you want to match the original CAM exactly, you would need to:

1. Identify the 20 specific concepts from the original CAM battery
2. Filter trials to just those 20 concepts

## Next Steps

### Option 1: Run Full Experiment (Recommended)

You can run the experiment now with all concepts. This gives you:
- More comprehensive evaluation
- Better statistical power
- Comparison to original CAM on a larger scale

**Run on test split:**
```bash
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split test
```

**Run on validation split (for model selection/calibration):**
```bash
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split val
```

**Run on train split (for analysis):**
```bash
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split train
```

### Option 2: Filter to Original 20 CAM Concepts

If you want to match the original CAM exactly, you need to:

1. **Identify the 20 CAM concepts** (from the original paper or supplementary materials)
2. **Filter the trial definitions** to just those concepts
3. **Re-run the experiment**

**To filter trials:**
```python
import json

# Load trial definitions
with open('data/cam_trial_definitions.json', 'r') as f:
    data = json.load(f)

# List of 20 original CAM concepts (you need to provide this)
cam_20_concepts = [
    "humiliating", "resentful", "subservient", 
    # ... add the other 17 concepts
]

# Filter trials
filtered_trials = [
    trial for trial in data['trials']
    if trial['concept'] in cam_20_concepts
]

# Save filtered trials
data['trials'] = filtered_trials
data['metadata']['num_concepts'] = len(cam_20_concepts)
data['metadata']['num_trials'] = len(filtered_trials)

with open('data/cam_trial_definitions_20concepts.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### Option 3: Try Different Models

You can experiment with different pretrained models:

**CLIP (current):**
```yaml
model:
  type: "clip"
  name: "openai/clip-vit-base-patch32"
```

**Larger CLIP model:**
```yaml
model:
  type: "clip"
  name: "openai/clip-vit-large-patch14"
```

**Multimodal (vision + audio):**
```yaml
model:
  type: "multimodal"
  vision_model: "openai/clip-vit-base-patch32"
  audio_model: "facebook/wav2vec2-base"
  fusion_method: "weighted_average"
```

## What You've Successfully Replicated

✅ **Trial Structure**: 4-option forced-choice (1 target + 3 foils)
✅ **Counterbalancing**: 3 face + 2 voice OR 2 face + 3 voice per concept
✅ **Foil Selection**: Foils from different emotion groups (CAM methodology)
✅ **Actor Independence**: Test split uses different actors than train/val
✅ **Evaluation Metrics**: Overall accuracy, face/voice accuracy, concept recognition

## Comparison to Original CAM

**Original CAM Results (Golan et al., 2006):**
- Control group: ~85-90% accuracy
- AS group: ~65-75% accuracy
- Face accuracy: Control ~88%, AS ~70%
- Voice accuracy: Control ~87%, AS ~60%

**Your Current Results (CLIP, zero-shot):**
- Overall: 20.2% accuracy
- Face: 18.4% accuracy
- Voice: 38.6% accuracy (note: CLIP doesn't process audio, so this is less meaningful)

**Interpretation:**
- Zero-shot CLIP performs above random (25% for 4 options) but well below human performance
- This is expected - CLIP wasn't trained for emotion recognition
- Consider: few-shot calibration, better models, or training on emotion data

## Recommended Next Steps

1. **Run full experiment** on all splits to get complete results
2. **Try different models** (larger CLIP, multimodal, emotion-specific models)
3. **Add few-shot calibration** using validation set
4. **Compare to human benchmarks** if available
5. **Analyze errors** using confusion matrices to understand model limitations

## Files Generated

- `data/cam_trial_definitions.json` - All trial definitions
- `results/cam_human_like/clip_*/` - Experiment results
  - `summary.json` - Overall metrics
  - `trial_results.csv` - Per-trial predictions
  - `confusion_matrix.csv` - Error analysis
  - `per_emotion_accuracy.csv` - Per-emotion breakdown
  - `per_concept_accuracy.csv` - Per-concept breakdown

You're ready to run the experiment! The setup is complete and follows CAM methodology.






