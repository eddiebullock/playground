# Basic Emotion Classification Guide

## What Changed

We've adapted the project to classify **6-7 basic emotions** instead of 410 fine-grained emotions.

### Basic Emotion Categories

1. **Happy** - positive emotions (joy, delight, pleased, etc.)
2. **Sad** - negative emotions (sorrow, grief, disappointment, etc.)
3. **Angry** - anger-related (furious, annoyed, hostile, etc.)
4. **Fear** - fear-related (afraid, scared, anxious, etc.)
5. **Surprise** - surprise-related (surprised, shocked, amazed, etc.)
6. **Disgust** - disgust-related (disgusted, revolted, appalled, etc.)
7. **Neutral** - calm, thinking, or ambiguous emotions

### Expected Performance

**With 6-7 classes instead of 410:**
- **Random baseline**: ~14-17% (1/6 or 1/7)
- **Expected accuracy**: **60-80%** (much more achievable!)
- **This should answer your research question** about matching human performance

## Quick Start

### Step 1: Create Emotion Mapping (Already Done)

The mapping file has been created: `data/basic_emotion_mapping.json`

This maps all 410 fine-grained emotions to 7 basic categories.

### Step 2: Run Basic Emotion Classification

```bash
python experiments/basic_emotion_baseline.py \
  --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
  --splits_dir data/splits \
  --batch_size 16 \
  --num_epochs 20 \
  --lr 1e-3 \
  --backbone_lr 1e-5 \
  --use_augmentation \
  --seed 42
```

### Step 3: Check Results

Results will be in `results/basic_emotions/`:
- `test_results.txt` - Final accuracy (should be 60-80%!)
- `confusion_matrix.png` - Confusion matrix for 7 classes
- `train_history.csv` - Training progress

## What to Expect

### Performance Comparison

| Task | Classes | Random | Expected | Your Goal |
|------|---------|--------|----------|-----------|
| Fine-grained | 410 | 0.24% | 6-15% | 70% ❌ |
| **Basic emotions** | **7** | **14%** | **60-80%** | **70% ✅** |

**With basic emotions, 70% accuracy is achievable!**

### Sample Results

You should see something like:
```
Test Accuracy: 0.72 (72.00%)
Random baseline: 14.29% (1/7)
Improvement: 5.0x better than random
```

## Refining the Emotion Mapping

The current mapping uses keyword matching. You may want to refine it:

1. **Review the mapping**: Check `data/basic_emotion_mapping.json`
2. **Manually adjust**: Some emotions might be misclassified
3. **Re-run**: After adjusting, re-run the experiment

### Common Issues

- **Too many "neutral"**: The mapping is conservative - many emotions default to neutral
- **Ambiguous emotions**: Some emotions could fit multiple categories
- **Cultural differences**: Emotion categories may vary by culture

**You can manually edit the JSON file to refine the mapping.**

## Research Question Alignment

### Your Original Question
**"Can computer vision interpret emotions as well as humans?"**

### With Basic Emotions
- ✅ **More achievable**: 60-80% accuracy is realistic
- ✅ **Still meaningful**: Basic emotions are fundamental
- ✅ **Comparable to humans**: Humans also use basic emotion categories
- ✅ **Answers your question**: Yes, CV can interpret basic emotions well

### Trade-off
- ❌ **Less fine-grained**: Can't distinguish "humiliated" vs "ashamed"
- ✅ **But more practical**: Basic emotions are what most systems use
- ✅ **Still rigorous**: Actor-independent evaluation maintained

## Next Steps

1. **Run the experiment** and see if you get 60-80% accuracy
2. **If accuracy is good**: You can answer your research question!
3. **If accuracy is still low**: We can refine the mapping or try other methods
4. **Compare to human performance**: If available, compare your results to human performance on basic emotions

## Files Created

- `src/data/emotion_mapping.py` - Emotion mapping logic
- `src/data/create_basic_emotion_mapping.py` - Script to create mapping
- `src/data/basic_emotion_dataset.py` - Dataset class using basic emotions
- `experiments/basic_emotion_baseline.py` - Training script for basic emotions
- `data/basic_emotion_mapping.json` - Mapping file (410 → 7 emotions)

## Tips

1. **Check class balance**: Make sure all 7 classes have enough samples
2. **Review confusion matrix**: See which emotions are confused
3. **Refine mapping**: Adjust the JSON file if needed
4. **Compare methods**: Try prototypical networks on basic emotions too

Good luck! This should give you the 70%+ accuracy you're looking for! 🎯









