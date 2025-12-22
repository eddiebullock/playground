# EU-Emotion Stimulus Set: Setup Guide

## Obtaining the Dataset

**Contact**: admin@autismresearchcentre.com
- Free for research purposes
- Commercial use prohibited
- Request: "EU-Emotion Stimulus Set for research"

## Dataset Structure (Expected)

Based on the description, EU-Emotion likely has:
- **20 emotions/mental states**
- **Multiple modalities**: Face, voice, body gestures, contextual scenes
- **Video format** (likely, given "stimulus set" and multiple modalities)
- **Diverse actors**: Child and adult

## Integration Plan

### Step 1: Dataset Loader

Create `eu_emotion_dataset.py` similar to `fer2013_dataset.py`:

```python
class EUEmotionDataset(Dataset):
    """
    Dataset loader for EU-Emotion Stimulus Set.
    
    Expected structure:
    eu_emotion/
    ├── train/
    │   ├── emotion1/
    │   │   ├── face_videos/
    │   │   ├── voice_audio/
    │   │   └── body_scenes/
    │   └── ...
    ├── test/
    └── val/
    """
```

### Step 2: Fine-Tuning Script Update

Update `finetune_clip_emotions.py` to support EU-Emotion:

```python
# Add EU-Emotion option
parser.add_argument('--eu_emotion_dir', type=str, help='Path to EU-Emotion dataset')

# In main():
if args.eu_emotion_dir:
    train_dataset = EUEmotionDataset(args.eu_emotion_dir, split='train')
    val_dataset = EUEmotionDataset(args.eu_emotion_dir, split='test')
```

### Step 3: Two-Stage Fine-Tuning

```bash
# Stage 1: EU-Emotion (external dataset)
python finetune_clip_emotions.py \
    --eu_emotion_dir data/eu_emotion \
    --output_dir models/clip_eu_emotion_finetuned \
    --num_epochs 10

# Stage 2: CAM (starting from EU-Emotion model)
python finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/path/to/cam/stimuli" \
    --model_name models/clip_eu_emotion_finetuned/best_model \
    --output_dir models/clip_eu_cam_finetuned \
    --num_epochs 5  # Fewer epochs (already trained on emotions)
```

## Expected Results

| Stage | Dataset | Expected CAM Accuracy |
|-------|---------|---------------------|
| 0 | Zero-shot | 37% |
| 1 | EU-Emotion | 50-60% |
| 2 | EU-Emotion → CAM | **70-80%** |

## Advantages Over FER2013

1. **20 emotions** (matches CAM) vs 7 basic emotions
2. **Complex mental states** vs basic emotions
3. **Multiple modalities** (face, voice, body, context) vs face only
4. **Video format** vs static images
5. **Same research context** (Autism Research Centre)

## Timeline

1. **Now**: Request EU-Emotion dataset
2. **After receiving**: Set up dataset loader
3. **Test**: Run EU-Emotion fine-tuning (1-2 epochs)
4. **Full training**: EU-Emotion → CAM two-stage on HPC
5. **Evaluate**: Compare EU-Emotion → CAM vs CAM only

## Recommendation

**Use EU-Emotion for two-stage fine-tuning!**

This gives you:
- ✅ Maximum performance (70-80% expected)
- ✅ Methodological rigor (external dataset first)
- ✅ Perfect emotion alignment (20 vs 20)
- ✅ Complex emotions (not basic)

This is the best approach for your PhD thesis!





