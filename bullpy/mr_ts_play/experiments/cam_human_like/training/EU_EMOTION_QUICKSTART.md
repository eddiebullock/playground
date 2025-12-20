# EU-Emotion Dataset: Quick Start Guide

## Overview

EU-Emotion Stimulus Set is ideal for two-stage fine-tuning:
- **20 complex emotions** (matches CAM's 20 concepts perfectly!)
- **Multiple modalities**: Face, voice, body, context
- **External dataset** (no overlap with CAM test set)
- **Expected performance**: 70-80% on CAM after two-stage fine-tuning

## Step 0: Extract Dataset (if needed)

If your dataset is still in ZIP files, extract it first:

```bash
python experiments/cam_human_like/training/extract_eu_emotion.py \
    --source_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --target_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_extracted"
```

This will extract all ZIP files to a new directory. The script will:
- Skip already extracted files
- Handle split archives (if any)
- Show progress and summary

## Step 1: Test Dataset Loader

Before fine-tuning, verify the dataset loader works with your data:

```bash
python experiments/cam_human_like/training/test_eu_emotion.py \
    --eu_emotion_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_extracted"
```

This will:
- ✅ Detect your dataset structure automatically
- ✅ Show dataset statistics (samples, emotions, distribution)
- ✅ Test sample loading
- ✅ Verify compatibility with fine-tuning pipeline

**If the test passes**, proceed to Step 2. If it fails, check the error message - the loader is flexible and should handle most structures.

## Step 2: Quick Test Fine-Tuning (1-2 epochs)

**Note**: Use the extracted directory path, not the ZIP files directory.

Test the full pipeline locally with a small number of epochs:

```bash
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --eu_emotion_dir /path/to/your/eu_emotion/dataset \
    --output_dir models/clip_eu_emotion_test \
    --num_epochs 1 \
    --batch_size 4 \
    --device mps  # or 'cuda' if you have GPU
```

**Expected output:**
- Training loss should decrease
- Validation accuracy should be > 10% (random is ~5% for 20 classes)
- Model checkpoints saved to `models/clip_eu_emotion_test/`

## Step 3: Full Fine-Tuning (on HPC)

Once verified locally, run full training on HPC:

```bash
# Stage 1: EU-Emotion (external dataset)
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --eu_emotion_dir /home/eb2007/data/eu_emotion \
    --output_dir models/clip_eu_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --device cuda

# Stage 2: CAM (starting from EU-Emotion model)
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root /home/eb2007/data/CAM \
    --model_name models/clip_eu_emotion_finetuned/best_model \
    --output_dir models/clip_eu_cam_finetuned \
    --num_epochs 5 \
    --batch_size 16 \
    --device cuda
```

## Dataset Structure Support

The loader automatically detects common structures:

### Structure 1: Emotion-based (recommended)
```
eu_emotion/
├── train/
│   ├── happy/
│   │   ├── video1.mp4
│   │   └── video2.mov
│   ├── sad/
│   └── ...
├── test/
└── val/
```

### Structure 2: Modality-based
```
eu_emotion/
├── train/
│   ├── face/
│   │   ├── happy/
│   │   └── sad/
│   ├── voice/
│   └── body/
├── test/
└── val/
```

### Structure 3: Flat structure
```
eu_emotion/
├── train/
│   ├── happy_video1.mp4
│   ├── sad_video1.mp4
│   └── ...
├── test/
└── val/
```

## Options

### Modality Selection

If your dataset has multiple modalities, you can specify which to use:

```bash
--eu_emotion_modality face    # Face videos only (default)
--eu_emotion_modality voice   # Voice/audio only
--eu_emotion_modality body    # Body gesture videos only
--eu_emotion_modality all     # All modalities
```

### Video Frame Extraction

Control how many frames are extracted from videos:

```bash
--num_frames 8  # Default: 8 frames per video
```

## Troubleshooting

### "No samples found"
- Check that your dataset directory path is correct
- Verify files have supported extensions (`.mp4`, `.mov`, `.avi`, `.jpg`, `.png`)
- Check that emotion names can be extracted from filenames/directories

### "Video has no frames"
- Some video files might be corrupted or empty
- The loader will skip these automatically
- Check your video files are valid

### "Emotion extraction failed"
- The loader tries multiple strategies to extract emotion names
- If automatic detection fails, organize files into emotion-named directories
- Or rename files with emotion prefix (e.g., `happy_video1.mp4`)

## Expected Performance

| Stage | Dataset | Expected CAM Accuracy |
|-------|---------|---------------------|
| 0 | Zero-shot CLIP | 37% |
| 1 | EU-Emotion fine-tuned | 50-60% |
| 2 | EU-Emotion → CAM | **70-80%** |

## Next Steps

1. ✅ Test dataset loader with your data
2. ✅ Run quick local test (1-2 epochs)
3. ✅ Transfer EU-Emotion dataset to HPC
4. ✅ Run full two-stage fine-tuning on HPC
5. ✅ Evaluate fine-tuned model on CAM test set

## Questions?

If the loader doesn't work with your dataset structure, check:
- What directory structure do you have?
- What file formats are your videos/images?
- How are emotions labeled in your dataset?

The loader is designed to be flexible - it should work with most common structures!

