# FER2013 Dataset Setup Guide

## Overview

FER2013 (Facial Expression Recognition 2013) is a standard emotion recognition dataset:
- **35,887 images** (48×48 pixels, grayscale)
- **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Splits**: Training (~28,709), PublicTest (~3,589), PrivateTest (~3,589)

This is an **external dataset** (no overlap with CAM), making it ideal for rigorous fine-tuning.

## Quick Setup

### Option 1: Using Kaggle API (Recommended)

**Prerequisites:**
1. Install Kaggle API: `pip install kaggle`
2. Set up Kaggle credentials:
   ```bash
   # Download kaggle.json from https://www.kaggle.com/settings
   # Place it at ~/.kaggle/kaggle.json
   # Set permissions: chmod 600 ~/.kaggle/kaggle.json
   ```

**Download and setup:**
```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
python experiments/cam_human_like/training/download_fer2013.py \
    --output_dir data/fer2013 \
    --use_kaggle
```

### Option 2: Manual Download

1. **Download from Kaggle:**
   - Go to: https://www.kaggle.com/datasets/msambare/fer2013
   - Click "Download" (requires Kaggle account)
   - Extract `fer2013.zip` to get `fer2013.csv`

2. **Run setup script:**
   ```bash
   python experiments/cam_human_like/training/download_fer2013.py \
       --input_dir /path/to/extracted/fer2013 \
       --output_dir data/fer2013
   ```

### Option 3: Direct CSV Placement

If you already have `fer2013.csv`:
```bash
# Place fer2013.csv in data/fer2013/
cp fer2013.csv data/fer2013/

# Run setup
python experiments/cam_human_like/training/download_fer2013.py \
    --output_dir data/fer2013
```

## What the Script Does

1. **Downloads** FER2013 from Kaggle (if `--use_kaggle`)
2. **Reads** `fer2013.csv` with pixel strings
3. **Converts** pixel strings to 48×48 images
4. **Resizes** to 224×224 (CLIP input size)
5. **Organizes** into directory structure:
   ```
   data/fer2013/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   ├── test/
   │   └── (same structure)
   └── val/
       └── (same structure)
   ```

## Expected Output

```
Downloading from Kaggle...
Authenticating with Kaggle API...
Downloading FER2013 dataset from Kaggle...
Downloaded to data/fer2013_raw

Reading FER2013 CSV: data/fer2013_raw/fer2013.csv
Converting pixel strings to images...
Converting images: 100%|████████| 35887/35887 [02:30<00:00, 238.45it/s]

Conversion complete!
Train images: 28709
Test images: 3589
Val images: 3589

Dataset saved to: data/fer2013
```

## Verification

Check the dataset structure:
```bash
ls data/fer2013/train/
# Should show: angry, disgust, fear, happy, neutral, sad, surprise

ls data/fer2013/train/happy/ | head -5
# Should show: 000000.jpg, 000001.jpg, etc.
```

## Using FER2013 for Fine-Tuning

After setup, use it for fine-tuning:
```bash
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --fer2013_dir data/fer2013 \
    --output_dir models/clip_fer2013_finetuned \
    --num_epochs 10 \
    --device cuda  # or mps, cpu
```

## Troubleshooting

### Kaggle API Authentication Error
```bash
# Make sure kaggle.json is in ~/.kaggle/
# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

### CSV Not Found
- Check that `fer2013.csv` exists in the input directory
- The script searches recursively, but verify the path

### Out of Memory
- FER2013 is ~35K images, conversion uses ~2-3GB RAM
- If issues, process in batches (modify script)

### Slow Conversion
- Normal: ~2-3 minutes for 35K images
- If very slow, check disk I/O

## Dataset Size

- **Raw CSV**: ~15 MB
- **Converted images**: ~500-800 MB (depends on JPEG quality)
- **Time to convert**: ~2-3 minutes

## Next Steps

After FER2013 is set up:
1. ✅ Test CAM fine-tuning locally (1-2 epochs)
2. ✅ Set up FER2013 (you are here)
3. ⏭️ Run full CAM fine-tuning on HPC
4. ⏭️ Run FER2013 fine-tuning on HPC
5. ⏭️ Compare results: CAM vs FER2013 fine-tuning





