# Local Testing Guide: EU-Emotion Fine-Tuning Pipeline

This guide walks you through testing the EU-Emotion fine-tuning pipeline locally on your MacBook Air 2025 before moving to HPC.

## Overview

**Goal**: Fine-tune CLIP on EU-Emotion (faces only) and evaluate on CAM test set, comparing to the 37% zero-shot baseline.

**Pipeline**:
1. Extract only face files from EU-Emotion ZIPs (~20-40GB instead of 213GB)
2. Test dataset loader
3. Fine-tune CLIP on EU-Emotion (1-2 epochs, local test)
4. Evaluate fine-tuned model on CAM test set
5. Compare results to 37% baseline

## Prerequisites

- EU-Emotion dataset ZIP files downloaded to:
  `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`
- CAM dataset available at:
  `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`
- Python environment with PyTorch, transformers, etc.

## Quick Start

### Option 1: Automated Pipeline (Recommended)

Run the complete test pipeline:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
./experiments/cam_human_like/training/test_eu_finetuning_local.sh
```

This script will:
1. Extract face files (if not already extracted)
2. Test the dataset loader
3. Run fine-tuning (1-2 epochs)
4. Provide instructions for CAM evaluation

### Option 2: Manual Steps

#### Step 1: Copy Face Files Only

```bash
python experiments/cam_human_like/training/copy_eu_faces_only.py \
    --source_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --target_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces"
```

**Expected**: Copies only face video files from the already-extracted dataset, reducing size from ~213GB to ~20-40GB.

**Note**: The EU-Emotion dataset is already extracted. This script finds face files in:
- `emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/EmotionName/*.mp4`
- `emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/EmotionName/*.mov`

#### Step 2: Test Dataset Loader

```bash
python experiments/cam_human_like/training/test_eu_emotion.py \
    --eu_emotion_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces" \
    --eu_emotion_modality face \
    --num_frames 8
```

**Expected**: Prints dataset statistics (number of samples, emotions found).

#### Step 3: Fine-Tune CLIP (1-2 Epochs, Test Run)

```bash
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --eu_emotion_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces" \
    --eu_emotion_modality face \
    --output_dir models/clip_eu_emotion_local_test \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device mps \
    --num_frames 8 \
    --use_multiframe
```

**Time Estimates**:
- **MPS (Mac M3)**: ~20-40 minutes (2 epochs)
- **CPU**: ~4-8 hours (2 epochs)

**Expected**: 
- Training loss decreases
- Validation accuracy improves (even if small)
- Model saved to `models/clip_eu_emotion_local_test/best_model/`

#### Step 4: Evaluate on CAM Test Set

```bash
python experiments/cam_human_like/training/evaluate_on_cam.py \
    --model_path models/clip_eu_emotion_local_test/best_model \
    --trial_definitions data/cam_trial_definitions_20concepts.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
    --split test \
    --device mps \
    --num_frames 8 \
    --use_multiframe
```

**Expected**: 
- Overall accuracy printed (should be > 37%)
- Comparison to baseline
- Results saved to JSON file

## Multi-Frame Processing

The pipeline now supports **multiple frames per video** (not just middle frame):

- **`--use_multiframe`** (default): Processes all frames and averages features
  - Better temporal information
  - More robust predictions
  - Slower training

- **`--single_frame`**: Uses only middle frame
  - Faster training
  - Less temporal information
  - Good for quick tests

## Expected Results

### Baseline (Zero-Shot CLIP)
- **Overall Accuracy**: 37.0%
- **Face Accuracy**: ~37%
- **Voice Accuracy**: ~27%

### After EU-Emotion Fine-Tuning (Expected)
- **Overall Accuracy**: 50-60% (Stage 1: EU-Emotion only)
- **Face Accuracy**: 55-65%
- **Voice Accuracy**: 40-50%

### After Two-Stage Fine-Tuning (Future)
- **Overall Accuracy**: 70-80% (Stage 2: EU-Emotion → CAM)
- **Face Accuracy**: 75-85%
- **Voice Accuracy**: 60-70%

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
--batch_size 4  # or even 2
```

Reduce number of frames:
```bash
--num_frames 4  # instead of 8
```

### Slow Training

Use single frame mode:
```bash
--single_frame  # faster, less accurate
```

Or reduce epochs for testing:
```bash
--num_epochs 1  # just verify it works
```

### Dataset Loading Errors

Check that face files were extracted:
```bash
ls "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces"
```

Verify dataset structure:
```bash
python experiments/cam_human_like/training/test_eu_emotion.py \
    --eu_emotion_dir "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces" \
    --eu_emotion_modality face
```

## Next Steps

After successful local testing:

1. **Full Training on HPC**: Transfer extracted face files to HPC and run full training (10 epochs)
2. **Two-Stage Fine-Tuning**: Fine-tune on EU-Emotion, then on CAM train split
3. **Architecture Improvements**: Test different frame aggregation methods (attention, max pooling)

## Files Created

- `extract_eu_faces_only.py`: Extracts only face files from ZIPs
- `test_eu_finetuning_local.sh`: Automated test pipeline
- `evaluate_on_cam.py`: Evaluates fine-tuned model on CAM
- Updated `eu_emotion_dataset.py`: Supports multiple frames
- Updated `finetune_clip_emotions.py`: Handles multi-frame processing

## Notes

- **Face files only**: We extract only face videos (~20-40GB) instead of the full dataset (213GB)
- **Multi-frame**: Uses all frames from each video and averages features (better than middle frame only)
- **Local first**: Test locally to verify pipeline works before moving to HPC
- **HPC later**: Full training (10 epochs) should be done on HPC for speed

