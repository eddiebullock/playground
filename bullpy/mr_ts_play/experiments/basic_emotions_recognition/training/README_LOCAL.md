# Local Training Guide: Basic Emotions Recognition

This guide explains how to run the basic emotions fine-tuning experiments locally on your laptop.

## Overview

The experiment fine-tunes CLIP models on Ekman's 6 basic emotions (+ neutral = 7 classes) for both CAM and EU-Emotion datasets.

## Prerequisites

1. **Data locations** (update paths in scripts if needed):
   - CAM data: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`
   - EU-Emotion data: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`

2. **Python environment**: Activate your virtual environment (if using one)

3. **Dependencies**: Install required packages (see main README.md)

## Quick Start

### Run CAM Basic Emotions Training

```bash
./experiments/basic_emotions_recognition/training/run_local_cam.sh
```

This will:
1. Generate basic emotion trials from CAM data
2. Fine-tune CLIP on 7 basic emotions
3. Evaluate the model on test set

### Run EU-Emotion Basic Emotions Training

```bash
./experiments/basic_emotions_recognition/training/run_local_eu_emotion.sh
```

This will:
1. Generate basic emotion trials from EU-Emotion data (or use existing)
2. Fine-tune CLIP on 7 basic emotions
3. Evaluate the model on test set

## Device Auto-Detection

The scripts automatically detect your device:
- **MPS** (Mac M1/M2 GPU): ~2-4 hours per job
- **CUDA** (NVIDIA GPU): ~30-60 minutes per job
- **CPU**: ~3-5 hours per job

Batch sizes are automatically optimized:
- CPU: batch_size=4
- MPS: batch_size=8
- CUDA: batch_size=16

## Output Locations

- **CAM model**: `models/basic_emotions_cam/best_model/`
- **CAM evaluation**: `models/basic_emotions_cam/evaluation/`
- **EU-Emotion model**: `models/basic_emotions_eu_emotion/best_model/`
- **EU-Emotion evaluation**: `models/basic_emotions_eu_emotion/evaluation/`

## Configuration

Default training settings:
- **Epochs**: 12 (with early stopping)
- **Learning rate**: 5e-5
- **Weight decay**: 0.01
- **Frames**: 16 per video
- **Early stopping**: patience=5, min_delta=0.001

## Troubleshooting

### Path Issues

If you get "data not found" errors, update the paths in the scripts:
- `CAM_DATA_ROOT` in `run_local_cam.sh`
- `EU_EMOTIONS_DATA_ROOT` in `run_local_eu_emotion.sh`

### Out of Memory

If you get OOM errors:
- Reduce `BATCH_SIZE` in the script
- Reduce `NUM_FRAMES` (try 8 instead of 16)
- Use CPU instead of GPU

### Slow Training

- Check device detection: The script prints the detected device
- GPU training is 10-20x faster than CPU
- MPS (Mac GPU) is 5-10x faster than CPU

## Next Steps

After training completes, you can:
1. **Run LLM augmentation experiments** (see `llm_augmentation/` directory)
2. **Compare basic vs. complex emotions** performance
3. **Analyze results** in the evaluation directories

## Manual Execution

If you prefer to run steps manually:

```bash
# Step 1: Generate trials
python experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py \
    --dataset_type cam \
    --input_trials data/cam_trial_definitions_20concepts.json \
    --mapping_file data/basic_emotion_mapping.json \
    --output_dir models/basic_emotions_cam \
    --train_ratio 0.8 \
    --seed 42

# Step 2: Fine-tune
python experiments/basic_emotions_recognition/training/finetune_basic_emotions.py \
    --train_trials models/basic_emotions_cam/cam_basic_emotions_train.json \
    --val_trials models/basic_emotions_cam/cam_basic_emotions_test.json \
    --data_root "/path/to/CAM/data" \
    --output_dir models/basic_emotions_cam/best_model \
    --num_epochs 12 \
    --batch_size 8 \
    --device mps  # or cuda or cpu

# Step 3: Evaluate
python experiments/basic_emotions_recognition/training/evaluate_basic_emotions.py \
    --model_path models/basic_emotions_cam/best_model \
    --test_trials models/basic_emotions_cam/cam_basic_emotions_test.json \
    --data_root "/path/to/CAM/data" \
    --output_dir models/basic_emotions_cam/evaluation
```


