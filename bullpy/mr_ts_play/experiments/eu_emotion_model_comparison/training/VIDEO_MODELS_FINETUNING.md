# Fine-Tuning I3D and TimeSformer Video Models

This guide explains how to fine-tune I3D and TimeSformer models on the EU-Emotion dataset using the task-specific approach (4-option forced-choice).

## Overview

Both I3D and TimeSformer are video models that process full video sequences (not just individual frames). They are currently implemented but need fine-tuning to work effectively on emotion recognition tasks.

## Setup

### Prerequisites

1. **Install required packages:**
   ```bash
   pip install pytorchvideo  # For I3D
   pip install transformers  # For TimeSformer (should already be installed)
   ```

2. **Verify train/val splits exist:**
   ```bash
   ls data/trial_definitions/eu_emotion_train.json
   ls data/trial_definitions/eu_emotion_val.json
   ```
   
   If they don't exist, create them:
   ```bash
   python experiments/eu_emotion_model_comparison/training/create_train_val_splits.py \
     --test_trials data/trial_definitions/eu_emotion_test.json \
     --data_root /path/to/EU_emotions \
     --output_dir data/trial_definitions
   ```

## Fine-Tuning

### Quick Start

Run the fine-tuning script for both models:

```bash
bash experiments/eu_emotion_model_comparison/training/finetune_i3d_timesformer.sh
```

This will:
1. Fine-tune I3D model (saves to `models/i3d_emotion_finetuned_task_specific/`)
2. Fine-tune TimeSformer model (saves to `models/timesformer_emotion_finetuned_task_specific/`)

### Individual Model Fine-Tuning

**Fine-tune I3D:**
```bash
python experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py \
    --model i3d \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root /path/to/EU_emotions \
    --output_dir models/i3d_emotion_finetuned_task_specific \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_frames 16 \
    --frame_sampling uniform
```

**Fine-tune TimeSformer:**
```bash
python experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py \
    --model timesformer \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root /path/to/EU_emotions \
    --output_dir models/timesformer_emotion_finetuned_task_specific \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_frames 8 \
    --frame_sampling uniform
```

### Training Parameters

- **num_epochs**: Number of training epochs (default: 20)
- **batch_size**: Batch size (default: 4, smaller for video models due to memory)
- **learning_rate**: Learning rate (default: 1e-4)
- **num_frames**: 
  - I3D: 16 frames (default)
  - TimeSformer: 8 frames (default)
- **frame_sampling**: "uniform", "temporal", or "keyframe" (default: "uniform")

## Testing

After fine-tuning, test the models on the test set:

**Test I3D:**
```bash
python experiments/eu_emotion_model_comparison/training/test_video_models.py \
    --model i3d \
    --model_path models/i3d_emotion_finetuned_task_specific/best_model.pth \
    --test_trials data/trial_definitions/eu_emotion_test.json \
    --data_root /path/to/EU_emotions
```

**Test TimeSformer:**
```bash
python experiments/eu_emotion_model_comparison/training/test_video_models.py \
    --model timesformer \
    --model_path models/timesformer_emotion_finetuned_task_specific/best_model.pth \
    --test_trials data/trial_definitions/eu_emotion_test.json \
    --data_root /path/to/EU_emotions
```

## Model Architecture

### Task-Specific Approach

Both models use the **task-specific fine-tuning approach** (same as CLIP):
- Each video is paired with 4 candidate labels (1 correct + 3 foils)
- Loss is cross-entropy over the 4 options (not all 27 emotions)
- Matches the evaluation format exactly (4-option forced-choice)

### I3D Model

- **Architecture**: I3D ResNet-50 (from pytorchvideo)
- **Input**: Video tensor (B, C, T, H, W) with 16 frames
- **Feature dimension**: 2048
- **Output**: 4 scores (one per candidate label)

### TimeSformer Model

- **Architecture**: TimeSformer base (from transformers)
- **Input**: 8 frames processed with VideoMAEImageProcessor
- **Feature dimension**: 768
- **Output**: 4 scores (one per candidate label)

## Expected Training Time

- **I3D**: ~4-8 hours (depending on GPU)
- **TimeSformer**: ~3-6 hours (depending on GPU)

Both models can be trained in parallel if you have multiple GPUs.

## Troubleshooting

### Out of Memory Errors

If you get OOM errors:
1. Reduce `batch_size` (try 2 or 1)
2. Reduce `num_frames` (I3D: 8, TimeSformer: 4)
3. Use gradient accumulation

### Model Loading Errors

- **I3D**: Make sure `pytorchvideo` is installed
- **TimeSformer**: Make sure `transformers` version supports TimeSformer (>=4.20.0)

### Data Path Issues

Make sure the `data_root` path is correct and contains the EU-Emotion dataset structure:
```
EU_emotions/
  emotions 1/
    HD Version - Face, Body, Social/
      Faces - HD Version/
        EDITED/
          EmotionName/
            *.mp4
```

## Next Steps

After fine-tuning and testing:
1. Update the video model wrappers to use fine-tuned models
2. Run full model comparison with fine-tuned I3D and TimeSformer
3. Compare results with other fine-tuned models (ResNet, ViT, etc.)
