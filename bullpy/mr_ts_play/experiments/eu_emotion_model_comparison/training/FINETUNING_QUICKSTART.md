# Fine-Tuning Quick Start Guide

## Overview

This guide will help you fine-tune ResNet50, ViT, and EfficientNet models on the EU-Emotion dataset.

## Step 1: Create Train/Val Splits

First, create train/val splits from your EU-Emotion dataset:

```bash
python experiments/eu_emotion_model_comparison/training/create_train_val_splits.py \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir data/trial_definitions \
    --train_ratio 0.8
```

This will create:
- `data/trial_definitions/eu_emotion_train.json`
- `data/trial_definitions/eu_emotion_val.json`

## Step 2: Fine-Tune a Model

### ResNet50 (Recommended - Easiest)

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**Expected time:**
- CPU: ~4-8 hours
- GPU: ~1-2 hours
- MPS (Mac): ~2-4 hours

**Expected accuracy:** ~45-55% (after fine-tuning)

### ViT (Vision Transformer)

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model vit_base \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/vit_emotion_finetuned \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --device auto
```

**Note:** Requires `timm` package: `pip install timm`

**Expected accuracy:** ~50-60%

### EfficientNet

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model efficientnet_b0 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/efficientnet_b0_emotion_finetuned \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**Note:** Requires `timm` package: `pip install timm`

**Expected accuracy:** ~45-55%

## Step 3: Update Configuration

After fine-tuning, update the config to use your fine-tuned model:

Edit `experiments/eu_emotion_model_comparison/configs/comparison_config.yaml`:

```yaml
model_configs:
  resnet50:
    fine_tuned_path: "models/resnet50_emotion_finetuned/best_model.pth"
  
  vit_base:
    fine_tuned_path: "models/vit_emotion_finetuned/best_model.pth"
  
  efficientnet_b0:
    fine_tuned_path: "models/efficientnet_b0_emotion_finetuned/best_model.pth"
```

## Step 4: Evaluate Fine-Tuned Models

Run evaluation with fine-tuned models:

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --models resnet50 vit_base efficientnet_b0 \
    --device auto \
    --skip-failed
```

## Tips

### Batch Size
- **GPU**: Use 16-32
- **CPU/MPS**: Use 8-16
- **Out of memory?** Reduce batch size

### Learning Rate
- **ResNet/EfficientNet**: Start with 1e-4
- **ViT**: Start with 5e-5 (lower, more sensitive)
- **Too high?** Loss will be unstable
- **Too low?** Training will be slow

### Number of Epochs
- Start with 20 epochs
- Monitor validation accuracy
- Stop early if validation accuracy plateaus

### Frame Sampling
- `uniform`: Evenly spaced frames (default, recommended)
- `temporal`: First, middle, last frames
- `keyframe`: Middle portion of video

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 8`
- Reduce number of frames: `--num_frames 4`
- Use CPU: `--device cpu` (slower but no memory issues)

### Slow Training
- Use GPU if available: `--device cuda`
- Use MPS on Mac: `--device mps`
- Reduce batch size if causing memory issues

### Poor Accuracy
- Train for more epochs: `--num_epochs 30`
- Adjust learning rate (try 5e-5 or 2e-4)
- Check train/val split (should be balanced)

### Model Not Loading
- Check path: `fine_tuned_path` must point to `best_model.pth`
- Check emotion mapping: Should be in same directory as model
- Verify model was trained successfully

## Expected Results

After fine-tuning, you should see:
- **ResNet50**: ~45-55% accuracy (vs 25% random)
- **ViT**: ~50-60% accuracy
- **EfficientNet**: ~45-55% accuracy

These are lower than CLIP (55.6%) because:
- CLIP is a vision-language model (better semantic understanding)
- CLIP was already fine-tuned on emotions
- These models start from ImageNet (general vision, not emotions)

But they're still valuable for:
- Baseline comparison
- Ensemble methods
- Understanding which architectures work best

## Next Steps

After fine-tuning:
1. Evaluate on test set
2. Compare to CLIP and FER2013
3. Create ensemble (combine models)
4. Analyze per-emotion performance
