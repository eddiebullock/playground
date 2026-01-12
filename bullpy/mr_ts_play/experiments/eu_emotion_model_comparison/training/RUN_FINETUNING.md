# Run Fine-Tuning: Step-by-Step Guide

## Quick Start (ResNet50 - Recommended First)

### Step 1: Create Train/Val Splits

```bash
python experiments/eu_emotion_model_comparison/training/create_train_val_splits.py \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir data/trial_definitions \
    --train_ratio 0.8 \
    --seed 42
```

This creates:
- `data/trial_definitions/eu_emotion_train.json`
- `data/trial_definitions/eu_emotion_val.json`

### Step 2: Fine-Tune ResNet50

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
- MPS (Mac): ~2-4 hours
- CPU: ~4-8 hours
- GPU: ~1-2 hours

**Expected accuracy:** ~45-55% (vs 25% random)

### Step 3: Update Config

After training completes, update `experiments/eu_emotion_model_comparison/configs/comparison_config.yaml`:

```yaml
model_configs:
  resnet50:
    fine_tuned_path: "models/resnet50_emotion_finetuned/best_model.pth"
```

### Step 4: Evaluate

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --models resnet50 \
    --device auto
```

## Fine-Tune Other Models

### ViT (Vision Transformer)

**Requires:** `pip install timm`

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

**Expected accuracy:** ~50-60%

### EfficientNet

**Requires:** `pip install timm`

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

**Expected accuracy:** ~45-55%

## Batch Fine-Tuning (All Models)

You can fine-tune multiple models sequentially:

```bash
# ResNet50
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 20 --batch_size 16 --device auto

# ViT
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model vit_base \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/vit_emotion_finetuned \
    --num_epochs 20 --batch_size 8 --device auto

# EfficientNet
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model efficientnet_b0 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/efficientnet_b0_emotion_finetuned \
    --num_epochs 20 --batch_size 16 --device auto
```

## Monitor Training

Training will show:
- Progress bar for each epoch
- Train loss and accuracy
- Validation loss and accuracy
- Best model saved automatically

**Example output:**
```
Epoch 1/20
Training: 100%|████████| 45/45 [02:15<00:00, loss=2.1234, acc=35.67%]
Validation: 100%|████████| 12/12 [00:15<00:00, loss=1.9876, acc=42.34%]
Train Loss: 2.1234, Train Acc: 35.67%
Val Loss: 1.9876, Val Acc: 42.34%
✅ Saved best model (Val Acc: 42.34%)
```

## After Fine-Tuning

1. **Update config** with fine-tuned model paths
2. **Run evaluation** on test set
3. **Compare results** to CLIP and FER2013
4. **Analyze per-emotion** performance

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 8`
- Reduce frames: `--num_frames 4`

### Slow Training
- Use GPU/MPS: `--device mps` or `--device cuda`
- Reduce batch size if causing memory issues

### Poor Accuracy
- Train longer: `--num_epochs 30`
- Adjust learning rate: try `5e-5` or `2e-4`
- Check data splits are balanced
