# Fine-Tuning Setup Complete! ✅

## What I've Created

### 1. Training Scripts

✅ **`training/finetune_vision_models.py`**
- Fine-tunes ResNet50, ViT, EfficientNet on EU-Emotion
- Supports all model variants
- Automatic train/val split handling
- Saves best model and emotion mappings

✅ **`training/create_train_val_splits.py`**
- Creates train/val splits from EU-Emotion dataset
- Balanced splits by emotion
- Actor-independent (if actor info available)

### 2. Updated Model Wrappers

✅ **`models/cnn_vit_wrappers.py`**
- Updated ResNet, ViT, EfficientNet to support fine-tuned models
- Automatically loads fine-tuned models if path provided
- Maps emotion predictions to candidate labels

✅ **`models/model_factory.py`**
- Updated to pass `fine_tuned_path` from config to models

### 3. Documentation

✅ **`training/FINETUNING_QUICKSTART.md`** - Quick start guide
✅ **`training/RUN_FINETUNING.md`** - Step-by-step instructions

## Quick Start: Fine-Tune ResNet50

### Step 1: Create Train/Val Splits (5 minutes)

```bash
python experiments/eu_emotion_model_comparison/training/create_train_val_splits.py \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir data/trial_definitions \
    --train_ratio 0.8
```

### Step 2: Fine-Tune ResNet50 (2-4 hours on Mac)

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

### Step 3: Update Config

Edit `experiments/eu_emotion_model_comparison/configs/comparison_config.yaml`:

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

## Expected Results

After fine-tuning:
- **ResNet50**: ~45-55% accuracy (vs 25% random, 55.6% CLIP)
- **ViT**: ~50-60% accuracy
- **EfficientNet**: ~45-55% accuracy

## What Gets Saved

After fine-tuning, you'll have:
```
models/resnet50_emotion_finetuned/
├── best_model.pth          # Best model checkpoint
├── emotion_mapping.json     # Emotion label mappings
└── checkpoint_epoch_*.pth   # Periodic checkpoints
```

## Next Steps

1. **Fine-tune ResNet50** (start here - easiest)
2. **Fine-tune ViT** (if time permits)
3. **Fine-tune EfficientNet** (if time permits)
4. **Evaluate all fine-tuned models** on test set
5. **Compare to CLIP and FER2013**
6. **Create ensemble** (combine models)

## Tips

- **Start with ResNet50** - easiest to fine-tune, good baseline
- **Monitor training** - watch validation accuracy
- **Adjust batch size** if out of memory (reduce to 8)
- **Train for 20 epochs** - usually sufficient
- **Save checkpoints** - can resume if interrupted

## Troubleshooting

**Out of memory?**
- Reduce batch size: `--batch_size 8`
- Reduce frames: `--num_frames 4`

**Slow training?**
- Use MPS on Mac: `--device mps`
- Use GPU if available: `--device cuda`

**Poor accuracy?**
- Train longer: `--num_epochs 30`
- Adjust learning rate: try `5e-5` or `2e-4`

## Files Created

- ✅ `training/finetune_vision_models.py` - Main fine-tuning script
- ✅ `training/create_train_val_splits.py` - Split creation script
- ✅ `training/FINETUNING_QUICKSTART.md` - Quick start guide
- ✅ `training/RUN_FINETUNING.md` - Detailed instructions
- ✅ Updated `models/cnn_vit_wrappers.py` - Support for fine-tuned models
- ✅ Updated `models/model_factory.py` - Pass fine_tuned_path from config

## Ready to Go!

Everything is set up. Start with ResNet50 fine-tuning - it's the easiest and will give you a good baseline to compare against CLIP and FER2013.

Good luck! 🚀
