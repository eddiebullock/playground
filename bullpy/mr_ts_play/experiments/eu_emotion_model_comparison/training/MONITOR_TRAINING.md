# How to Monitor Full Training

## Current Status

Training has started! The script creates separate log files for each model:

- **ResNet50**: `resnet50_training.log` (currently running)
- **ViT**: `vit_training.log` (will start after ResNet50)
- **EfficientNet**: `efficientnet_training.log` (will start after ViT)

## Monitor Current Training

**Watch ResNet50 training in real-time:**
```bash
tail -f resnet50_training.log
```

**Or check last 50 lines:**
```bash
tail -n 50 resnet50_training.log
```

## What You'll See

**Good progress indicators:**
- "Epoch X/10" - Shows current epoch
- "Training: XX%|████..." - Progress bar
- "Validation: XX%|████..." - Validation progress
- "✅ Saved best model" - Model checkpoint saved
- "Best validation accuracy: XX%" - Current best score

**Expected timeline:**
- **ResNet50**: ~30-40 minutes (10 epochs × ~3-4 min/epoch)
- **ViT**: ~50-60 minutes (10 epochs × ~5-6 min/epoch)
- **EfficientNet**: ~30-40 minutes (10 epochs × ~3-4 min/epoch)
- **Total**: ~2-2.5 hours

## Check All Logs

**See which models have completed:**
```bash
grep -E "(Training completed|Best validation accuracy)" resnet50_training.log vit_training.log efficientnet_training.log 2>/dev/null
```

## Quick Status Check

**See current epoch and progress:**
```bash
tail -n 5 resnet50_training.log
```

**Check if process is still running:**
```bash
ps aux | grep finetune_vision_models | grep -v grep
```
