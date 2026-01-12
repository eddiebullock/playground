# Retraining Plan: Task-Specific Models

## Current Status

### ✅ ResNet50
- **Status**: Completed (36.70% validation accuracy at epoch 1)
- **Issue**: Training accuracy showed 0% (bug now fixed)
- **Retrain?**: **YES** - To get proper training accuracy tracking and potentially better results

### ⚠️ ViT
- **Status**: Was running but very slow (~128s/iteration, 12-16 hours remaining)
- **Issue**: Too slow, needs optimization
- **Retrain?**: **YES** - With optimized settings (smaller batch, fewer frames)

### ❌ EfficientNet
- **Status**: Not started yet
- **Retrain?**: **NO** - Will use optimized settings from the start

## Optimization Changes

**Speed improvements:**
- **Batch size**: 8 → 4 (ResNet/EfficientNet), 8 → 2 (ViT)
- **Frames per video**: 8 → 4 (2x faster frame extraction)
- **Epochs**: 12 → 10 (slightly fewer)

**Expected speedup:**
- ResNet50: ~2x faster (smaller batch + fewer frames)
- ViT: ~4x faster (batch 2 + fewer frames) = ~3-4 hours instead of 12-16 hours
- EfficientNet: ~2x faster

## Retraining Commands

### Option 1: Retrain All (Recommended)
```bash
bash experiments/eu_emotion_model_comparison/training/train_all_models_task_specific_optimized.sh
```

This will:
1. Retrain ResNet50 (with fixed training accuracy tracking)
2. Retrain ViT (with optimized settings)
3. Train EfficientNet (with optimized settings)

**Total time**: ~6-8 hours (vs 20+ hours before)

### Option 2: Keep ResNet50, Only Retrain ViT + EfficientNet

If you want to keep the current ResNet50 model:
1. Comment out ResNet50 in the script
2. Run only ViT and EfficientNet

## Why Retrain ResNet50?

Even though it completed, we should retrain because:
1. **Training accuracy bug fixed** - Now we'll see real training metrics
2. **Optimized settings** - Fewer frames might actually work better (less overfitting)
3. **Consistency** - All models trained with same settings for fair comparison
4. **Time is reasonable** - With optimizations, ResNet50 will be ~30-40 minutes

## Expected Results After Optimization

**Time estimates:**
- ResNet50: ~30-40 minutes (10 epochs, batch 4, 4 frames)
- ViT: ~3-4 hours (10 epochs, batch 2, 4 frames)
- EfficientNet: ~30-40 minutes (10 epochs, batch 4, 4 frames)
- **Total**: ~5-6 hours (much better than 20+ hours!)

**Performance:**
- Should be similar or slightly better (fewer frames = less overfitting risk)
- Training accuracy will now be tracked correctly
- All models will have consistent training settings
