# Training Time Optimization

## Is 2-4 Hours Necessary?

**Short answer: No, you can reduce it significantly!**

## Ways to Speed Up Training

### Option 1: Fewer Epochs (Fastest)

**Default:** 20 epochs  
**Faster:** 10 epochs (often enough)

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 10 \  # Reduced from 20
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**Time savings:** ~50% (1-2 hours instead of 2-4)

**Trade-off:** Might be slightly lower accuracy, but often negligible

### Option 2: Larger Batch Size (If Memory Allows)

**Default:** batch_size=16  
**Faster:** batch_size=32 (if you have GPU/MPS)

**Time savings:** ~30-40% (faster per epoch)

**Trade-off:** Needs more memory

### Option 3: Fewer Frames

**Default:** num_frames=8  
**Faster:** num_frames=4

```bash
--num_frames 4  # Instead of 8
```

**Time savings:** ~50% (half the frames to process)

**Trade-off:** Might lose some temporal information

### Option 4: Early Stopping

The script already has learning rate scheduling. You could add early stopping:

- Stop if validation accuracy doesn't improve for 5 epochs
- Often converges in 10-15 epochs anyway

### Option 5: Start Small, Scale Up

**Test first with minimal training:**

```bash
# Quick test (30 minutes)
--num_epochs 5 \
--batch_size 32 \
--num_frames 4
```

**Then scale up if needed:**
- If accuracy is improving, train longer
- If it plateaus early, you're done

## Recommended: Start with 10 Epochs

**Best balance:**
- ✅ ~1-2 hours (instead of 2-4)
- ✅ Usually enough for convergence
- ✅ Can always train longer if needed

```bash
--num_epochs 10  # Start here
--batch_size 16
--num_frames 8
```

**Monitor validation accuracy:**
- If still improving at epoch 10 → train 10 more
- If plateaued → you're done!

## Realistic Time Estimates

### With Optimizations (10 epochs, batch_size=16)

- **MPS (Mac):** ~1-2 hours
- **CPU:** ~2-3 hours  
- **GPU:** ~30-60 minutes

### For All Models

- **ResNet50:** 1-2 hours
- **ViT:** 1.5-2.5 hours
- **EfficientNet:** 1-2 hours
- **Total:** ~3.5-6.5 hours (can run sequentially or in parallel)

## Quick Test Strategy

**Before full training, do a quick test:**

```bash
# 5 epochs, 30 minutes max
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 5 \
    --batch_size 16 \
    --device auto
```

**Check results:**
- If validation accuracy > 40% → Good, train full 10-20 epochs
- If validation accuracy < 30% → Check data/implementation
- If still improving → Train longer

## Bottom Line

**2-4 hours is NOT necessary:**
- ✅ Start with **10 epochs** (~1-2 hours)
- ✅ Monitor validation accuracy
- ✅ Stop early if it plateaus
- ✅ Can always train longer if needed

**Most models converge in 10 epochs anyway!**
