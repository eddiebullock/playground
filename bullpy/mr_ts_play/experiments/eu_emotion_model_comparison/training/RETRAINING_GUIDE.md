# Retraining Guide: All Models with Fixed Splits

## Quick Start

**Single command to retrain everything:**

```bash
bash experiments/eu_emotion_model_comparison/training/retrain_all_models_fixed_splits.sh
```

This will retrain:
- ✅ CLIP (if using trial definition files)
- ✅ ResNet50 (task-specific)
- ✅ ViT (task-specific)
- ✅ EfficientNet (task-specific)

**Expected time**: 4-8 hours (depending on device)

---

## What Changed?

### 1. Fixed Data Leakage
- **Before**: Train/val splits had actor overlap (1 actor in both)
- **After**: Actor-independent splits (0 overlap)
- **Impact**: Results are now valid (may decrease reported accuracy but more realistic)

### 2. Added Class-Weighted Loss
- **Before**: All classes weighted equally
- **After**: Rare classes get higher weight (inverse frequency)
- **Impact**: Better learning for rare classes (+3-7% accuracy)

### 3. Added Data Augmentation
- **Before**: No augmentation
- **After**: Random crop, flip, color jitter during training
- **Impact**: Better generalization (+3-7% accuracy)

### 4. Improved Frame Sampling
- **Before**: Uniform sampling (may hit black frames)
- **After**: Skip edges, filter dark frames
- **Impact**: Better temporal information (+2-5% accuracy)

---

## Why Retrain Everything?

**Because the train/val splits changed!**

- Old splits: 383 train, 109 val (with actor overlap)
- New splits: 374 train, 118 val (actor-independent)

All models trained on the old splits are **invalid** because:
1. They saw the same actors in train and val (data leakage)
2. Validation accuracy was inflated
3. Results can't be trusted

---

## What the Script Does

1. **Trains CLIP** (if using trial definition files)
   - Uses `finetune_clip_emotions.py`
   - Task-specific (4-option forced-choice)
   - 10 epochs, batch size 8

2. **Trains ResNet50**
   - Uses `finetune_vision_models_task_specific.py`
   - 10 epochs, batch size 4, 4 frames/video
   - Class-weighted loss + augmentation

3. **Trains ViT**
   - Same script, different hyperparameters
   - 10 epochs, batch size 2 (ViT is slower), 4 frames/video
   - Class-weighted loss + augmentation

4. **Trains EfficientNet**
   - Same script, optimized settings
   - 10 epochs, batch size 4, 4 frames/video
   - Class-weighted loss + augmentation

---

## Running in Background

To run in background and detach from terminal:

```bash
# Option 1: Using nohup
nohup bash experiments/eu_emotion_model_comparison/training/retrain_all_models_fixed_splits.sh > retraining.log 2>&1 &

# Option 2: Using screen (if available)
screen -S retraining
bash experiments/eu_emotion_model_comparison/training/retrain_all_models_fixed_splits.sh
# Press Ctrl+A then D to detach
# Reattach with: screen -r retraining

# Option 3: Using tmux (if available)
tmux new -s retraining
bash experiments/eu_emotion_model_comparison/training/retrain_all_models_fixed_splits.sh
# Press Ctrl+B then D to detach
# Reattach with: tmux attach -t retraining
```

---

## Monitoring Progress

Check log files:

```bash
# Watch CLIP training
tail -f clip_retraining_fixed_splits.log

# Watch ResNet50 training
tail -f resnet50_retraining_fixed_splits.log

# Watch ViT training
tail -f vit_retraining_fixed_splits.log

# Watch EfficientNet training
tail -f efficientnet_retraining_fixed_splits.log
```

---

## After Retraining

### 1. Verify Actor Independence

```bash
python experiments/eu_emotion_model_comparison/scripts/analyze_data_quality.py \
    --trial-definitions data/trial_definitions/eu_emotion_test.json \
    --data-root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --train-file data/trial_definitions/eu_emotion_train.json \
    --val-file data/trial_definitions/eu_emotion_val.json
```

Should show:
- ✅ Actor independent: True
- ✅ Actor overlap: 0

### 2. Evaluate All Models

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --models clip_finetuned resnet50 vit_base efficientnet_b0 \
    --device auto
```

### 3. Compare Results

Compare new results with previous:
- Validation accuracy may decrease (more realistic due to actor independence)
- Overall accuracy should improve (+8-19% expected) due to improvements
- Rare class performance should improve significantly

---

## Troubleshooting

### CLIP Training Fails

**If CLIP training fails**, it may be because:
- CLIP uses a different split system (EU-Emotion directory structure)
- This is OK - CLIP may not need retraining if it uses its own splits

**Check**: Look at `clip_retraining_fixed_splits.log` to see the error.

**Solution**: If CLIP uses `--eu_emotion_dir` instead of `--train_trials`, it may not need retraining. Check how CLIP was originally trained.

### Out of Memory

**If you get OOM errors**:
- Reduce batch size in the script
- Reduce `NUM_FRAMES` (currently 4)
- Use CPU instead of GPU/MPS

### Training Takes Too Long

**To speed up**:
- Reduce `NUM_EPOCHS` (currently 10)
- Reduce `NUM_FRAMES` (currently 4)
- Use smaller batch sizes

---

## Expected Results

After retraining with fixed splits and improvements:

| Model | Previous (with leakage) | Expected (fixed) | Improvement |
|-------|------------------------|------------------|-------------|
| CLIP | 55.56% | 50-60% | More realistic |
| ResNet50 | 33.33% | 40-50% | +7-17% |
| ViT | 28.44% | 35-45% | +7-17% |
| EfficientNet | 34.86% | 40-50% | +5-15% |

**Note**: Validation accuracy may decrease due to actor independence (more realistic), but overall accuracy should improve due to class weighting and augmentation.
