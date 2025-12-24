# Quick Start: Improved EU-Emotion Training

## What Was Changed

### ✅ 1. Regenerated Trials (10 per emotion)
- **New**: 267 total trials (213 train, 54 test)
- **Old**: 135 total trials (108 train, 27 test)
- **Improvement**: 2x more data for training

### ✅ 2. Increased Epochs (2 → 5)
- **New**: 5 epochs for better learning
- **Old**: 2 epochs (too few)
- **Expected**: Better emotion recognition

### ✅ 3. Added Prompt Templates
- **New**: "a photo of a person feeling [emotion]"
- **Old**: Raw emotion labels
- **Expected**: Better text-image alignment

### ✅ 4. Voice File Detection
- Code now discovers voice files (695 files found)
- Note: Voice files are .mp3 audio, CLIP can't process directly

## Training Status

Training is running in the background with improved settings.

**Monitor progress**:
```bash
tail -f results/eu_emotion_replication/training_log_v2.txt
```

**Check if still running**:
```bash
ps aux | grep finetune_clip_emotions
```

## Expected Results

| Metric | Previous (2 epochs, 108 trials) | Expected (5 epochs, 213 trials) |
|--------|--------------------------------|--------------------------------|
| Validation Accuracy | 33.33% | **50-60%** |
| Loss | 1.37 → 1.25 | **1.0-1.2** |
| Training Time | ~3 minutes | **~15-20 minutes** |

## After Training Completes

1. **Check results**:
   ```bash
   cat results/eu_emotion_replication/training_log_v2.txt | grep "Validation Accuracy"
   ```

2. **Evaluate on test set**:
   ```bash
   python experiments/cam_human_like/training/evaluate_on_cam.py \
       --model_path results/eu_emotion_replication/model_checkpoints_v2/best_model \
       --trial_definitions results/eu_emotion_replication/eu_emotion_trial_definitions_test.json \
       --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
       --dataset_type eu_emotion \
       --num_frames 8 \
       --use_multiframe
   ```

3. **Compare to previous**:
   - Previous: 33.33% validation accuracy
   - Target: 50-60% validation accuracy
   - Improvement: +17-27 percentage points

## Files Generated

- `results/eu_emotion_replication/eu_emotion_trial_definitions_train.json` (213 trials)
- `results/eu_emotion_replication/eu_emotion_trial_definitions_test.json` (54 trials)
- `results/eu_emotion_replication/model_checkpoints_v2/` (model checkpoints)
- `results/eu_emotion_replication/training_log_v2.txt` (training log)



