# Debug: Process is Stopped

## The Problem

Process 12604 is **stopped** (status "TN"). It's not running - it's stuck or crashed.

## Solution: Kill and Run in Foreground

**Step 1: Kill the stopped process**

```bash
kill 12604
# Or
pkill -f finetune_vision_models
```

**Step 2: Run in foreground to see the error**

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned_test \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**This will show you:**
- Any import errors
- Any data loading errors
- Any other issues

**Once you see it working in foreground, then run in background with unbuffered output:**

```bash
python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned_test \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > resnet50_test.log 2>&1 &
disown
```
