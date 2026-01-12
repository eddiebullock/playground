# Fix: Process Suspended Issue

## The Problem

The process got suspended with "suspended (tty output)". This happens when nohup doesn't properly redirect output.

## Solution: Kill and Restart

### Step 1: Kill the suspended process

```bash
# Find the process
jobs

# Kill it (if it's job [1])
kill %1

# Or find and kill by PID
pkill -f finetune_vision_models
```

### Step 2: Use a better command

**Option A: Use `disown` instead of nohup**

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
    --device auto > resnet50_test.log 2>&1 &
disown
```

**Option B: Use `screen` or `tmux` (best for long runs)**

```bash
# Start a screen session
screen -S training

# Run the command (without nohup)
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

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r training
```

**Option C: Fix nohup (redirect stderr properly)**

```bash
nohup python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned_test \
    --num_epochs 1 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto </dev/null >resnet50_test.log 2>&1 &
```

## Recommended: Use `disown` Method

**Simplest and most reliable:**

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
    --device auto > resnet50_test.log 2>&1 &
disown
```

**Then check progress:**
```bash
tail -f resnet50_test.log
```
