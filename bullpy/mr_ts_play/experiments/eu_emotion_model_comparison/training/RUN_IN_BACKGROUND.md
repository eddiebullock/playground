# Running Fine-Tuning in the Background

## Command to Run in Background

**This will run ResNet50 fine-tuning in the background and save logs:**

```bash
nohup python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > resnet50_training.log 2>&1 &
```

**What this does:**
- `nohup` - Keeps running even if you disconnect
- `> resnet50_training.log` - Saves output to log file
- `2>&1` - Saves errors to same log file
- `&` - Runs in background

## How to Check Progress

### While it's running:

**View live log:**
```bash
tail -f resnet50_training.log
```

**View last 50 lines:**
```bash
tail -n 50 resnet50_training.log
```

**Check if still running:**
```bash
ps aux | grep finetune_vision_models
```

### After it completes:

**Check final results:**
```bash
tail -n 100 resnet50_training.log
```

**Check if model was saved:**
```bash
ls -lh models/resnet50_emotion_finetuned/best_model.pth
```

## Stop the Process (if needed)

**Find the process ID:**
```bash
ps aux | grep finetune_vision_models
```

**Kill it:**
```bash
kill <process_id>
```

## For All Models (Sequential)

**If you want to run all models one after another:**

**ResNet50:**
```bash
nohup python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > resnet50_training.log 2>&1 &
```

**Wait for it to finish, then ViT:**
```bash
nohup python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model vit_base \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/vit_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --device auto > vit_training.log 2>&1 &
```

**Then EfficientNet:**
```bash
nohup python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model efficientnet_b0 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/efficientnet_b0_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > efficientnet_training.log 2>&1 &
```

## Tips

1. **Keep laptop plugged in** - Training uses power
2. **Don't close terminal** - Process will keep running with nohup
3. **Check logs when you return** - See if training completed successfully
4. **Verify model saved** - Check `models/resnet50_emotion_finetuned/best_model.pth` exists

## What to Look For in Logs

**Good signs:**
- "Epoch 1/10", "Epoch 2/10", etc.
- "Val Acc: XX.XX%" increasing
- "✅ Saved best model"
- "Training completed!"

**Bad signs:**
- "Out of memory" errors
- "File not found" errors
- Process stopped early
