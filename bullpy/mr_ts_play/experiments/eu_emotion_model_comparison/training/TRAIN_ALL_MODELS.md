# Train All Models: Sequential vs Parallel

## Option 1: Sequential Training (Recommended) ✅

**Train all models one after another automatically:**

```bash
bash experiments/eu_emotion_model_comparison/training/train_all_models.sh
```

**What this does:**
- Trains ResNet50 first (~1-2 hours)
- Then trains ViT (~1.5-2.5 hours)
- Then trains EfficientNet (~1-2 hours)
- Total: ~3.5-6.5 hours
- Each model waits for the previous one to finish

**Advantages:**
- ✅ No memory issues
- ✅ Can monitor each one
- ✅ Automatic - just run and walk away
- ✅ Logs saved separately for each model

**To run in background:**
```bash
nohup bash experiments/eu_emotion_model_comparison/training/train_all_models.sh > all_models_training.log 2>&1 &
disown
```

## Option 2: Train All at Once (NOT Recommended) ❌

**Why NOT to do this:**
- ❌ Will cause out of memory errors
- ❌ Everything runs slower (resource contention)
- ❌ Hard to monitor/debug
- ❌ If one fails, others keep running (waste time)

**On Mac MPS:**
- Limited GPU memory
- Can only handle one model at a time
- Running multiple = crashes

## Option 3: Manual Sequential (If You Want Control)

**Train one at a time, manually:**

**ResNet50:**
```bash
python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > resnet50_training.log 2>&1 &
disown
```

**Wait for it to finish, then ViT:**
```bash
python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model vit_base \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/vit_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --device auto > vit_training.log 2>&1 &
disown
```

**Then EfficientNet:**
```bash
python -u experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model efficientnet_b0 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/efficientnet_b0_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto > efficientnet_training.log 2>&1 &
disown
```

## Recommended: Use the Script

**Just run:**
```bash
bash experiments/eu_emotion_model_comparison/training/train_all_models.sh
```

**Or in background:**
```bash
nohup bash experiments/eu_emotion_model_comparison/training/train_all_models.sh > all_models_training.log 2>&1 &
disown
```

**Check progress:**
```bash
# Check overall progress
tail -f all_models_training.log

# Check individual model logs
tail -f resnet50_training.log
tail -f vit_training.log
tail -f efficientnet_training.log
```

## Time Estimate

**Sequential (one after another):**
- ResNet50: ~1-2 hours
- ViT: ~1.5-2.5 hours
- EfficientNet: ~1-2 hours
- **Total: ~3.5-6.5 hours**

**Can run overnight or while you're away!**
