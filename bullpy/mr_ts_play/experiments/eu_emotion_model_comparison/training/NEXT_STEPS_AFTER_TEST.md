# Next Steps: After Test Run Success

## ✅ Test Run Completed!

Your 1-epoch test worked! Now run the full training.

## Step 1: Full ResNet50 Training (10 epochs)

**Run in background:**

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

**Check progress:**
```bash
tail -f resnet50_training.log
```

**Expected time:** ~1-2 hours  
**Expected accuracy:** ~45-55% (much better than 8%!)

## Step 2: Fine-Tune Other Models

**After ResNet50 completes, fine-tune ViT:**

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

## Step 3: Update Config

**After all models are fine-tuned, update the config:**

Edit `experiments/eu_emotion_model_comparison/configs/comparison_config.yaml`:

```yaml
model_configs:
  resnet50:
    fine_tuned_path: "models/resnet50_emotion_finetuned/best_model.pth"
  
  vit_base:
    fine_tuned_path: "models/vit_emotion_finetuned/best_model.pth"
  
  efficientnet_b0:
    fine_tuned_path: "models/efficientnet_b0_emotion_finetuned/best_model.pth"
```

## Step 4: Evaluate on Test Set

**Run evaluation with all fine-tuned models:**

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --models resnet50 vit_base efficientnet_b0 \
    --device auto \
    --skip-failed
```

## What to Expect

**After 10 epochs:**
- ResNet50: ~45-55% accuracy
- ViT: ~50-60% accuracy  
- EfficientNet: ~45-55% accuracy

**Much better than the 8% from 1 epoch!**
