# Quick Start Commands

## Step 1: Fine-Tune ResNet50 (Start Here)

**Optimized for speed (~1-2 hours):**

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**Expected time:** ~1-2 hours on Mac (MPS)

**What to watch for:**
- Validation accuracy should improve over epochs
- Best model saved automatically when validation accuracy improves
- Check `models/resnet50_emotion_finetuned/best_model.pth` when done

## Step 2: Fine-Tune ViT (After ResNet50)

**Once ResNet50 is done:**

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model vit_base \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/vit_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --device auto
```

**Expected time:** ~1.5-2.5 hours

## Step 3: Fine-Tune EfficientNet (Optional)

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model efficientnet_b0 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/efficientnet_b0_emotion_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**Expected time:** ~1-2 hours

## Step 4: Update Config and Evaluate

**After fine-tuning, update the config:**

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

**Then evaluate:**

```bash
python experiments/eu_emotion_model_comparison/scripts/run_comparison.py \
    --config experiments/eu_emotion_model_comparison/configs/comparison_config.yaml \
    --models resnet50 vit_base efficientnet_b0 \
    --device auto \
    --skip-failed
```

## Quick Test (5 epochs, ~30 minutes)

**If you want to test first before full training:**

```bash
python experiments/eu_emotion_model_comparison/training/finetune_vision_models.py \
    --model resnet50 \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions" \
    --output_dir models/resnet50_emotion_finetuned_test \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device auto
```

**Check validation accuracy:**
- If > 40% → Good, train full 10 epochs
- If < 30% → Check data/implementation

## Troubleshooting

**Out of memory?**
- Reduce batch size: `--batch_size 8`

**Too slow?**
- Reduce epochs: `--num_epochs 5` (test first)
- Reduce frames: `--num_frames 4`

**Want better accuracy?**
- Train longer: `--num_epochs 20`
- Adjust learning rate: try `5e-5` or `2e-4`
