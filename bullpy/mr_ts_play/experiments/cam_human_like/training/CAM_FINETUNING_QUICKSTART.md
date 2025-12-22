# CAM Fine-Tuning Quick Start

## Overview

Fine-tune CLIP on CAM train split to improve performance from 37% (zero-shot) to 65-75% (fine-tuned).

## Prerequisites

- CAM train/val splits exist: `data/splits/train.csv`, `data/splits/val.csv`
- CAM data root accessible
- PyTorch, transformers, and other dependencies installed

## Quick Start

### Option 1: Use the shell script (easiest)

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
./experiments/cam_human_like/training/run_cam_finetuning.sh
```

### Option 2: Run directly with Python

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
    --output_dir models/clip_cam_finetuned \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --device cpu  # or "mps" for Mac M1/M2, "cuda" for GPU
```

## Expected Results

- **Training time**: ~2-4 hours (CPU), ~30-60 min (GPU/MPS)
- **Validation accuracy**: Should improve from ~37% to 60-75%
- **Model saved to**: `models/clip_cam_finetuned/best_model/`

## After Fine-Tuning

1. **Update config** to use fine-tuned model:
   ```yaml
   # configs/cam_config.yaml
   model:
     name: "models/clip_cam_finetuned/best_model"
   ```

2. **Run experiment** with fine-tuned model:
   ```bash
   python experiments/cam_human_like/run_experiment.py \
       --config configs/cam_config.yaml \
       --split all --no-actor-filtering
   ```

3. **Expected performance**: 65-75% face accuracy (vs 37% zero-shot)

## Monitoring Training

The script will print:
- Training loss per epoch
- Validation accuracy per epoch
- Best model saved when validation accuracy improves

Checkpoints are saved every 2 epochs to `models/clip_cam_finetuned/epoch_N/`

## Troubleshooting

### Out of memory
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--num_frames` (try 4 instead of 8)

### Slow training
- Use GPU/MPS if available: `--device mps` or `--device cuda`
- Reduce `--num_epochs` for quick test (try 3-5)

### Dataset errors
- Verify train.csv and val.csv exist
- Check data_root path is correct
- Ensure video files are accessible

## Next Steps

After CAM fine-tuning works:
1. Try FER2013 fine-tuning (external dataset - more rigorous)
2. Compare results: CAM vs FER2013 fine-tuning
3. Report both in thesis (task-specific vs general emotion recognition)





