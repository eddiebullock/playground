# Running CAM Fine-Tuning Test

## Quick Test Command

Run this to test CAM fine-tuning locally (2 epochs, reduced settings):

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
    --output_dir models/clip_cam_finetuned_test \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device cpu \
    --num_frames 4
```

**Note**: Change `--device cpu` to `--device mps` if you have Mac M1/M2, or `--device cuda` if you have a GPU.

## What to Expect

### Output
- Loading datasets...
- Train samples: ~2000-3000 (face trials only)
- Val samples: ~400-600 (face trials only)
- Loading CLIP model...
- Epoch 1/2: Training loss, validation accuracy
- Epoch 2/2: Training loss, validation accuracy
- Model saved to `models/clip_cam_finetuned_test/best_model/`

### Time Estimates
- **CPU**: ~4-8 hours (2 epochs)
- **MPS (Mac M1/M2)**: ~20-40 minutes
- **CUDA (GPU)**: ~10-20 minutes

### Success Indicators
✅ No errors during dataset loading
✅ Training loss decreases over epochs
✅ Validation accuracy improves (even if small)
✅ Model saved to output directory

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`:
```bash
# Activate your conda/virtual environment first
conda activate your_env
# or
source venv/bin/activate
```

### Dataset Errors
If dataset loading fails:
- Check that `data/splits/train.csv` and `val.csv` exist
- Verify data_root path is correct
- Ensure video files are accessible

### Out of Memory
If you get OOM errors:
- Reduce `--batch_size` to 4 or 2
- Reduce `--num_frames` to 2

### Slow Training
- Use `--device mps` if you have Mac M1/M2
- Use `--device cuda` if you have GPU
- Reduce to 1 epoch for quick test: `--num_epochs 1`

## After Test Works

Once the test completes successfully:
1. ✅ Script works correctly
2. ✅ Ready for full training (10 epochs)
3. ✅ Can submit to HPC for faster training

## Next Steps

After successful test:
1. Run full training on HPC (10 epochs, ~50-100 min on GPU)
2. Or run locally if you have MPS (1.5-3 hours)
3. Evaluate fine-tuned model on CAM test set





