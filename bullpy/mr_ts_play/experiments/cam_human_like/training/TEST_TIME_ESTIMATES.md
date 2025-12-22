# Local Test Time Estimates

## Quick Test (1-2 epochs)

### Settings for Quick Test
- **Epochs**: 2
- **Batch size**: 8 (reduced from 16)
- **Frames**: 4 (reduced from 8)
- **Purpose**: Verify script works, check for errors

### Time Estimates

| Device | Time per Epoch | Total (2 epochs) |
|--------|---------------|-------------------|
| **CPU** | 2-4 hours | **4-8 hours** |
| **MPS (Mac M1/M2)** | 10-20 min | **20-40 minutes** |
| **CUDA (GPU)** | 5-10 min | **10-20 minutes** |

### Full Training (10 epochs)

| Device | Time per Epoch | Total (10 epochs) |
|--------|---------------|-------------------|
| **CPU** | 2-4 hours | **20-40 hours** |
| **MPS (Mac M1/M2)** | 10-20 min | **100-200 minutes (1.5-3 hours)** |
| **CUDA (GPU)** | 5-10 min | **50-100 minutes (1-2 hours)** |

## Recommendation

### For Quick Test:
1. **If you have MPS (Mac M1/M2)**: Run 2 epochs locally (~20-40 min)
2. **If you have CPU only**: Consider running just 1 epoch (~2-4 hours) or wait for HPC
3. **If you have CUDA**: Run 2 epochs locally (~10-20 min)

### After Test Works:
- **Run full training on HPC** (50-100 minutes on GPU)
- Or run locally if you have MPS (1.5-3 hours)

## Quick Test Command

```bash
# Quick test with reduced settings
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data data/splits/train.csv \
    --val_data data/splits/val.csv \
    --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
    --output_dir models/clip_cam_finetuned_test \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --device cpu  # or "mps" or "cuda"
    --num_frames 4
```

Or use the test script:
```bash
./experiments/cam_human_like/training/test_finetuning.sh
```

## What to Check During Test

1. **No errors**: Script loads data, model, and starts training
2. **Training loss decreases**: Should see loss going down
3. **Validation accuracy**: Should see some improvement (even if small)
4. **Model saves**: Check that `models/clip_cam_finetuned_test/best_model/` is created

If all works, you're ready for HPC!





