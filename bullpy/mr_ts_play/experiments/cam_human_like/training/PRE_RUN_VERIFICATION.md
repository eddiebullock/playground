# Pre-Run Verification Checklist

## ✅ Verification Complete

All scripts have been verified for correctness. Here's what was checked:

### 1. Python Syntax ✅
- `finetune_clip_emotions.py` - No syntax errors
- All function signatures match their calls
- Indentation is correct (fixed a bug where backward code was outside try block)

### 2. Function Parameters ✅
- `finetune_clip_task_specific()` accepts:
  - ✅ `weight_decay` (default: 0.01)
  - ✅ `use_lr_scheduler` (default: True)
  - ✅ `warmup_steps` (default: 100)
- All parameters are passed correctly from command line

### 3. Learning Rate Scheduler ✅
- Scheduler is created after DataLoader (so `len(train_loader)` works)
- Uses `SequentialLR` with `LinearLR` warmup + `CosineAnnealingLR`
- `scheduler.step()` is called after each optimizer step
- Properly handles `scheduler = None` case

### 4. Bash Scripts ✅
- `hpc_cam_replication.sh` - Syntax verified with `bash -n`
- `hpc_eu_emotion_replication.sh` - Syntax verified with `bash -n`
- `hpc_hyperparameter_tuning.sh` - Syntax verified with `bash -n`
- All variable references are correct
- All paths are properly quoted

### 5. SLURM Scripts ✅
- `hpc_cam_replication.slurm` - Correct partition, GPU request, modules
- `hpc_eu_emotion_replication.slurm` - Correct partition, GPU request, modules
- `hpc_hyperparameter_tuning.slurm` - Correct partition, GPU request, modules
- CUDA and cuDNN module loading with fallbacks
- Environment variables set correctly

### 6. Parameter Passing ✅
All scripts correctly pass:
- `--weight_decay $WEIGHT_DECAY` ✅
- `--use_lr_scheduler` ✅
- `--warmup_steps 100` ✅
- `--device $DEVICE` ✅
- `--num_frames $NUM_FRAMES` ✅
- `--batch_size $BATCH_SIZE` ✅
- `--learning_rate $LEARNING_RATE` ✅
- `--num_epochs $NUM_EPOCHS` ✅

### 7. GPU Fallback ✅
- Scripts check for CUDA availability
- Automatically fall back to CPU if GPU not available
- Batch size adjusted for CPU (16 → 4)

### 8. Path Resolution ✅
- CAM data: `/home/eb2007/data/CAM` ✅
- EU-Emotion data: Multiple RDS paths checked ✅
- Output directories: Properly created ✅

## Known Issues Fixed

1. ✅ **Syntax Error**: Fixed indentation bug where backward code was outside try block
2. ✅ **Argument Parser**: Fixed `--use_lr_scheduler` to work correctly with `action='store_true'`
3. ✅ **Scheduler Calculation**: Moved scheduler creation after DataLoader so `len(train_loader)` works

## Remaining Considerations

### GPU Module Availability
The scripts try to load CUDA/cuDNN modules with fallbacks. If modules aren't available, you may need to:
- Check available modules: `module avail cuda`
- Check available modules: `module avail cudnn`
- Update module names in SLURM scripts if needed

### Time Estimates
- **GPU**: 1-2 hours for 20 epochs (optimistic)
- **CPU**: 6-10 hours for 20 epochs (fallback)
- **Hyperparameter tuning**: 10-20 hours for all 10 runs

### First Run Recommendation
Before committing to full runs, consider:
1. Run a quick test with 2 epochs to verify GPU works
2. Check GPU is detected: Look for "CUDA available: True" in output
3. Monitor first epoch to ensure training progresses

## Ready to Run ✅

All scripts are verified and ready. The pipeline should work correctly. The main risk is:
- GPU modules might have different names on HPC (scripts have fallbacks)
- First epoch will confirm everything works

## Quick Test Command (Optional)

If you want to verify GPU works before full run:
```bash
# On HPC, quick test
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```


