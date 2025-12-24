# Transfer Updated Files to HPC

## Files Updated

1. **`finetune_clip_emotions.py`** - Added error handling and traceback printing
2. **`hpc_cam_test.slurm`** - Increased time limit to 4 hours
3. **`hpc_eu_emotion_test.slurm`** - Increased time limit to 4 hours

## Transfer Command

Run this on your **local machine**:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/transfer_to_hpc.sh
```

This will transfer:
- Updated `finetune_clip_emotions.py` (with error handling)
- Updated test SLURM scripts (with 4-hour time limit)
- All other necessary files

## After Transfer

Once transferred, on HPC you can:

1. **Run tests again**:
   ```bash
   sbatch experiments/cam_human_like/training/hpc_cam_test.slurm
   sbatch experiments/cam_human_like/training/hpc_eu_emotion_test.slurm
   ```

2. **Or run interactively** to see errors in real-time:
   ```bash
   srun --time=2:00:00 --cpus-per-task=4 --mem=32G -p icelake --pty bash
   cd ~/mr_ts_play
   source venv/bin/activate
   module load python/3.8
   bash experiments/cam_human_like/training/hpc_cam_test.sh
   ```

## What Changed

### Error Handling
- Added try/except blocks in validation loop
- Added traceback printing for exceptions
- Errors will now be visible in output files

### Time Limits
- Increased from 2 hours to 4 hours
- Should prevent time limit failures during validation

