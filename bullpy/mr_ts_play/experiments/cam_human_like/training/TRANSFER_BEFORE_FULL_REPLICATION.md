# Transfer Updated Files Before Full Replication

## Files That Were Updated

1. **`hpc_cam_replication.sh`** - Fixed undefined variable reference
2. **`finetune_clip_emotions.py`** - Added error handling
3. **`hpc_cam_test.slurm`** - Increased time limit to 4 hours
4. **`hpc_eu_emotion_test.slurm`** - Increased time limit to 4 hours

## Transfer Command

Run this on your **local machine**:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/transfer_to_hpc.sh
```

## Verify Transfer on HPC

After transferring, verify the files on HPC:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play

# Check if the fixed script is there (should NOT have CAM_TRIAL_DEFINITIONS reference)
grep -n "CAM_TRIAL_DEFINITIONS" experiments/cam_human_like/training/hpc_cam_replication.sh
# Should return nothing (or only in comments)

# Check if error handling is in finetune script
grep -n "try:" experiments/cam_human_like/training/finetune_clip_emotions.py | head -5
# Should show try/except blocks

# Check time limits
grep "time=" experiments/cam_human_like/training/hpc_cam_replication.slurm
grep "time=" experiments/cam_human_like/training/hpc_eu_emotion_replication.slurm
```

## Critical Fix

The `hpc_cam_replication.sh` script had a reference to undefined variable `CAM_TRIAL_DEFINITIONS` which would cause it to fail. This has been fixed - make sure the updated version is on HPC before running full replication.

## Quick Check

```bash
# On HPC, check if the script will work
cd ~/mr_ts_play
bash -n experiments/cam_human_like/training/hpc_cam_replication.sh
# Should return no errors (syntax check)
```



