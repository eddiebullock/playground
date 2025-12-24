# Fixing Silent Job Failures

## Problem

Both test jobs failed silently:
- **EU-Emotion**: Stopped at "Epoch 1/2" (training just started)
- **CAM**: Stopped after "Average Loss: 1.2760" during validation (40% complete)
- **No Python traceback** in output
- **Exit code 1** (failure)

## Most Likely Causes

1. **Process killed by system** (OOM, time limit, etc.)
2. **Unhandled exception** during validation/checkpoint saving
3. **DataLoader worker crash** (multiprocessing issue)

## Fixes Applied

### 1. Added Error Handling

Updated `finetune_clip_emotions.py`:
- Added try/except in validation loop
- Added error handling in main()
- Added traceback printing

### 2. Next Steps to Debug

Run this diagnostic on HPC:

```bash
cd ~/mr_ts_play

# Test if validation works in isolation
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from experiments.cam_human_like.training.finetune_clip_emotions import *
from experiments.cam_human_like.training.task_specific_dataset import TaskSpecificTrialDataset
from torch.utils.data import DataLoader
import json

# Load a small subset
with open('results/cam_test/cam_trial_definitions_test_all_files.json', 'r') as f:
    trials = json.load(f)['trials'][:5]  # Just 5 trials

dataset = TaskSpecificTrialDataset(
    trials=trials,
    data_root='/home/eb2007/data/CAM',
    num_frames=8
)

loader = DataLoader(dataset, batch_size=1, num_workers=0)

print("Testing data loading...")
for i, batch in enumerate(loader):
    print(f"Batch {i}: {len(batch[0])} frames")
    if i >= 2:
        break
print("Data loading works!")
EOF
```

## Immediate Fix: Increase Time Limit

The 2-hour limit might be too short. Update test scripts:

```bash
# In hpc_cam_test.slurm and hpc_eu_emotion_test.slurm
# Change from:
#SBATCH --time=02:00:00
# To:
#SBATCH --time=04:00:00  # 4 hours for safety
```

## Alternative: Run Interactively to See Errors

```bash
# Request interactive job
srun --time=2:00:00 --cpus-per-task=4 --mem=32G -p icelake --pty bash

# Then run the script directly
cd ~/mr_ts_play
source venv/bin/activate
module load python/3.8
bash experiments/cam_human_like/training/hpc_cam_test.sh
```

This will show errors in real-time.

## Check System Logs

```bash
# Check if process was killed
dmesg | tail -50 | grep -i "killed\|oom"

# Check job accounting
sacct -j 19801104 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
sacct -j 19801105 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

## Most Likely Solution

Based on the output:
1. **CAM test**: Completed epoch 1, failed during validation
2. **EU-Emotion test**: Failed early in training

**Recommendation**: 
1. Increase time limit to 4 hours
2. Add `num_workers=0` to DataLoader (already done, but verify)
3. Run interactively to see real-time errors
4. Check if validation is trying to process corrupted files

