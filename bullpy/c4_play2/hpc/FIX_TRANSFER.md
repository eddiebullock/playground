# Fix: Wrong Directory

## Problem
You're in `/Users/eb2007/playground/bullpy/mr_ts_play` but the files are in `c4_play2`.

## Solution: Change Directory First

```bash
# Navigate to the correct project directory
cd /Users/eb2007/playground/bullpy/c4_play2

# Verify you're in the right place
ls hpc/

# You should see:
# - hpc_config_comprehensive.yaml
# - comprehensive_ml_optimization.py
# - run_comprehensive_optimization.slurm

# Now run the rsync command
rsync -avz --progress \
  hpc/hpc_config_comprehensive.yaml \
  hpc/comprehensive_ml_optimization.py \
  hpc/run_comprehensive_optimization.slurm \
  eb2007@login.hpc.cam.ac.uk:/home/eb2007/c4/
```

## Quick One-Liner

```bash
cd /Users/eb2007/playground/bullpy/c4_play2 && rsync -avz --progress hpc/hpc_config_comprehensive.yaml hpc/comprehensive_ml_optimization.py hpc/run_comprehensive_optimization.slurm eb2007@login.hpc.cam.ac.uk:/home/eb2007/c4/
```
