# Check Partition Access and Fix GPU Job Submission

## The Error
```
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

This means your account doesn't have access to the `ukaea-amp` partition, or you need to specify an account.

## Step 1: Check Partition Access (Run on HPC)

```bash
# Check which partitions you can access
sinfo -p ukaea-amp
sinfo -p ampere
sinfo -p icelake

# Check your account and partition associations
sacctmgr show user $USER withassoc format=account,partition 2>&1

# Check what account your previous jobs used
sacct -u $USER --starttime=today --format=JobID,Account,Partition,State | head -10
```

## Step 2: Try Different Partitions

### Option A: Try `ampere` partition (most common GPU partition)

The scripts have been updated to use `ampere` instead of `ukaea-amp`. Try submitting again:

```bash
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm
```

### Option B: If `ampere` also fails, check if you need an account

```bash
# Check what account you should use
sacctmgr show user $USER withassoc format=account,partition 2>&1 | grep -E "ampere|ukaea"

# If you see an account like "baron-coh+" or similar, add it to SLURM script:
# Add this line after #SBATCH -p ampere:
# #SBATCH --account=baron-coh+
```

### Option C: Fall back to CPU (if no GPU access)

If you don't have GPU access, use the CPU partition:

```bash
# Edit the SLURM script
nano experiments/cam_human_like/training/hpc_cam_replication.slurm

# Change:
# #SBATCH --gres=gpu:1
# #SBATCH -p ampere
# To:
# #SBATCH -p icelake
# (Remove the --gres=gpu:1 line)

# The .sh script will automatically detect CPU and adjust batch size
```

## Step 3: Quick Test Commands

```bash
# Test 1: Check if ampere partition works
sbatch --partition=ampere --gres=gpu:1 --time=00:10:00 --wrap="echo 'GPU test'; nvidia-smi"

# Test 2: Check if you need account
# (If test 1 fails, try with account from sacctmgr output)
sbatch --account=YOUR_ACCOUNT --partition=ampere --gres=gpu:1 --time=00:10:00 --wrap="echo 'GPU test'; nvidia-smi"

# Test 3: Check CPU partition (should always work)
sbatch --partition=icelake --time=00:10:00 --wrap="echo 'CPU test'"
```

## Step 4: Update Scripts Based on Results

### If `ampere` works:
✅ Scripts are already updated to use `ampere` - you're good to go!

### If you need an account:
Edit the SLURM scripts and add:
```bash
#SBATCH --account=YOUR_ACCOUNT_NAME
```

### If no GPU access:
1. Remove `--gres=gpu:1` from SLURM scripts
2. Change partition to `icelake`
3. Scripts will automatically use CPU (batch size will be reduced)

## Quick Fix: CPU Fallback Script

If GPU doesn't work, I can create a CPU-only version. For now, you can manually edit:

```bash
# On HPC, quick edit to use CPU
cd ~/mr_ts_play
sed -i 's/#SBATCH --gres=gpu:1/#SBATCH --gres=gpu:1  # Commented: no GPU access/' experiments/cam_human_like/training/hpc_cam_replication.slurm
sed -i 's/#SBATCH -p ampere/#SBATCH -p icelake/' experiments/cam_human_like/training/hpc_cam_replication.slurm
```

The training script will automatically detect CPU and adjust batch size.


