# Quick GPU Test - 2 Epochs

## Step 1: Transfer Scripts (Local Machine)

```bash
bash experiments/cam_human_like/training/transfer_to_hpc.sh
```

## Step 2: Modify for Quick Test (On HPC)

**Option A: Edit the script temporarily (recommended)**

```bash
# SSH to HPC first
ssh eb2007@login-cpu.hpc.cam.ac.uk

# Navigate to project
cd ~/mr_ts_play

# Edit CAM replication script
nano experiments/cam_human_like/training/hpc_cam_replication.sh

# Change line 45 from:
# NUM_EPOCHS=20  # Optimized for GPU: 20 epochs for better convergence
# To:
# NUM_EPOCHS=2  # QUICK TEST: 2 epochs to verify GPU works

# Save and exit (Ctrl+X, then Y, then Enter)
```

**Option B: Use sed to change it automatically**

```bash
# On HPC
cd ~/mr_ts_play
sed -i 's/NUM_EPOCHS=20/NUM_EPOCHS=2  # QUICK TEST/' experiments/cam_human_like/training/hpc_cam_replication.sh
```

## Step 3: Submit Quick Test Job

```bash
# Submit the test job
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm

# Note the job ID (e.g., "Submitted batch job 19812345")
```

## Step 4: Monitor and Verify GPU

```bash
# Check job status
squeue -u $USER

# Watch output file (replace JOB_ID with your job ID)
tail -f cam_replication_JOB_ID.out

# Or check the most recent output
tail -f $(ls -t cam_replication_*.out | head -1)
```

**Look for these GPU verification messages:**
- `CUDA available: True`
- `GPU device: NVIDIA A100` (or similar)
- `Loaded cuda/11.8` (or similar version)
- `Using LR scheduler: warmup=100 steps...`

**Check for errors:**
- If you see `CUDA available: False` → GPU not detected, check module loading
- If you see `Warning: Could not load CUDA module` → Module names might be different

## Step 5: Check Results After Job Completes

```bash
# Check if job completed successfully
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed

# Check final output
tail -50 $(ls -t cam_replication_*.out | head -1)

# Verify GPU was used (look for speed indicators)
grep -E "Epoch|Training|CUDA|GPU" $(ls -t cam_replication_*.out | head -1) | head -20
```

## Step 6: If Test Passes, Run Full Job

**First, restore the original epoch count:**

```bash
# Restore to 20 epochs
sed -i 's/NUM_EPOCHS=2  # QUICK TEST/NUM_EPOCHS=20  # Optimized for GPU: 20 epochs for better convergence/' experiments/cam_human_like/training/hpc_cam_replication.sh

# Or manually edit:
nano experiments/cam_human_like/training/hpc_cam_replication.sh
# Change NUM_EPOCHS back to 20
```

**Then submit full job:**

```bash
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm
```

## Quick Verification Commands

```bash
# Check GPU detection (run this in an interactive session or check job output)
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Check available CUDA modules
module avail cuda 2>&1 | head -10

# Check available cuDNN modules  
module avail cudnn 2>&1 | head -10
```



