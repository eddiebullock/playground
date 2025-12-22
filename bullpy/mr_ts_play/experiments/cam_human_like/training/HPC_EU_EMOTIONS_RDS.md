# Transferring EU Emotions to RDS Storage

## Why Transfer to RDS?

- **Avoids /home quota**: EU emotions library is large (likely 10-50GB+)
- **Direct transfer**: Bypasses /home entirely
- **Resumable**: Can pause and resume transfers
- **More space**: 1100GB available on RDS vs 50GB on /home

## Step 1: Check What's Already Transferred

On HPC, check if anything was already transferred to /home:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
du -sh ~/data/EU_emotions 2>/dev/null || echo "Nothing in /home/data/EU_emotions"
```

## Step 2: Transfer Directly to RDS

From your local machine, run:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh
```

This will:
- Transfer directly to `/rds-d7/project/45718/users/eb2007/data/EU_emotions`
- Show progress
- Support resume if interrupted (use `--partial` flag)

## Step 3: Resume Interrupted Transfer

If the transfer stops, you can resume it:

```bash
# The script uses --partial flag, so you can just re-run it
bash experiments/cam_human_like/training/transfer_eu_emotions_to_rds.sh
```

Or manually:

```bash
rsync -avh --progress --partial \
    --exclude '.DS_Store' \
    --exclude '*.tmp' \
    "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/" \
    eb2007@login-cpu.hpc.cam.ac.uk:/rds-d7/project/45718/users/eb2007/data/EU_emotions/
```

## Step 4: Verify Transfer

On HPC:

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk

# Check size
du -sh /rds-d7/project/45718/users/eb2007/data/EU_emotions

# Check structure
ls -lh /rds-d7/project/45718/users/eb2007/data/EU_emotions | head -20

# Count files
find /rds-d7/project/45718/users/eb2007/data/EU_emotions -type f | wc -l
```

## Step 5: Update Scripts to Use RDS Path

The EU emotions data will be at:
```
/rds-d7/project/45718/users/eb2007/data/EU_emotions
```

Update your training scripts to use this path instead of `/home/eb2007/data/EU_emotions`.

## Benefits

1. **No quota issues**: RDS has 1100GB available
2. **Faster**: Direct transfer, no intermediate /home step
3. **Resumable**: Can pause/resume if interrupted
4. **Persistent**: Data survives across sessions
5. **Shared**: Can be accessed by your research group

## Alternative: Move Existing Transfer

If you already started transferring to /home, you can move it:

```bash
# On HPC
ssh eb2007@login-cpu.hpc.cam.ac.uk

# Create RDS directory
mkdir -p /rds-d7/project/45718/users/eb2007/data

# Move existing data (if any)
if [ -d ~/data/EU_emotions ]; then
    mv ~/data/EU_emotions /rds-d7/project/45718/users/eb2007/data/
    echo "Moved existing data to RDS"
fi
```

Then continue transferring the rest directly to RDS.

