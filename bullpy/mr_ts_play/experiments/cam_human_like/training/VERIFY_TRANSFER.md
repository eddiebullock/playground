# How to Verify CAM Dataset Transfer to HPC

## Quick Check Commands

### On HPC (SSH session):

```bash
# Count total files
find ~/data/CAM -type f | wc -l

# Check total size
du -sh ~/data/CAM

# Count files by type
find ~/data/CAM -name "*.mov" | wc -l
find ~/data/CAM -name "*.aif" | wc -l

# Check a specific directory
ls -lh ~/data/CAM/01/0100104/ | head -10
```

### From Local Machine:

```bash
# Use the verification script
cd /Users/eb2007/playground/bullpy/mr_ts_play
./verify_transfer.sh
```

Or manually:

```bash
# Count local files
find "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" -type f | wc -l

# Count HPC files (via SSH)
ssh eb2007@login-cpu.hpc.cam.ac.uk "find /home/eb2007/data/CAM -type f | wc -l"

# Compare sizes
du -sh "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
ssh eb2007@login-cpu.hpc.cam.ac.uk "du -sh /home/eb2007/data/CAM"
```

## Expected Results

### File Counts
- **Total files**: ~6006 (based on your rsync output)
- **Video files (.mov)**: ~4944
- **Audio files (.aif)**: ~1000+ (definitions folder)

### File Sizes
- **Total size**: ~500 MB - 1 GB (depending on video compression)
- Local and HPC sizes should match (within a few MB)

## What to Look For

### ✅ Success Indicators:
- File counts match (or very close - within 1-2 files)
- Sizes match (or very close - within a few MB)
- Sample files exist and have correct sizes

### ⚠️ Warning Signs:
- File count mismatch > 10 files
- Size mismatch > 50 MB
- Missing directories

## Detailed Verification

### Check Specific Files:

```bash
# On HPC, verify a few specific files exist
ssh eb2007@login-cpu.hpc.cam.ac.uk "ls -lh ~/data/CAM/01/0100104/0100104M1Vhumiliating.mov"
ssh eb2007@login-cpu.hpc.cam.ac.uk "ls -lh ~/data/CAM/definitions/ | head -10"
```

### Compare Directory Structure:

```bash
# On HPC
ssh eb2007@login-cpu.hpc.cam.ac.uk "find ~/data/CAM -type d | sort"

# Locally
find "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" -type d | sort
```

## If Files Are Missing

If verification shows missing files:

1. **Retry transfer** (rsync will only transfer missing files):
   ```bash
   rsync -avz --progress --partial \
       "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions/" \
       eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/CAM/
   ```

2. **Check for specific missing files**:
   ```bash
   # Find files that exist locally but not on HPC
   # (This requires more complex scripting - rsync --dry-run can help)
   ```

## Quick Verification Script

Run this from your local machine:

```bash
./verify_transfer.sh
```

This will:
- Count files locally
- Count files on HPC
- Compare sizes
- Report any mismatches









