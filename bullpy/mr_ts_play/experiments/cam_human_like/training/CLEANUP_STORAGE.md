# HPC Storage Cleanup Guide

## Current Status

From your quota output:
- **/home**: 55.0 GB used / 50.0 GB quota (OVER QUOTA - grace period until 2025-12-29)
- **rds-ePtR33Nsgi4**: 12465.8 GB used / 23000.0 GB (plenty of space)

## Cleanup Commands

Run these on HPC to free up space:

### 1. Check Disk Usage by Directory

```bash
cd ~/mr_ts_play

# See what's taking up space
du -sh * | sort -h
du -sh results/* 2>/dev/null | sort -h
du -sh models/* 2>/dev/null | sort -h
```

### 2. Remove Python Cache

```bash
# Remove __pycache__ directories
find ~/mr_ts_play -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

# Remove .pyc files
find ~/mr_ts_play -name "*.pyc" -delete

# Remove .pyo files
find ~/mr_ts_play -name "*.pyo" -delete
```

### 3. Remove Pip Cache

```bash
# Check pip cache size
du -sh ~/.cache/pip 2>/dev/null

# Remove pip cache (saves several GB)
rm -rf ~/.cache/pip
```

### 4. Remove Old Model Checkpoints

```bash
cd ~/mr_ts_play

# List all checkpoint directories
find results -type d -name "epoch_*" 2>/dev/null
find results -type d -name "model_checkpoints" 2>/dev/null

# Remove old epoch checkpoints (keep only best_model)
find results -type d -name "epoch_*" -exec rm -r {} + 2>/dev/null

# Or remove specific old checkpoints
# rm -rf results/cam_test/model_checkpoints/epoch_*
# rm -rf results/eu_emotion_test/model_checkpoints/epoch_*
```

### 5. Remove Failed Job Output Files

```bash
# Remove old .out and .err files (keep recent ones)
# Keep last 10 jobs
ls -t *.out | tail -n +11 | xargs rm -f
ls -t *.err | tail -n +11 | xargs rm -f

# Or remove specific old files
rm -f eu_emotion_test_*.out eu_emotion_test_*.err
rm -f cam_test_*.out cam_test_*.err
```

### 6. Remove Temporary Files

```bash
# Remove .DS_Store files (Mac)
find ~/mr_ts_play -name ".DS_Store" -delete

# Remove .swp files (vim)
find ~/mr_ts_play -name "*.swp" -delete

# Remove .log files
find ~/mr_ts_play -name "*.log" -delete
```

### 7. Clean Up Large Files

```bash
# Find large files (>100MB)
find ~/mr_ts_play -type f -size +100M -exec ls -lh {} \; | awk '{print $5, $9}'

# Find largest files
find ~/mr_ts_play -type f -exec ls -lh {} \; | awk '{print $5, $9}' | sort -h | tail -20
```

### 8. Clean Up Results Directory

```bash
cd ~/mr_ts_play/results

# See what's in results
du -sh * | sort -h

# Remove old test results (keep only latest)
# Be careful - only remove what you don't need
rm -rf cam_test/model_checkpoints/epoch_*
rm -rf eu_emotion_test/model_checkpoints/epoch_*
```

## Quick Cleanup Script

Run this to clean up common items:

```bash
cd ~/mr_ts_play

echo "Cleaning up storage..."

# Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Pip cache
echo "Removing pip cache..."
rm -rf ~/.cache/pip

# Old epoch checkpoints (keep best_model)
echo "Removing old epoch checkpoints..."
find results -type d -name "epoch_*" -exec rm -r {} + 2>/dev/null

# Temporary files
echo "Removing temporary files..."
find . -name ".DS_Store" -delete
find . -name "*.swp" -delete
find . -name "*.log" -delete

echo "Cleanup complete!"
echo ""
echo "Checking disk usage..."
du -sh ~/mr_ts_play
quota -s
```

## Check What's Taking Space

```bash
# Check total usage
du -sh ~/mr_ts_play

# Check by subdirectory
du -sh ~/mr_ts_play/* | sort -h

# Check results directory
du -sh ~/mr_ts_play/results/* 2>/dev/null | sort -h

# Check models directory
du -sh ~/mr_ts_play/models/* 2>/dev/null | sort -h
```

## Safe Cleanup (Recommended)

Start with safe items first:

```bash
# 1. Python cache (safe)
find ~/mr_ts_play -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find ~/mr_ts_play -name "*.pyc" -delete

# 2. Pip cache (safe, can re-download)
rm -rf ~/.cache/pip

# 3. Check what's left
du -sh ~/mr_ts_play
quota -s

# 4. Then remove old checkpoints if still needed
```

## After Cleanup

Check your quota again:

```bash
quota -s
du -sh ~/mr_ts_play
```

You should be under the 50 GB limit for /home.

