# How to Debug Failed SLURM Jobs

## Your Failed Jobs

- **Job 19801104**: `eu_emotion_test` - FAILED
- **Job 19801105**: `cam_test` - FAILED

## Step 1: Check Output Files

The SLURM scripts specify output files:
- `--output=eu_emotion_test_%j.out` → `eu_emotion_test_19801104.out`
- `--error=eu_emotion_test_%j.err` → `eu_emotion_test_19801104.err`
- `--output=cam_test_%j.out` → `cam_test_19801105.out`
- `--error=cam_test_%j.err` → `cam_test_19801105.err`

### Commands to Run on HPC:

```bash
# Navigate to project directory
cd ~/mr_ts_play

# Check EU-Emotion test output (stdout)
cat eu_emotion_test_19801104.out

# Check EU-Emotion test errors (stderr)
cat eu_emotion_test_19801104.err

# Check CAM test output (stdout)
cat cam_test_19801105.out

# Check CAM test errors (stderr)
cat cam_test_19801105.err

# Or view last 50 lines (most recent errors)
tail -50 eu_emotion_test_19801104.out
tail -50 cam_test_19801105.out
```

## Step 2: Check Job Details

```bash
# Get detailed information about the jobs
scontrol show job 19801104
scontrol show job 19801105

# Check if jobs are still in queue (should be empty if failed)
squeue -j 19801104,19801105
```

## Step 3: Common Issues to Check

### 1. Missing Files/Scripts
```bash
# Check if scripts exist
ls -la experiments/cam_human_like/training/hpc_eu_emotion_test.sh
ls -la experiments/cam_human_like/training/hpc_cam_test.sh
ls -la experiments/cam_human_like/training/create_cam_trials_from_all_files.py
```

### 2. Python Environment
```bash
# Check if venv exists
ls -la ~/mr_ts_play/venv/bin/activate
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate

# Test Python import
python3 -c "import torch; print(torch.__version__)"
```

### 3. Data Paths
```bash
# Check CAM data
ls -la /home/eb2007/data/CAM | head -10

# Check EU-Emotion data
ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions | head -10
```

### 4. Module Loading
```bash
# Test module loading
module purge
module load python/3.8
python --version
```

## Step 4: Most Likely Issues

Based on the scripts, common failures:

1. **Missing Script**: `create_cam_trials_from_all_files.py` not transferred
   - **Fix**: Run `transfer_to_hpc.sh` again

2. **Missing Virtual Environment**: venv not found
   - **Fix**: Run `setup_rds_venv.sh`

3. **Missing Data**: CAM or EU-Emotion data not found
   - **Fix**: Check data paths in error output

4. **Python Module**: Cannot load Python module
   - **Fix**: Check available modules with `module avail python`

5. **Import Error**: Missing Python packages
   - **Fix**: Install packages in venv

## Quick Diagnostic Script

Run this on HPC to check everything:

```bash
cd ~/mr_ts_play

echo "=== Checking Scripts ==="
ls -la experiments/cam_human_like/training/hpc_*_test.sh
ls -la experiments/cam_human_like/training/create_cam_trials_from_all_files.py

echo ""
echo "=== Checking Virtual Environment ==="
if [ -f "${HOME}/mr_ts_play/venv/bin/activate" ]; then
    echo "✅ Found: ~/mr_ts_play/venv"
elif [ -f "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/users/eb2007/venv/bin/activate" ]; then
    echo "✅ Found: RDS venv"
else
    echo "❌ No venv found!"
fi

echo ""
echo "=== Checking Data ==="
if [ -d "/home/eb2007/data/CAM" ]; then
    echo "✅ CAM data found"
    ls -la /home/eb2007/data/CAM | head -3
else
    echo "❌ CAM data not found"
fi

if [ -d "${HOME}/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions" ]; then
    echo "✅ EU-Emotion data found"
    ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions | head -3
else
    echo "❌ EU-Emotion data not found"
fi

echo ""
echo "=== Checking Python ==="
module purge
if module load python/3.8 2>/dev/null; then
    echo "✅ Python module loaded"
    python --version
else
    echo "❌ Cannot load Python module"
fi
```

## Next Steps

1. Run the diagnostic commands above
2. Check the `.out` and `.err` files
3. Share the error messages if you need help fixing them



