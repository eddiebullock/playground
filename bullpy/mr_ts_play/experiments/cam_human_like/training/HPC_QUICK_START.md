# HPC Quick Start Guide: Separate Experiments

## Overview

This guide helps you run **quick tests** to verify the pipeline works before submitting full replication jobs.

## Methodology: Separate Experiments

- **Experiment 1**: EU-Emotion mental state recognition (separate)
- **Experiment 2**: CAM mental state recognition (separate)
- **Unified methodology**: Same procedure for both (discover all files, generate trials)

## Quick Test Workflow

### Step 1: Transfer Updated Scripts

On your **local machine**:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/transfer_to_hpc.sh
```

This transfers:
- Updated replication scripts (using all files)
- CAM trial generator (`create_cam_trials_from_all_files.py`)
- Test scripts for quick validation

### Step 2: Run Quick Tests

On **HPC**, run quick tests to verify the pipeline:

#### Test 1: EU-Emotion Quick Test

```bash
ssh eb2007@login-cpu.hpc.cam.ac.uk
cd ~/mr_ts_play
sbatch experiments/cam_human_like/training/hpc_eu_emotion_test.slurm
```

**What it does**:
- 2 epochs (quick validation)
- 5 trials per emotion (reduced from 10)
- ~30-60 minutes runtime
- Verifies EU-Emotion pipeline works

**Check results**:
```bash
tail -f eu_emotion_test_*.out
```

#### Test 2: CAM Quick Test

```bash
sbatch experiments/cam_human_like/training/hpc_cam_test.slurm
```

**What it does**:
- 2 epochs (quick validation)
- 5 trials per concept (reduced from 10)
- ~1-2 hours runtime
- Verifies CAM pipeline works

**Check results**:
```bash
tail -f cam_test_*.out
```

### Step 3: Run Full Replications

Once tests pass, run full replications:

#### Full EU-Emotion Replication

```bash
sbatch experiments/cam_human_like/training/hpc_eu_emotion_replication.slurm
```

**Configuration**:
- 10 epochs
- 10 trials per emotion
- ~6-10 hours runtime
- Expected: 60-75% accuracy

#### Full CAM Replication

```bash
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm
```

**Configuration**:
- 10 epochs
- 10 trials per concept
- ~6-10 hours runtime
- Expected: 70-85% accuracy
- Compare to: Human baselines (70-88%)

## What Changed

### Updated Methodology

**Before**: Two-stage training (EU-Emotion → CAM)
- Speculative, label mismatch issues

**Now**: Separate experiments
- Clear, robust, high impact
- Each experiment answers specific question
- No label mismatch

### Updated Scripts

1. **CAM Replication** (`hpc_cam_replication.sh`)
   - Now uses `create_cam_trials_from_all_files.py`
   - Discovers ALL valid files (not just 100)
   - Generates trials from all files
   - 10 trials per concept

2. **Test Scripts** (NEW)
   - `hpc_cam_test.sh` / `hpc_cam_test.slurm`
   - `hpc_eu_emotion_test.sh` / `hpc_eu_emotion_test.slurm`
   - Quick validation (2 epochs, reduced trials)

## Expected Results

### EU-Emotion Test
- **Runtime**: 30-60 minutes
- **Accuracy**: Should be above random (25%)
- **Purpose**: Verify pipeline works

### CAM Test
- **Runtime**: 1-2 hours
- **Accuracy**: Should be above zero-shot (37%)
- **Purpose**: Verify pipeline works

### Full Replications
- **EU-Emotion**: 60-75% accuracy
- **CAM**: 70-85% accuracy (approaching human performance)

## Troubleshooting

### If tests fail:

1. **Check Python environment**:
   ```bash
   source ~/mr_ts_play/venv/bin/activate
   python --version
   ```

2. **Check data paths**:
   ```bash
   ls -la /home/eb2007/data/CAM
   ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions
   ```

3. **Check job output**:
   ```bash
   tail -f <job_id>.out
   tail -f <job_id>.err
   ```

4. **Check job status**:
   ```bash
   squeue -u eb2007
   ```

## Next Steps

1. ✅ Transfer updated scripts
2. ✅ Run quick tests
3. ✅ Verify tests pass
4. ✅ Run full replications
5. ✅ Analyze results
6. ✅ Compare to human baselines

## Summary

- **Methodology**: Separate experiments (robust, high impact)
- **Quick tests**: 2 epochs, reduced trials (verify pipeline)
- **Full replications**: 10 epochs, all files (final results)
- **Expected**: 70-85% CAM accuracy (approaching human performance)

