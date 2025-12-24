# Changes Summary: Updated to Separate Experiments Methodology

## What Changed

### Methodology Update

**Before**: Two-stage training (EU-Emotion → CAM)
- Speculative approach
- Label mismatch issues
- Unclear benefit

**Now**: Separate experiments
- Clear, robust, high impact
- Each experiment answers specific question
- No label mismatch
- Unified methodology (same procedure for both)

### Script Updates

#### 1. CAM Replication (`hpc_cam_replication.sh`)

**Changed**:
- Now uses `create_cam_trials_from_all_files.py` (NEW)
- Discovers ALL valid files (not just 100 pre-defined trials)
- Generates trials from all available files
- 10 trials per concept (configurable)

**Before**:
```bash
# Used pre-defined trial definitions
create_cam_splits.py --trial-definitions data/cam_trial_definitions_20concepts.json
```

**Now**:
```bash
# Generates trials from all files
create_cam_trials_from_all_files.py --cam-dir /home/eb2007/data/CAM --trials-per-concept 10
```

#### 2. New Test Scripts

**Created**:
- `hpc_cam_test.sh` / `hpc_cam_test.slurm` - Quick CAM test (2 epochs, 5 trials/concept)
- `hpc_eu_emotion_test.sh` / `hpc_eu_emotion_test.slurm` - Already existed, updated

**Purpose**: Quick validation before full replication runs

#### 3. New Scripts

**Created**:
- `create_cam_trials_from_all_files.py` - Generates CAM trials from all valid files
- `FINAL_METHODOLOGY.md` - Updated methodology documentation
- `TWO_STAGE_TRAINING_CLARIFICATION.md` - Explains why separate experiments
- `HPC_QUICK_START.md` - Quick start guide

#### 4. Updated Transfer Script

**Updated**: `transfer_to_hpc.sh`
- Now transfers all new scripts
- Includes CAM trial generator
- Includes test scripts

## How to Use

### Step 1: Transfer Updated Scripts

On your **local machine**:

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play
bash experiments/cam_human_like/training/transfer_to_hpc.sh
```

### Step 2: Run Quick Tests

On **HPC**:

```bash
# Test EU-Emotion pipeline
sbatch experiments/cam_human_like/training/hpc_eu_emotion_test.slurm

# Test CAM pipeline
sbatch experiments/cam_human_like/training/hpc_cam_test.slurm
```

**Quick tests**:
- 2 epochs
- Reduced trials (5 per emotion/concept)
- 30-60 minutes (EU-Emotion) or 1-2 hours (CAM)
- Verify pipeline works

### Step 3: Run Full Replications

Once tests pass:

```bash
# Full EU-Emotion replication
sbatch experiments/cam_human_like/training/hpc_eu_emotion_replication.slurm

# Full CAM replication
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm
```

**Full replications**:
- 10 epochs
- Full trials (10 per emotion/concept)
- 6-10 hours runtime
- Final results

## Expected Results

### Quick Tests
- **Purpose**: Verify pipeline works
- **Accuracy**: Should be above random/baseline
- **Runtime**: 30-60 min (EU-Emotion), 1-2 hours (CAM)

### Full Replications
- **EU-Emotion**: 60-75% accuracy
- **CAM**: 70-85% accuracy (approaching human performance)
- **Comparison**: CAM compares to human baselines (70-88%)

## Key Differences

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| **CAM Data** | 100 pre-defined trials | ALL valid files (~200-300 trials) |
| **Methodology** | Two-stage training | Separate experiments |
| **EU-Emotion** | Same (all files) | Same (all files) |
| **Training** | EU-Emotion → CAM | Separate (no transfer) |
| **Clarity** | Speculative | Clear, robust |

## Benefits

1. **More Data**: Uses ALL available files (better learning)
2. **Clearer**: Separate experiments (easier to interpret)
3. **Robust**: No label mismatch issues
4. **High Impact**: Directly answers research question
5. **Testable**: Quick tests verify pipeline before full runs

## Files Changed

### New Files
- `create_cam_trials_from_all_files.py`
- `hpc_cam_test.sh`
- `hpc_cam_test.slurm`
- `FINAL_METHODOLOGY.md`
- `TWO_STAGE_TRAINING_CLARIFICATION.md`
- `HPC_QUICK_START.md`
- `CHANGES_SUMMARY.md` (this file)

### Updated Files
- `hpc_cam_replication.sh` - Now uses trial generator
- `transfer_to_hpc.sh` - Includes new scripts
- `hpc_eu_emotion_test.sh` - Minor updates

### Unchanged Files
- `hpc_eu_emotion_replication.sh` - Already uses all files
- `evaluate_on_cam.py` - Works with both datasets
- `finetune_clip_emotions.py` - Works with both datasets

## Next Steps

1. ✅ Transfer updated scripts to HPC
2. ✅ Run quick tests
3. ✅ Verify tests pass
4. ✅ Run full replications
5. ✅ Analyze results
6. ✅ Compare to human baselines

## Questions?

See:
- `FINAL_METHODOLOGY.md` - Complete methodology
- `HPC_QUICK_START.md` - Quick start guide
- `TWO_STAGE_TRAINING_CLARIFICATION.md` - Why separate experiments

