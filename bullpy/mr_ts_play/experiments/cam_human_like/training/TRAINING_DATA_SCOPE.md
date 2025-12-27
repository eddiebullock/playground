# What Data Does the Model Actually Train On?

## Short Answer

**NO**, the model is **NOT** training on all ~2,496 valid files. It only trains on the **~80 trials** specified in `train_trials.json`.

## Detailed Explanation

### How Training Works

1. **Trial Definitions File** (`cam_trial_definitions_20concepts.json`)
   - Contains exactly **100 specific trials**
   - Each trial specifies:
     - A specific video file path
     - 4 candidate emotion labels (1 correct + 3 foils)
     - Correct answer index
     - Modality (face or voice)

2. **Split Creation** (`create_cam_splits.py`)
   - Takes the 100 trials
   - Splits them 80/20 (train/test)
   - Creates `train_trials.json` (~80 trials) and `test_trials.json` (~20 trials)

3. **Dataset Loader** (`TaskSpecificTrialDataset`)
   - Loads trials from `train_trials.json`
   - **Does NOT** scan the dataset directory
   - **Only uses** the specific trials listed in the JSON file
   - For each trial, loads the video file specified in the trial definition

4. **Training**
   - Model trains on **only the ~80 trials** in `train_trials.json`
   - Does **NOT** use the other ~2,400+ valid files

### Code Evidence

```python
# In TaskSpecificTrialDataset.__init__()
with open(trial_definitions_file, 'r') as f:
    data = json.load(f)
self.trials = data.get('trials', [])  # Only loads trials from JSON
```

The dataset does **NOT** do:
- `os.listdir()` to find all files
- `glob()` to discover videos
- Any automatic file discovery

It **ONLY** uses the trials explicitly listed in the JSON file.

## Why This Design?

### To Match Original CAM Methodology

The CAM experiment follows **Golan et al. (2006)**:
- Uses **specific 100 trials** with carefully selected foils
- Enables fair comparison with human performance
- Maintains experimental rigor

### If We Used All Files

Using all ~2,496 valid files would:
- ❌ Not match the original experimental design
- ❌ Not have proper foil selection (need 3 wrong answers per trial)
- ❌ Not be comparable to published results
- ❌ Be a different experiment entirely

## Current Situation

### What Model Trains On:
- **Expected**: 80 trials (from 100 total)
- **Actual**: ~40 usable trials (only face, voice corrupted)
- **Missing**: ~40 trials (corrupted voice files)

### What Model Does NOT Train On:
- The other ~2,400+ valid files in the dataset
- These files exist but are not part of the experiment design

## Could We Use More Files?

### Yes, but it would be a DIFFERENT experiment:

**Option 1: Data Augmentation Approach**
- Create more trials from the 2,496 valid files
- Generate foils for each new trial
- Would increase training data significantly
- But would not match original CAM methodology

**Option 2: Pre-training Approach**
- Pre-train on all valid files (unsupervised or with emotion labels)
- Fine-tune on the 100 CAM trials
- Could improve performance while maintaining experimental validity

**Option 3: Extended CAM Experiment**
- Create more CAM-style trials with proper foil selection
- Would require manual curation
- Could expand to 200, 300, or more trials
- Still maintains experimental rigor

## Summary

| Question | Answer |
|----------|--------|
| Does model train on all 2,496 files? | **NO** - Only ~80 trials from JSON |
| Is this by design? | **YES** - Matches original CAM methodology |
| Could we use more files? | **YES** - But would be a different experiment |
| Should we use more files? | **MAYBE** - Depends on research goals |

## Recommendation

**For CAM replication** (matching original study):
- ✅ Use only the 100 trials (as designed)
- ✅ Fix corrupted files to get all 100 usable
- ✅ Maintain experimental validity

**For improved performance** (different experiment):
- Could pre-train on all valid files
- Then fine-tune on CAM trials
- Best of both worlds: more data + experimental validity



