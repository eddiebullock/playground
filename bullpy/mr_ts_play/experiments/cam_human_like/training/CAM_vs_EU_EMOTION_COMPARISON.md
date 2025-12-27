# CAM vs EU-Emotion: Data Discovery Comparison

## Key Difference

**CAM**: Uses **pre-defined** trial definitions (fixed 100 trials)
**EU-Emotion**: **Discovers** all files first, then **generates** trials from them

## CAM Experiment

### How It Works:
1. **Trial Definitions**: Pre-defined in `cam_trial_definitions_20concepts.json`
   - Exactly 100 specific trials
   - Manually curated with carefully selected foils
   - Matches original Golan et al. (2006) methodology

2. **Training**:
   - Loads trials from JSON file
   - **Does NOT** scan dataset directory
   - Only uses the 100 specific trials
   - Result: ~80 training trials (from 100 defined)

3. **Limitation**:
   - Cannot use more files (experimental design constraint)
   - Fixed to 100 trials to match original study

## EU-Emotion Experiment

### How It Works:

1. **Step 1: Discover All Files** (`create_eu_emotion_trials.py`)
   - Uses `EUEmotionDataset` to **scan the entire dataset directory**
   - Discovers **ALL available face/voice files**
   - Groups files by emotion
   - Finds all valid video files in the dataset

2. **Step 2: Generate Trials**
   - Creates forced-choice trials from discovered files
   - Generates 10 trials per emotion (if enough files available)
   - Each trial: 1 correct emotion + 3 foils (wrong answers)
   - Creates train/test split (80/20)

3. **Step 3: Train on Generated Trials**
   - Uses `TaskSpecificTrialDataset` (same as CAM)
   - Only trains on trials in the generated JSON file
   - Result: ~270 trials (27 emotions × 10 trials)

### Key Difference:
- **Discovers ALL files first** (can use more data)
- **Generates trials** from available files (flexible)
- **Trains on specific generated trials** (still controlled)

## Comparison Table

| Aspect | CAM | EU-Emotion |
|--------|-----|------------|
| **File Discovery** | ❌ No (uses pre-defined) | ✅ Yes (scans directory) |
| **Trial Definitions** | Pre-defined (100 trials) | Generated from files |
| **Number of Trials** | Fixed (100) | Flexible (~270) |
| **Uses All Files?** | ❌ No (only 100 specific) | ✅ Yes (discovers all, generates trials) |
| **Training Data** | ~80 trials | ~216 trials (80% of ~270) |
| **Experimental Design** | Matches original study | Flexible methodology |

## Why The Difference?

### CAM:
- **Experimental replication** of Golan et al. (2006)
- Must use **exact same trials** for fair comparison
- Cannot change trial definitions
- Rigorous scientific comparison

### EU-Emotion:
- **Pre-training dataset** (not a replication)
- Goal is to **learn emotion representations**
- Can use **all available data**
- More flexible, data-driven approach

## Summary

**CAM**: 
- ❌ Does NOT use all ~2,496 valid files
- Uses only 100 pre-defined trials
- By design (experimental replication)

**EU-Emotion**:
- ✅ DOES discover and use all available files
- Generates trials from discovered files
- More flexible, can use more data
- Still trains on specific generated trials (not all files directly)

## Current Results

### CAM:
- Expected: 80 training trials
- Actual: ~40 usable (voice corrupted)
- Missing: ~40 trials

### EU-Emotion:
- Expected: ~216 training trials (80% of ~270)
- Actual: ~216 trials (all valid)
- Uses: All discovered files (generates trials from them)

## Recommendation

**For CAM**: Fix corrupted files to get all 100 trials usable
**For EU-Emotion**: Already using all available files (working as designed)



