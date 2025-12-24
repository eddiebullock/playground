# Research Methodology Recommendation: High-Impact Mental State Recognition Study

## Research Question

**Primary**: Can computer vision identify mental states accurately?  
**Secondary**: Can it approach/match human performance levels?

**Target Performance**:
- **Human (Control)**: ~88% accuracy
- **Human (AS)**: ~70% accuracy  
- **Current Zero-shot CLIP**: 37% accuracy
- **Target**: **70-85% accuracy** (approaching human performance)

## Recommended Unified Methodology

### Core Principle: **Use ALL Available Data + Consistent Methodology**

For maximum impact and robustness, both datasets should follow the **same procedure**:

1. ✅ **Discover ALL available valid files** (maximize data usage)
2. ✅ **Generate trials from discovered files** (consistent approach)
3. ✅ **Two-stage training** (EU-Emotion → CAM)
4. ✅ **Comprehensive evaluation** (compare to human baselines)

## Unified Procedure for Both Datasets

### Step 1: Discover All Files
- Scan dataset directory
- Find ALL valid video files (>50KB)
- Group by emotion/concept and modality
- **Result**: Use all available data, not just a subset

### Step 2: Generate Trials
- Create forced-choice trials (4 options each)
- 10-15 trials per emotion/concept (if enough files)
- Proper foil selection (semantically different)
- **Result**: More training data, better learning

### Step 3: Train/Test Split
- 80/20 split (concept-balanced)
- Ensures each concept in both splits
- **Result**: Proper evaluation, no data leakage

### Step 4: Training
- Two-stage approach:
  - **Stage 1**: Pre-train on EU-Emotion (all files)
  - **Stage 2**: Fine-tune on CAM (all files)
- **Result**: Better performance than single-stage

## Why This Approach?

### 1. Maximizes Data Usage
- **CAM**: Uses ~200-300 trials (from all ~2,496 valid files)
- **EU-Emotion**: Uses ~270 trials (from all discovered files)
- **Benefit**: More training data = better model performance

### 2. Consistent Methodology
- Same approach for both datasets
- Fair comparison
- Easier to interpret results

### 3. Better Performance
- More data → less underfitting
- Two-stage training → better representations
- Expected: **70-85% accuracy** (approaching human)

### 4. Maintains Rigor
- Proper train/test splits
- No data leakage
- Reproducible methodology
- Comparable to human baselines

## Implementation

### For EU-Emotion (Already Working)
- ✅ Uses `create_eu_emotion_trials.py`
- ✅ Discovers all files
- ✅ Generates trials
- ✅ **No changes needed**

### For CAM (Needs Update)
- ❌ Currently uses only 100 pre-defined trials
- ✅ **New script**: `create_cam_trials_from_all_files.py`
- ✅ Discovers all valid files
- ✅ Generates trials from all files
- ✅ Maintains 20 concepts (for comparison)

## Expected Results

### Stage 1: EU-Emotion Pre-training
- **Training trials**: ~216 (80% of ~270)
- **Test trials**: ~54 (20% of ~270)
- **Expected accuracy**: 60-75%
- **Purpose**: Learn general emotion representations

### Stage 2: CAM Fine-tuning
- **Training trials**: ~160-240 (80% of ~200-300)
- **Test trials**: ~40-60 (20% of ~200-300)
- **Expected accuracy**: **70-85%**
- **Comparison**:
  - Zero-shot: 37%
  - Human (AS): 70%
  - Human (Control): 88%
  - **Target**: 70-85% ✅

## Comparison: Current vs Recommended

| Aspect | Current | Recommended |
|--------|---------|-------------|
| **CAM Data Usage** | 100 trials | ~200-300 trials |
| **EU-Emotion Data** | All files ✅ | All files ✅ |
| **Methodology** | Inconsistent | Unified ✅ |
| **Training** | Single-stage | Two-stage ✅ |
| **Expected Accuracy** | 50-60% | 70-85% ✅ |
| **Human Comparison** | Limited | Comprehensive ✅ |
| **Impact** | Moderate | **High** ✅ |

## Advantages

### 1. Scientific Rigor
- Proper methodology
- Reproducible
- Comparable to baselines

### 2. Maximum Performance
- Uses all available data
- Two-stage training
- Better representations

### 3. High Impact
- Addresses research question directly
- Shows CV can approach human performance
- Publication-ready results

### 4. Consistent Approach
- Same methodology for both datasets
- Fair comparison
- Easier interpretation

## Next Steps

1. **Test CAM trial generator** (`create_cam_trials_from_all_files.py`)
   - Verify it discovers all valid files
   - Check trial generation quality
   - Validate train/test splits

2. **Update training pipeline**
   - Support two-stage training
   - Unified interface

3. **Run experiments**
   - Stage 1: EU-Emotion pre-training
   - Stage 2: CAM fine-tuning
   - Evaluation

4. **Analysis**
   - Compare to human baselines
   - Statistical significance
   - Comprehensive metrics

## Conclusion

For a **high-impact study** addressing whether computer vision can identify mental states accurately:

✅ **Use ALL available data** (maximize learning)  
✅ **Unified methodology** (consistent approach)  
✅ **Two-stage training** (better performance)  
✅ **Comprehensive evaluation** (compare to human baselines)  
✅ **Maintain rigor** (proper methodology)

This approach will:
- **Maximize performance** (using all data)
- **Maintain scientific rigor** (proper methodology)
- **Enable fair comparison** (consistent approach)
- **Address research question** (can CV approach human performance?)
- **Generate high impact** (publication-ready results)

**Expected outcome**: 70-85% accuracy, approaching human performance levels (70-88%).

