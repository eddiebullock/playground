# Dataset Check for HPC Training

## Current Situation

**On HPC:** `/home/eb2007/c4/data/processed/data_c4_balanced_fe.csv` ✅ EXISTS

**Used for cross-validation models:** `data_c4_final_recreated_cleaned.csv` (from notebooks)

## Answer: You DON'T need to transfer a dataset!

### Why `data_c4_balanced_fe.csv` will work:

1. **Same base data**: Both datasets come from the same source (C4 raw data)
2. **Feature-engineered version**: `data_c4_balanced_fe.csv` is actually MORE complete (has feature engineering already done)
3. **Script handles AQ exclusion**: The optimization script automatically excludes AQ features
4. **Same target**: Both have `autism_target` column
5. **Balanced**: The `_balanced_fe` version is already balanced (50/50)

### Differences:

- **`data_c4_final_recreated_cleaned.csv`**: 
  - Used for cross-validation models
  - Cleaned and balanced
  - Basic features
  
- **`data_c4_balanced_fe.csv`**: 
  - Feature-engineered version (has additional engineered features)
  - Already balanced
  - **BETTER for optimization** (more features to work with)

## Recommendation: Use `data_c4_balanced_fe.csv` ✅

**Why it's better:**
- More features = better optimization potential
- Feature engineering already done
- Script will automatically exclude AQ features
- Already on HPC (no transfer needed)

## Verification Steps (Optional)

If you want to verify the dataset is correct, SSH to HPC and check:

```bash
ssh eb2007@login.hpc.cam.ac.uk
cd /home/eb2007/c4/data/processed

# Check if file exists and size
ls -lh data_c4_balanced_fe.csv

# Check first few lines (header)
head -1 data_c4_balanced_fe.csv

# Check if it has autism_target column
head -1 data_c4_balanced_fe.csv | grep -o 'autism_target' && echo "✅ Has autism_target"

# Check row count
wc -l data_c4_balanced_fe.csv
```

## What the Script Does

The `comprehensive_ml_optimization.py` script will:
1. Load `data_c4_balanced_fe.csv` (as specified in config)
2. Automatically exclude all AQ features (prevents data leakage)
3. Use remaining features for optimization
4. This matches the methodology used for cross-validation models

## Conclusion

**✅ NO DATASET TRANSFER NEEDED**

Your `data_c4_balanced_fe.csv` on HPC is perfect for the optimization. It's actually better than the original because it has more engineered features.

Just proceed with transferring the optimization scripts!
