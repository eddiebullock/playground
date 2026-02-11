# Transfer Instructions for Comprehensive ML Optimization

## Quick Summary

**Dataset Status:** You already have `data_c4_balanced_fe.csv` on HPC at `/home/eb2007/c4/data/processed/`. This should work, but the cross-validation models were trained on `data_c4_final_recreated_cleaned.csv`. The balanced_fe version should be fine as it's feature-engineered.

## Step 1: Transfer Files from Local to HPC

Run these commands from your local machine (in the project directory):

```bash
cd /Users/eb2007/playground/bullpy/c4_play2

# Transfer the three new files
rsync -avz --progress \
  hpc/hpc_config_comprehensive.yaml \
  hpc/comprehensive_ml_optimization.py \
  hpc/run_comprehensive_optimization.slurm \
  eb2007@login.hpc.cam.ac.uk:/home/eb2007/c4/
```

## Step 2: Verify Dataset (Optional Check)

If you want to verify the dataset is correct, SSH to HPC and check:

```bash
ssh eb2007@login.hpc.cam.ac.uk
cd /home/eb2007/c4/data/processed
ls -lh data_c4_balanced_fe.csv
head -1 data_c4_balanced_fe.csv  # Check header
```

The dataset should have:
- `autism_target` column
- No AQ features (they'll be excluded automatically)
- Feature-engineered columns

## Step 3: Submit Job on HPC

```bash
# SSH to HPC (if not already there)
ssh eb2007@login.hpc.cam.ac.uk

# Navigate to working directory
cd /home/eb2007/c4

# Verify files transferred correctly
ls -lh hpc_config_comprehensive.yaml comprehensive_ml_optimization.py run_comprehensive_optimization.slurm

# Activate virtual environment
source venv/bin/activate

# Install CatBoost if needed
pip install catboost

# Submit job
sbatch run_comprehensive_optimization.slurm

# Check job status
squeue -u eb2007

# Monitor output
tail -f logs/comprehensive_optimization_*.out
```

## What Gets Optimized

1. **Feature Selection**: Tests k=50, 75, 100, 150, 200 features
2. **7 Models** with expanded hyperparameter grids:
   - Random Forest
   - XGBoost  
   - LightGBM
   - Gradient Boosting
   - Extra Trees (new)
   - CatBoost (new)
   - Logistic Regression
3. **Threshold Optimization** for each model
4. **Ensemble** from top 5 models

## Expected Results

- **Current**: F1 ~0.835, AUC ~0.90
- **Target**: F1 0.87-0.90, AUC 0.92-0.94
- **Runtime**: 24-48 hours

## Files Created

After completion, you'll have:
- `results/comprehensive_results_*.json` - Full results
- `results/comprehensive_results_*.csv` - Summary table
- `models/*_optimized_*.joblib` - All trained models
- `models/scaler_*.joblib` - Scaler
- `models/feature_selector_*.joblib` - Feature selector
