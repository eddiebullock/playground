# Comprehensive ML Optimization for Autism Classification

## Dataset Check

**Current dataset on HPC:** `/home/eb2007/c4/data/processed/data_c4_balanced_fe.csv`

**Dataset used for cross-validation models:** `data_c4_final_recreated_cleaned.csv` (from notebooks)

**Status:** The `data_c4_balanced_fe.csv` file on HPC should work as it's the feature-engineered version. However, if you want to match exactly what was used for the cross-validation models, you may need to transfer `data_c4_final_recreated_cleaned.csv` instead.

## Files Created

1. **hpc_config_comprehensive.yaml** - Expanded hyperparameter grids
2. **comprehensive_ml_optimization.py** - Main optimization script
3. **run_comprehensive_optimization.slurm** - SLURM job script

## Transfer Commands

### From your local machine, run:

```bash
# Navigate to the project directory
cd /Users/eb2007/playground/bullpy/c4_play2

# Transfer the comprehensive optimization files
rsync -avz --progress \
  hpc/hpc_config_comprehensive.yaml \
  hpc/comprehensive_ml_optimization.py \
  hpc/run_comprehensive_optimization.slurm \
  eb2007@login.hpc.cam.ac.uk:/home/eb2007/c4/

# If you need to transfer the dataset (check first if data_c4_balanced_fe.csv is sufficient)
# rsync -avz --progress \
#   data/processed/data_c4_final_recreated_cleaned.csv \
#   eb2007@login.hpc.cam.ac.uk:/home/eb2007/c4/data/processed/
```

## On HPC - Setup Steps

```bash
# SSH to HPC
ssh eb2007@login.hpc.cam.ac.uk

# Navigate to working directory
cd /home/eb2007/c4

# Verify files are there
ls -lh hpc_config_comprehensive.yaml comprehensive_ml_optimization.py run_comprehensive_optimization.slurm

# Verify dataset exists
ls -lh data/processed/data_c4_balanced_fe.csv

# Activate virtual environment (if not already activated)
source venv/bin/activate

# Install CatBoost if not already installed
pip install catboost

# Submit the job
sbatch run_comprehensive_optimization.slurm

# Monitor the job
squeue -u eb2007

# Check logs
tail -f logs/comprehensive_optimization_*.out
```

## What the Script Does

1. **Feature Selection Optimization**: Tests different numbers of features (50, 75, 100, 150, 200)
2. **Hyperparameter Tuning**: Comprehensive grid search for:
   - Random Forest (expanded grid)
   - XGBoost (expanded grid)
   - LightGBM (expanded grid)
   - Gradient Boosting (expanded grid)
   - Extra Trees (new)
   - CatBoost (new)
   - Logistic Regression (expanded grid)
3. **Threshold Optimization**: Finds optimal threshold for each model to maximize F1
4. **Ensemble Creation**: Creates voting ensemble from top 5 models
5. **Results Saving**: Saves models, scalers, feature selectors, and results

## Expected Runtime

- **Time**: 24-48 hours (depending on HPC load)
- **Resources**: 32 CPUs, 128GB RAM
- **Output**: Results in `results/` and models in `models/`

## Expected Improvements

Based on current performance (F1 ~0.835, AUC ~0.90):
- **Realistic target**: F1 0.87-0.90, AUC 0.92-0.94
- **Best case**: F1 0.90-0.92, AUC 0.94-0.95

## Monitoring

```bash
# Check job status
squeue -u eb2007

# View output log
tail -f logs/comprehensive_optimization_*.out

# View error log
tail -f logs/comprehensive_optimization_*.err

# Check results as they're generated
ls -lh results/
ls -lh models/
```

## Results Location

After completion, results will be in:
- `results/comprehensive_results_*.json` - Full results with hyperparameters
- `results/comprehensive_results_*.csv` - Summary table
- `models/*_optimized_*.joblib` - Trained models
- `models/scaler_*.joblib` - Scaler
- `models/feature_selector_*.joblib` - Feature selector (if used)
