# Quick Start Guide - Comprehensive ML Optimization

## ✅ Dataset Verified
- File: `data_c4_balanced_fe.csv` ✅
- Rows: 90,539 ✅
- Has `autism_target` ✅
- Has all required features ✅

## Step 1: Transfer Scripts from Local to HPC

**Run this from your local machine:**

```bash
cd /Users/eb2007/playground/bullpy/c4_play2

rsync -avz --progress \
  hpc/hpc_config_comprehensive.yaml \
  hpc/comprehensive_ml_optimization.py \
  hpc/run_comprehensive_optimization.slurm \
  eb2007@login.hpc.cam.ac.uk:/home/eb2007/c4/
```

## Step 2: On HPC - Verify Files Transferred

```bash
ssh eb2007@login.hpc.cam.ac.uk
cd /home/eb2007/c4

# Verify files are there
ls -lh hpc_config_comprehensive.yaml comprehensive_ml_optimization.py run_comprehensive_optimization.slurm
```

## Step 3: Setup Environment on HPC

```bash
# Activate virtual environment
source venv/bin/activate

# Install CatBoost if not already installed
pip install catboost

# Verify required packages
python -c "import sklearn, xgboost, lightgbm, pandas, numpy, joblib, yaml, catboost; print('All packages OK')"
```

## Step 4: Create Required Directories

```bash
cd /home/eb2007/c4
mkdir -p results models logs plots
```

## Step 5: Submit the Job

```bash
# Submit the optimization job
sbatch run_comprehensive_optimization.slurm

# Note the job ID that appears (e.g., "Submitted batch job 12345678")
```

## Step 6: Monitor the Job

```bash
# Check job status
squeue -u eb2007

# View output log (replace JOBID with your actual job ID)
tail -f logs/comprehensive_optimization_JOBID.out

# View error log if needed
tail -f logs/comprehensive_optimization_JOBID.err

# Check if results are being generated
ls -lh results/
ls -lh models/
```

## Step 7: Check Progress

The script will:
1. Optimize feature selection (tests k=50, 75, 100, 150, 200)
2. Optimize 7 models with hyperparameter tuning
3. Create ensemble from top 5 models
4. Save all results

**Expected runtime:** 24-48 hours

## Step 8: After Completion

```bash
# Check results
ls -lh results/comprehensive_results_*.json
ls -lh results/comprehensive_results_*.csv

# View summary
cat results/comprehensive_results_*.csv

# Check models saved
ls -lh models/*_optimized_*.joblib
```

## Troubleshooting

### If job fails immediately:
```bash
# Check error log
cat logs/comprehensive_optimization_*.err

# Common issues:
# - Missing packages: pip install catboost
# - Wrong Python version: module load python/3.9.12/gcc/pdcqf4o5
# - Dataset not found: verify data/processed/data_c4_balanced_fe.csv exists
```

### If job is taking too long:
- This is normal! Hyperparameter optimization is computationally intensive
- Check `logs/comprehensive_optimization_*.out` to see progress
- Each model optimization can take 4-8 hours

### To cancel job if needed:
```bash
# Find job ID
squeue -u eb2007

# Cancel job
scancel JOBID
```

## Expected Output Files

After completion, you'll have:
- `results/comprehensive_results_TIMESTAMP.json` - Full results with hyperparameters
- `results/comprehensive_results_TIMESTAMP.csv` - Summary table (sorted by F1)
- `models/random_forest_optimized_TIMESTAMP.joblib` - Optimized models
- `models/xgboost_optimized_TIMESTAMP.joblib`
- `models/lightgbm_optimized_TIMESTAMP.joblib`
- `models/gradient_boosting_optimized_TIMESTAMP.joblib`
- `models/extra_trees_optimized_TIMESTAMP.joblib`
- `models/catboost_optimized_TIMESTAMP.joblib` (if available)
- `models/logistic_regression_optimized_TIMESTAMP.joblib`
- `models/scaler_TIMESTAMP.joblib` - Scaler used
- `models/feature_selector_TIMESTAMP.joblib` - Feature selector (if used)

## Next Steps After Completion

1. Download results to local machine for analysis
2. Compare with baseline results (F1 ~0.835)
3. Identify best model and ensemble performance
4. Use best model for external validation on CARD dataset
