#!/bin/bash
#SBATCH --job-name=autism_exp_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp_array_%A_%a.out
#SBATCH --error=logs/exp_array_%A_%a.err
#SBATCH --partition=icelake
#SBATCH --array=0-4

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Set working directory
cd ~/predict_asc_c4

# Activate your virtual environment
source venv/bin/activate

# Define experiment commands
EXPERIMENTS=(
  "python scripts/experiments/experiment_sampling.py --sampling-method SMOTE --config experiments/configs/model_config.yaml --output-dir experiments/outputs/sampling_SMOTE"
  "python scripts/experiments/experiment_feature_engineering.py --feature-method interactions --config experiments/configs/model_config.yaml --output-dir experiments/outputs/fe_interactions"
  "python scripts/experiments/experiment_models.py --model XGBoost --config experiments/configs/model_config.yaml --output-dir experiments/outputs/model_XGBoost"
  "python scripts/experiments/experiment_nested_cv.py --outer-folds 5 --inner-folds 3 --model XGBoost --config experiments/configs/model_config.yaml --output-dir experiments/outputs/nested_cv_XGBoost"
  "python scripts/experiments/experiment_xgboost_grid.py --config experiments/configs/model_config.yaml --param-grid experiments/configs/xgboost_grid.yaml --output-dir experiments/outputs/xgb_grid"
)

# Run the experiment for this array task
${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]} 