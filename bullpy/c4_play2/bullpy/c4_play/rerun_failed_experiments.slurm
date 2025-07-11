#!/bin/bash
#SBATCH --job-name=rerun_failed
#SBATCH --partition=icelake
#SBATCH --account=baron-cohen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=rerun_failed_%j.out
#SBATCH --error=rerun_failed_%j.err

# Load modules
module load python/3.9.12/gcc/pdcqf4o5

# Activate virtual environment
source venv/bin/activate

# Set up logging
export PYTHONPATH="${PYTHONPATH}:/home/eb2007/predict_asc_c4"

echo "Starting rerun of failed experiments..."

# 1. Rerun modular training with random_forest (was job 11929488)
echo "=== Rerunning modular training with random_forest ==="
python src/model_training.py --config experiments/configs/model_config.yaml --model random_forest

# 2. Rerun sampling experiment with SMOTE (was job 11930831_0)
echo "=== Rerunning sampling experiment with SMOTE ==="
python scripts/experiments/experiment_sampling.py --method SMOTE --output_dir experiments/outputs/sampling_smote

# 3. Rerun feature engineering experiment with logistic_regression (was job 11930831_1)
echo "=== Rerunning feature engineering experiment with logistic_regression ==="
python scripts/experiments/experiment_feature_engineering.py --model logistic_regression --output_dir experiments/outputs/feature_engineering_lr

# 4. Rerun nested CV experiment (was job 11930831_3)
echo "=== Rerunning nested CV experiment ==="
python scripts/experiments/experiment_nested_cv.py --model xgboost --outer_folds 3 --inner_folds 3 --output_dir experiments/outputs/nested_cv_xgb

echo "All failed experiments rerun completed!" 