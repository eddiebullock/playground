#!/bin/bash
#SBATCH --job-name=autism_experiments
#SBATCH --output=~/predict_asc_c4/logs/deep_learning_%j.out
#SBATCH --error=~/predict_asc_c4/logs/deep_learning_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-4
#SBATCH --partition=icelake
#SBATCH --account=baron-cohen

# Load modules
module load python/3.9.12/gcc/pdcqf4o5

# Set working directory
cd ~/predict_asc_c4

# Activate virtual environment
source venv/bin/activate

# Set experiment commands
declare -a experiments=(
    "scripts/experiments/experiment_sampling.py --method SMOTE --config experiments/configs/model_config.yaml --output-dir experiments/outputs/sampling_smote"
    "scripts/experiments/experiment_feature_engineering.py --method logistic_regression --config experiments/configs/model_config.yaml --output-dir experiments/outputs/feature_engineering_lr"
    "scripts/experiments/experiment_models.py --config experiments/configs/model_config.yaml --output-dir experiments/outputs/model_comparison"
    "scripts/experiments/experiment_nested_cv.py --model xgboost --config experiments/configs/model_config.yaml --output-dir experiments/outputs/nested_cv_xgb"
    "scripts/experiments/experiment_xgboost_grid.py --config experiments/configs/xgboost_grid.yaml --output-dir experiments/outputs/xgboost_grid"
)

# Get experiment name for logging
declare -a experiment_names=(
    "sampling"
    "feature_engineering"
    "models"
    "nested_cv"
    "xgboost_grid"
)

# Get current experiment
EXPERIMENT_CMD="${experiments[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT_NAME="${experiment_names[$SLURM_ARRAY_TASK_ID]}"

echo "Starting experiment: $EXPERIMENT_NAME"
echo "Command: $EXPERIMENT_CMD"

# Run the experiment
python $EXPERIMENT_CMD

if [ $? -eq 0 ]; then
    echo "Experiment $EXPERIMENT_NAME completed successfully"
else
    echo "Experiment $EXPERIMENT_NAME failed"
    exit 1
fi 