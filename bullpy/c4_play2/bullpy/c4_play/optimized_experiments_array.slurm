#!/bin/bash
#SBATCH --job-name=autism_experiments
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --output=experiments_%A_%a.out
#SBATCH --error=experiments_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=eb2007@cam.ac.uk

# Load modules
module load python/3.9.12/gcc/pdcqf4o5

# Activate virtual environment
source venv/bin/activate

# Set up environment
export PYTHONPATH="${PYTHONPATH}:/home/eb2007/predict_asc_c4"
export PYTHONUNBUFFERED=1

# Create output directories
mkdir -p experiments/outputs
mkdir -p experiments/logs

# Define experiment configurations
declare -a experiments=(
    "sampling:scripts/experiments/experiment_sampling.py --method SMOTE --config experiments/configs/model_config.yaml --output-dir experiments/outputs/sampling_smote"
    "feature_engineering:scripts/experiments/experiment_feature_engineering.py --method logistic_regression --config experiments/configs/model_config.yaml --output-dir experiments/outputs/feature_engineering_lr"
    "models:scripts/experiments/experiment_models.py --config experiments/configs/model_config.yaml --output-dir experiments/outputs/model_comparison"
    "nested_cv:scripts/experiments/experiment_nested_cv.py --model xgboost --config experiments/configs/model_config.yaml --output-dir experiments/outputs/nested_cv_xgb"
    "xgboost_grid:scripts/experiments/experiment_xgboost_grid.py --config experiments/configs/xgboost_grid.yaml --output-dir experiments/outputs/xgboost_grid"
)

# Get the experiment for this array task
experiment="${experiments[$SLURM_ARRAY_TASK_ID]}"
experiment_name=$(echo "$experiment" | cut -d: -f1)
experiment_command=$(echo "$experiment" | cut -d: -f2-)

echo "Starting experiment: $experiment_name"
echo "Command: $experiment_command"

# Run the experiment
python $experiment_command

# Check exit status
if [ $? -eq 0 ]; then
    echo "Experiment $experiment_name completed successfully"
else
    echo "Experiment $experiment_name failed"
    exit 1
fi 