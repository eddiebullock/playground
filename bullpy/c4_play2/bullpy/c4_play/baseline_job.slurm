#!/bin/bash
#SBATCH --job-name=autism_experiments
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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

# Define experiment configurations using the wrapper script
declare -a experiments=(
    "sampling:python run_experiment.py sampling SMOTE experiments/configs/model_config.yaml experiments/outputs/sampling_smote"
    "feature_engineering:python run_experiment.py feature_engineering logistic_regression experiments/configs/model_config.yaml experiments/outputs/feature_engineering_lr"
    "models:python run_experiment.py models experiments/configs/model_config.yaml experiments/outputs/model_comparison"
    "nested_cv:python run_experiment.py nested_cv xgboost experiments/configs/model_config.yaml experiments/outputs/nested_cv_xgb"
    "xgboost_grid:python run_experiment.py xgboost_grid experiments/configs/xgboost_grid.yaml experiments/outputs/xgboost_grid"
)

# Get the experiment for this array task
experiment="${experiments[$SLURM_ARRAY_TASK_ID]}"
experiment_name=$(echo "$experiment" | cut -d: -f1)
experiment_command=$(echo "$experiment" | cut -d: -f2-)

echo "Starting experiment: $experiment_name"
echo "Command: $experiment_command"

# Run the experiment
$experiment_command

# Check exit status
if [ $? -eq 0 ]; then
    echo "Experiment $experiment_name completed successfully"
else
    echo "Experiment $experiment_name failed"
    exit 1
fi 