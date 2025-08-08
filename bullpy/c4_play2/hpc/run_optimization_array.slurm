#!/bin/bash
#SBATCH --job-name=autism_ml_array
#SBATCH --output=logs/optimization_%A_%a.out
#SBATCH --error=logs/optimization_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=icelake
#SBATCH --array=1-2

# Load modules
module load python/3.6
module load gcc/8.3.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import sklearn, xgboost, lightgbm, pandas, numpy" 2>/dev/null; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Requirements already installed"
fi

# Create necessary directories
mkdir -p logs results models plots data

# Run different optimizations based on array task ID
case $SLURM_ARRAY_TASK_ID in
    1)
        echo "=== TASK 1: Traditional ML Optimization ==="
        python traditional_ml_optimization.py --config hpc_config.yaml
        ;;
    2)
        echo "=== TASK 2: Deep Learning Optimization ==="
        python hpc_deep_learning.py --config hpc_config.yaml --k 100
        ;;
    *)
        echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID completed!" 