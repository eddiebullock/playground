#!/bin/bash
#SBATCH --job-name=autism_ml_optimization
#SBATCH --output=logs/optimization_%j.out
#SBATCH --error=logs/optimization_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=cclake,icelake,desktop

# Load modules (adjust based on your HPC system)
module load python/3.6
module load gcc/8.3.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate virtual environment (already created by setup_hpc.sh)
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

# Run Traditional ML Optimization
echo "Starting Traditional ML optimization..."
python traditional_ml_optimization.py --config hpc_config.yaml

# Run Deep Learning Optimization
echo "Starting Deep Learning optimization..."
python hpc_deep_learning.py --config hpc_config.yaml --k 100

echo "All optimizations completed!" 