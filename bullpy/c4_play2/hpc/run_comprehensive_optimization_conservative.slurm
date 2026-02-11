#!/bin/bash
#SBATCH --job-name=autism_ml_comprehensive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/eb2007/c4/logs/comprehensive_optimization_%j.out
#SBATCH --error=/home/eb2007/c4/logs/comprehensive_optimization_%j.err
#SBATCH --partition=icelake

# Load Python module
module load python/3.9.12/gcc/pdcqf4o5

# Create logs directory
mkdir -p /home/eb2007/c4/logs

# Activate virtual environment
source /home/eb2007/c4/venv/bin/activate

# Change to working directory
cd /home/eb2007/c4

# Create necessary directories
mkdir -p results models logs plots

# Check if required packages are installed
echo "Checking required packages..."
python -c "import sklearn, xgboost, lightgbm, pandas, numpy, joblib, yaml" 2>/dev/null || {
    echo "Installing required packages..."
    pip install scikit-learn xgboost lightgbm pandas numpy joblib pyyaml
}

# Check for CatBoost (optional but recommended)
python -c "import catboost" 2>/dev/null || {
    echo "Installing CatBoost (optional but recommended)..."
    pip install catboost
}

# Run comprehensive optimization
echo "Starting comprehensive ML optimization..."
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

python comprehensive_ml_optimization.py --config hpc_config_comprehensive.yaml

echo "Optimization completed!"
echo "Timestamp: $(date)"
