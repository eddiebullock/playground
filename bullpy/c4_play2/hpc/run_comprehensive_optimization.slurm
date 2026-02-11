#!/bin/bash
#SBATCH --job-name=autism_ml_opt
#SBATCH --output=/home/eb2007/c4/logs/comprehensive_optimization_%j.out
#SBATCH --error=/home/eb2007/c4/logs/comprehensive_optimization_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p icelake

# Load Python module (robust, like mr_ts_play scripts)
module purge
if module load python/3.11.0-icl 2>/dev/null; then
    echo "Loaded python/3.11.0-icl"
elif module load python 2>/dev/null; then
    echo "Loaded default python module"
else
    echo "Warning: could not load a python module; relying on venv python"
fi

# Create logs directory
mkdir -p /home/eb2007/c4/logs

# Activate virtual environment
source /home/eb2007/c4/venv/bin/activate

# Set threads to match cpus-per-task
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

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
