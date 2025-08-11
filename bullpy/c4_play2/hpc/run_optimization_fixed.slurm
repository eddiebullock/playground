#!/bin/bash
#SBATCH --job-name=autism_ml_optimization
#SBATCH --output=logs/optimization_%j.out
#SBATCH --error=logs/optimization_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=icelake

# Load available Python module (let the system choose)
echo "Loading Python module..."
module load python

# Load available GCC module (let the system choose)
echo "Loading GCC module..."
module load gcc

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if virtual environment exists and works
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Please run setup_hpc.sh first."
    exit 1
fi

# Test Python in virtual environment
if ! venv/bin/python --version >/dev/null 2>&1; then
    echo "Virtual environment is broken. Please run fix_python_environment.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify Python is working
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Create necessary directories
mkdir -p logs results models plots data

# Check if requirements are installed
echo "Checking requirements..."
if ! python -c "import sklearn, xgboost, lightgbm, pandas, numpy" 2>/dev/null; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Requirements already installed"
fi

# Run Traditional ML Optimization
echo "Starting Traditional ML optimization..."
python traditional_ml_optimization.py --config hpc_config.yaml

# Run Deep Learning Optimization
echo "Starting Deep Learning optimization..."
python hpc_deep_learning.py --config hpc_config.yaml --k 100

echo "All optimizations completed!"
