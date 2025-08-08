#!/bin/bash
#SBATCH --job-name=autism_ml_optimization
#SBATCH --output=logs/optimization_%j.out
#SBATCH --error=logs/optimization_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=compute

# Load modules (adjust based on your HPC system)
module load python/3.9
module load gcc/9.3.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs results models plots data

# Run Traditional ML Optimization
echo "Starting Traditional ML optimization..."
python traditional_ml_optimization.py --config hpc_config.yaml

# Run Deep Learning Optimization
echo "Starting Deep Learning optimization..."
python hpc_deep_learning.py --config hpc_config.yaml --k 100

echo "All optimizations completed!" 