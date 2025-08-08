#!/bin/bash
#SBATCH --job-name=autism_ml_parallel
#SBATCH --output=logs/optimization_parallel_%j.out
#SBATCH --error=logs/optimization_parallel_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=icelake

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

# Function to run traditional ML optimization
run_traditional_ml() {
    echo "Starting Traditional ML optimization (PID: $$)..."
    python traditional_ml_optimization.py --config hpc_config.yaml
    echo "Traditional ML optimization completed!"
}

# Function to run deep learning optimization
run_deep_learning() {
    echo "Starting Deep Learning optimization (PID: $$)..."
    python hpc_deep_learning.py --config hpc_config.yaml --k 100
    echo "Deep Learning optimization completed!"
}

# Run both optimizations in parallel
echo "Starting parallel optimization..."
run_traditional_ml &
TRADITIONAL_PID=$!

run_deep_learning &
DEEP_LEARNING_PID=$!

# Wait for both processes to complete
echo "Waiting for both optimizations to complete..."
wait $TRADITIONAL_PID
TRADITIONAL_EXIT=$?

wait $DEEP_LEARNING_PID
DEEP_LEARNING_EXIT=$?

# Check if both completed successfully
if [ $TRADITIONAL_EXIT -eq 0 ] && [ $DEEP_LEARNING_EXIT -eq 0 ]; then
    echo "All optimizations completed successfully!"
else
    echo "One or more optimizations failed!"
    echo "Traditional ML exit code: $TRADITIONAL_EXIT"
    echo "Deep Learning exit code: $DEEP_LEARNING_EXIT"
    exit 1
fi 