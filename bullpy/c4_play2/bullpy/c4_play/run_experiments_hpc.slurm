#!/bin/bash
#SBATCH --job-name=autism_experiments
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=experiments/logs/experiments_%j.out
#SBATCH --error=experiments/logs/experiments_%j.err

# Exit on any error
set -e

# Load modules
module load python/3.9.12/gcc/pdcqf4o5

# Change to project directory (adjust path as needed)
cd /home/eb2007/predict_asc_c4

# Activate virtual environment
source venv/bin/activate

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p experiments/outputs
mkdir -p experiments/logs

echo "Starting autism prediction experiments on HPC..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Check if data file exists
if [ ! -f "data/processed/features_full.csv" ]; then
    echo "ERROR: Data file not found at data/processed/features_full.csv"
    echo "Available files in data/processed/:"
    ls -la data/processed/ || echo "data/processed/ directory not found"
    exit 1
fi

echo "Data file found. Starting experiments..."

# Run all experiments
python run_all_experiments_improved.py --output-base-dir experiments/outputs

echo "Experiments completed at $(date)"

# Extract and summarize results
echo "Extracting results..."
python extract_all_experiment_results.py

echo "All done! Results saved to experiments/outputs/ and all_experiment_results_summary.txt"
echo "Check experiments/logs/ for detailed logs" 