#!/bin/bash
#SBATCH --job-name=autism_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --partition=icelake

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Set working directory
cd ~/predict_asc_c4

# Run baseline model training
echo "Starting baseline model training at $(date)"

python src/baseline_models.py \
    --data_path data/data_c4_raw.csv \
    --output_dir results/baseline \
    --test_size 0.2 \
    --random_state 42

echo "Baseline model training completed at $(date)" 