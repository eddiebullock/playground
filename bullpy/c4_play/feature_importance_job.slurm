#!/bin/bash
#SBATCH --job-name=autism_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/feature_importance_%j.out
#SBATCH --error=logs/feature_importance_%j.err
#SBATCH --partition=icelake

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Set working directory
cd ~/predict_asc_c4

# Run feature importance analysis
echo "Starting feature importance analysis at $(date)"

python src/feature_importance.py \
    --data_path data/data_c4_raw.csv \
    --output_dir results/feature_importance \
    --test_size 0.2 \
    --random_state 42

echo "Feature importance analysis completed at $(date)" 