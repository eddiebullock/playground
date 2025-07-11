#!/bin/bash
#SBATCH --job-name=autism_deep_learning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/deep_learning_%j.out
#SBATCH --error=logs/deep_learning_%j.err
#SBATCH --partition=icelake

# If GPU is available and needed, uncomment the next line:
# SBATCH --gres=gpu:1

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Set working directory
cd ~/predict_asc_c4

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run deep learning experiment
python scripts/experiments/experiment_deep_learning.py \
    --data_path data/processed/features_full.csv \
    --output_dir experiments/logs/deep_learning \
    --target_col autism_any \
    --epochs 20 \
    --batch_size 256 \
    --test_size 0.2 \
    --val_size 0.2 \
    --seed 42 