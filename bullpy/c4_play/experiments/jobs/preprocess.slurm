#!/bin/bash
#SBATCH --job-name=autism_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

# Load modules (adjust for your HPC system)
module load python/3.11
module load anaconda3

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Set working directory
cd /path/to/your/project

# Create logs directory
mkdir -p logs

# Run preprocessing pipeline
python scripts/cli/run_preprocessing.py \
    --config experiments/configs/data_config.yaml \
    --verbose

echo "Preprocessing job completed at $(date)" 