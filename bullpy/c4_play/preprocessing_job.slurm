#!/bin/bash
#SBATCH --job-name=autism_preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/preprocessing_%j.out
#SBATCH --error=logs/preprocessing_%j.err
#SBATCH --partition=icelake

# Load modules
module load python/3.9.12/gcc/pdcqf4o5

# Activate your virtual environment
source venv/bin/activate

# Run the preprocessing pipeline
python scripts/cli/run_preprocessing.py --config experiments/configs/data_config.yaml --verbose 