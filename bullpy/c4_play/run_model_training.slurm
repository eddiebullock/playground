#!/bin/bash
#SBATCH --job-name=modular_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/modular_train_%j.out
#SBATCH --error=logs/modular_train_%j.err
#SBATCH --partition=icelake

# Load required modules
module load python/3.9.12/gcc/pdcqf4o5

# Activate your virtual environment
source venv/bin/activate

# Set working directory
cd ~/predict_asc_c4

# Run the modular model training pipeline
python src/model_training.py --config experiments/configs/model_config.yaml 