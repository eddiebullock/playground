#!/bin/bash

# HPC Environment Setup Script for Autism Prediction Project
# Run this script on Cambridge HPC to set up your environment

echo "Setting up HPC environment for autism prediction project..."

# Create project directory
mkdir -p ~/autism_project
cd ~/autism_project

# Create logs directory
mkdir -p logs

# Load Python module
module load python/3.9

# Create virtual environment
echo "Creating virtual environment..."
python -m venv autism_env
source autism_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install imbalanced-learn optuna joblib

# Test environment
echo "Testing environment..."
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
print('Environment setup successful!')
print(f'Pandas version: {pd.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'XGBoost version: {xgb.__version__}')
"

# Create requirements.txt for future reference
pip freeze > requirements.txt

echo "Environment setup complete!"
echo "Next steps:"
echo "1. Upload your data files to ~/autism_project/"
echo "2. Upload your src/ directory to ~/autism_project/"
echo "3. Update data paths in your scripts"
echo "4. Submit test job: sbatch quick_test.slurm" 