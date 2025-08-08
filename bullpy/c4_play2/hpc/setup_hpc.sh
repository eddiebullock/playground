#!/bin/bash
# HPC Setup Script for Autism Classification Optimization
# This script sets up the environment and transfers necessary files

echo "Setting up HPC environment for autism classification optimization..."

# Create necessary directories
mkdir -p logs results models plots data

# Make scripts executable
chmod +x slurm_scripts/*.slurm
chmod +x *.py

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .gitignore for data files
echo "Creating .gitignore for data files..."
cat > .gitignore << EOF
# Data files (confidential)
data_c4_enhanced_fe_v2.csv
data_c4_matched_balanced.csv
data_c4_balanced_fe.csv

# Results and models
results/
models/
logs/
plots/

# Virtual environment
venv/

# Python cache
__pycache__/
*.pyc
*.pyo

# Jupyter notebooks
*.ipynb_checkpoints/

# SLURM output files
slurm-*.out
slurm-*.err

# Temporary files
*.tmp
*.temp
EOF

echo "HPC environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Copy your data file 'data_c4_enhanced_fe_v2.csv' to the hpc/data/ directory"
echo "2. Update the data path in hpc_config.yaml if needed"
echo "3. Submit jobs using: sbatch slurm_scripts/run_hyperparameter_tuning.slurm"
echo "4. Monitor jobs using: squeue -u $USER" 