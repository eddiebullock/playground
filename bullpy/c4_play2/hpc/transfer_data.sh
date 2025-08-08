#!/bin/bash
# Data Transfer Script for HPC
# This script transfers the necessary data files to the HPC system

echo "Data Transfer Script for HPC Optimization"
echo "========================================"

# Configuration
HPC_USER="your_username"
HPC_HOST="your_hpc_cluster"
HPC_PATH="/path/to/your/project/hpc"
LOCAL_DATA_PATH="../data/processed"

# Files to transfer
DATA_FILES=(
    "data_c4_enhanced_fe_v2.csv"
    "data_c4_matched_balanced.csv"
    "data_c4_balanced_fe.csv"
)

echo "Transferring data files to HPC..."

# Create data directory on HPC if it doesn't exist
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_PATH}/data"

# Transfer each data file
for file in "${DATA_FILES[@]}"; do
    if [ -f "${LOCAL_DATA_PATH}/${file}" ]; then
        echo "Transferring ${file}..."
        scp "${LOCAL_DATA_PATH}/${file}" "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/data/"
        echo "✓ ${file} transferred successfully"
    else
        echo "⚠ Warning: ${file} not found in ${LOCAL_DATA_PATH}"
    fi
done

echo ""
echo "Data transfer completed!"
echo ""
echo "Next steps:"
echo "1. SSH to HPC: ssh ${HPC_USER}@${HPC_HOST}"
echo "2. Navigate to project: cd ${HPC_PATH}"
echo "3. Run setup: ./setup_hpc.sh"
echo "4. Submit jobs: sbatch slurm_scripts/run_hyperparameter_tuning.slurm" 