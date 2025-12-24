#!/bin/bash
# Transfer project code to HPC

HPC_USER="eb2007"
HPC_HOST="login-cpu.hpc.cam.ac.uk"
HPC_PROJECT_DIR="/home/eb2007/mr_ts_play"

LOCAL_PROJECT_PATH="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "=========================================="
echo "Transferring Project Code to HPC"
echo "=========================================="
echo "Source: $LOCAL_PROJECT_PATH"
echo "Destination: $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR"
echo "=========================================="
echo ""

# Transfer project code (excluding large/unnecessary files)
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'models/' \
    --exclude 'results/' \
    --exclude '*.ipynb_checkpoints' \
    --exclude '.DS_Store' \
    "$LOCAL_PROJECT_PATH/" \
    $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/

echo ""
echo "=========================================="
echo "Project Code Transfer Complete!"
echo "=========================================="







