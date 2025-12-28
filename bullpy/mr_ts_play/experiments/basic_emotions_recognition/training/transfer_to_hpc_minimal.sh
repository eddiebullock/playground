#!/bin/bash
# Single-operation transfer: All files in one rsync + one SSH session
# This minimizes authentication prompts to just 2 (one for rsync, one for SSH)

set -e

HPC_HOST="eb2007@login-cpu.hpc.cam.ac.uk"
LOCAL_PROJECT_ROOT="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "============================================================"
echo "Transferring Basic Emotions Training Scripts to HPC"
echo "============================================================"
echo ""
echo "This will transfer all necessary files in ONE operation"
echo "You'll be asked for password/TOTP twice (rsync + SSH)"
echo ""

# Single rsync command to transfer all files
echo "Transferring files..."
rsync -avz --progress --partial \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/data/" \
    "${HPC_HOST}:~/mr_ts_play/experiments/basic_emotions_recognition/" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store'

# Single SSH session to do all setup
echo ""
echo "Setting up directories and permissions..."
ssh $HPC_HOST bash << 'ENDSSH'
    set -e
    cd ~/mr_ts_play
    
    # Ensure directories exist
    mkdir -p experiments/basic_emotions_recognition/training
    mkdir -p experiments/basic_emotions_recognition/data/basic_emotion_mappings
    
    # Make scripts executable
    chmod +x experiments/basic_emotions_recognition/training/*.sh 2>/dev/null || true
    chmod +x experiments/basic_emotions_recognition/training/*.py 2>/dev/null || true
    
    # Verify key files exist
    echo ""
    echo "✅ Transferred files:"
    ls -1 experiments/basic_emotions_recognition/training/*.py 2>/dev/null | wc -l | xargs echo "  Python scripts:"
    ls -1 experiments/basic_emotions_recognition/training/*.sh 2>/dev/null | wc -l | xargs echo "  Shell scripts:"
    ls -1 experiments/basic_emotions_recognition/training/*.slurm 2>/dev/null | wc -l | xargs echo "  SLURM scripts:"
    ls -1 experiments/basic_emotions_recognition/data/basic_emotion_mappings/*.json 2>/dev/null | wc -l | xargs echo "  Mapping files:"
    
    echo ""
    echo "✅ Setup complete!"
ENDSSH

echo ""
echo "✅ Transfer complete!"
echo ""
echo "Next steps on HPC:"
echo "1. SSH: ssh $HPC_HOST"
echo "2. Clean old trials: rm ~/rds/.../basic_emotions_*/*.json"
echo "3. Submit jobs: sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.slurm"
echo ""
