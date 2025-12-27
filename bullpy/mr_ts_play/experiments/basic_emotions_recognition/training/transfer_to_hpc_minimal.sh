#!/bin/bash
# Minimal transfer: Only what's needed for HPC training
# This transfers only the essential training scripts, not the entire experiment
# Uses a single SSH session to minimize authentication prompts

set -e

HPC_HOST="eb2007@login-cpu.hpc.cam.ac.uk"
LOCAL_PROJECT_ROOT="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "============================================================"
echo "Minimal Transfer: Basic Emotions Training Scripts to HPC"
echo "============================================================"
echo ""
echo "Transferring only essential files for HPC training:"
echo "  - Training scripts"
echo "  - HPC shell scripts"
echo "  - SLURM scripts"
echo "  - Mapping files"
echo ""
echo "Note: You'll be asked for password/TOTP once for rsync, then once for final setup"
echo ""

# Create directory structure and transfer files in one go
echo "Step 1: Creating directories and transferring all files..."
rsync -avz --progress --partial \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/finetune_basic_emotions.py" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/evaluate_basic_emotions.py" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.sh" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/hpc_basic_emotions_eu_emotion.sh" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.slurm" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/training/hpc_basic_emotions_eu_emotion.slurm" \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json" \
    "${HPC_HOST}:~/mr_ts_play/experiments/basic_emotions_recognition/" \
    --include='*/' \
    --include='training/*' \
    --include='data/basic_emotion_mappings/*' \
    --exclude='*'

# Single SSH session to do all setup tasks
echo ""
echo "Step 2: Setting up directories and permissions (one SSH session)..."
ssh $HPC_HOST bash << 'ENDSSH'
    set -e
    cd ~/mr_ts_play
    
    # Create directories if needed
    mkdir -p experiments/basic_emotions_recognition/training
    mkdir -p experiments/basic_emotions_recognition/data/basic_emotion_mappings
    
    # Move files to correct locations if rsync put them in wrong place
    if [ -f experiments/basic_emotions_recognition/create_basic_emotion_trials.py ]; then
        mv experiments/basic_emotions_recognition/*.py experiments/basic_emotions_recognition/training/ 2>/dev/null || true
        mv experiments/basic_emotions_recognition/*.sh experiments/basic_emotions_recognition/training/ 2>/dev/null || true
        mv experiments/basic_emotions_recognition/*.slurm experiments/basic_emotions_recognition/training/ 2>/dev/null || true
    fi
    
    if [ -f experiments/basic_emotions_recognition/eu_emotion_basic_mapping.json ]; then
        mkdir -p experiments/basic_emotions_recognition/data/basic_emotion_mappings
        mv experiments/basic_emotions_recognition/eu_emotion_basic_mapping.json \
           experiments/basic_emotions_recognition/data/basic_emotion_mappings/ 2>/dev/null || true
    fi
    
    # Make scripts executable
    chmod +x experiments/basic_emotions_recognition/training/*.sh 2>/dev/null || true
    chmod +x experiments/basic_emotions_recognition/training/*.py 2>/dev/null || true
    
    # Verify CAM mapping exists (should already be there)
    if [ ! -f data/basic_emotion_mapping.json ]; then
        echo "⚠️  Warning: CAM mapping file not found at data/basic_emotion_mapping.json"
        echo "   You may need to transfer it separately"
    else
        echo "✅ CAM mapping file found"
    fi
    
    echo "✅ Setup complete!"
    echo ""
    echo "Files transferred:"
    ls -lh experiments/basic_emotions_recognition/training/*.py 2>/dev/null | wc -l | xargs echo "  Python scripts:"
    ls -lh experiments/basic_emotions_recognition/training/*.sh 2>/dev/null | wc -l | xargs echo "  Shell scripts:"
    ls -lh experiments/basic_emotions_recognition/training/*.slurm 2>/dev/null | wc -l | xargs echo "  SLURM scripts:"
    ls -lh experiments/basic_emotions_recognition/data/basic_emotion_mappings/*.json 2>/dev/null | wc -l | xargs echo "  Mapping files:"
ENDSSH

echo ""
echo "✅ Minimal transfer complete!"
echo ""
echo "Next steps on HPC:"
echo "1. SSH to HPC: ssh eb2007@login-cpu.hpc.cam.ac.uk"
echo "2. Verify: ls -la ~/mr_ts_play/experiments/basic_emotions_recognition/training/"
echo "3. Generate trials: python experiments/basic_emotions_recognition/training/create_basic_emotion_trials.py ..."
echo "4. Submit jobs: sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.slurm"
echo ""
