#!/bin/bash
# Transfer basic_emotions_recognition experiment to HPC
# This transfers the experiment code to HPC (not data - data is already there or transferred separately)

set -e

HPC_HOST="eb2007@login-cpu.hpc.cam.ac.uk"

# Project root (local)
LOCAL_PROJECT_ROOT="/Users/eb2007/playground/bullpy/mr_ts_play"

# HPC project root
HPC_PROJECT_ROOT="${HOME}/mr_ts_play"

echo "============================================================"
echo "Transferring Basic Emotions Experiment to HPC"
echo "============================================================"
echo ""
echo "Local: $LOCAL_PROJECT_ROOT"
echo "HPC: $HPC_PROJECT_ROOT"
echo ""

# Check if local experiment exists
if [ ! -d "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition" ]; then
    echo "❌ Error: Local experiment not found at $LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition"
    exit 1
fi

echo "Step 1: Transferring experiment code..."
echo "This will transfer the basic_emotions_recognition experiment directory"
echo ""

# Transfer the experiment directory with retry logic
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES..."
    
    if rsync -avz --progress --partial \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        --exclude='*.log' \
        --exclude='data/llm_cache/*' \
        --exclude='data/trial_definitions/*.json' \
        "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/" \
        "${HPC_HOST}:${HPC_PROJECT_ROOT}/experiments/basic_emotions_recognition/"; then
        echo "✅ Transfer successful!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "⚠️  Transfer failed. Retrying in 5 seconds..."
            sleep 5
        else
            echo "❌ Transfer failed after $MAX_RETRIES attempts."
            echo "   This is often due to network issues or TOTP timeout."
            echo "   Try running the command again manually."
            exit 1
        fi
    fi
done

echo ""
echo "Step 2: Transferring required data files..."
echo ""

# Transfer the basic emotion mapping file (if it doesn't exist on HPC)
echo ""
echo "Step 2a: Transferring basic_emotion_mapping.json..."
rsync -avz --progress --partial \
    "$LOCAL_PROJECT_ROOT/data/basic_emotion_mapping.json" \
    "${HPC_HOST}:${HPC_PROJECT_ROOT}/data/basic_emotion_mapping.json" 2>/dev/null || {
    echo "⚠️  basic_emotion_mapping.json may already exist on HPC (this is OK)"
}

# Transfer EU-Emotion basic mapping (part of experiment)
echo ""
echo "Step 2b: Transferring EU-Emotion basic mapping..."
rsync -avz --progress --partial \
    "$LOCAL_PROJECT_ROOT/experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json" \
    "${HPC_HOST}:${HPC_PROJECT_ROOT}/experiments/basic_emotions_recognition/data/basic_emotion_mappings/eu_emotion_basic_mapping.json" || {
    echo "⚠️  Failed to transfer EU-Emotion mapping. You can transfer it manually later."
}

echo ""
echo "✅ Transfer complete!"
echo ""
echo "Next steps on HPC:"
echo "1. Verify experiment is there: ssh $HPC_HOST 'ls -la ${HPC_PROJECT_ROOT}/experiments/basic_emotions_recognition/'"
echo "2. Check data paths:"
echo "   - CAM data should be at: /home/eb2007/data/CAM"
echo "   - EU-Emotion data should be at: ~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
echo "3. Submit training job: sbatch experiments/basic_emotions_recognition/training/hpc_basic_emotions_cam.slurm"
echo ""

