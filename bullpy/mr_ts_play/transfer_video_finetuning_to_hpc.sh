#!/bin/bash
# Transfer video fine-tuning code to HPC
# Quick transfer of just the new video model fine-tuning files
# Uses single rsync command to minimize authentication prompts

HPC_USER="eb2007"
HPC_HOST="login.hpc.cam.ac.uk"
HPC_PROJECT_DIR="/home/eb2007/mr_ts_play"

LOCAL_PROJECT_PATH="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "=========================================="
echo "Transferring Video Fine-Tuning Code to HPC"
echo "=========================================="
echo "Source: $LOCAL_PROJECT_PATH"
echo "Destination: $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR"
echo "=========================================="
echo ""
echo "Note: You'll need to authenticate once (password + TOTP)"
echo ""

# Single rsync command - transfers entire directory structure in one go
# This requires only ONE authentication (password + TOTP)
rsync -avz --progress \
    "$LOCAL_PROJECT_PATH/experiments/eu_emotion_model_comparison/" \
    $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/experiments/eu_emotion_model_comparison/

# Also transfer the quick check script
rsync -avz --progress \
    "$LOCAL_PROJECT_PATH/experiments/eu_emotion_model_comparison/training/quick_check.sh" \
    $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/experiments/eu_emotion_model_comparison/training/ 2>/dev/null || \
rsync -avz --progress \
    "$LOCAL_PROJECT_PATH/experiments/eu_emotion_model_comparison/training/HPC_PREFLIGHT_CHECKLIST.md" \
    $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/experiments/eu_emotion_model_comparison/training/

# Also transfer trial definitions if they exist locally
if [ -f "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_train.json" ]; then
    echo ""
    echo "Transferring trial definitions..."
    rsync -avz --progress \
        "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_train.json" \
        "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_val.json" \
        $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/data/trial_definitions/ 2>/dev/null || \
    echo "Note: Trial definitions transfer failed (may need to create data/trial_definitions/ on HPC first)"
fi

echo ""
echo "=========================================="
echo "Video Fine-Tuning Code Transfer Complete!"
echo "=========================================="
echo ""
echo "Next steps on HPC:"
echo "1. Verify trial definitions: ls -la data/trial_definitions/eu_emotion_*.json"
echo "2. If missing, transfer manually or create them on HPC"
echo "3. Submit job: sbatch experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.slurm"
echo ""
