#!/bin/bash
# Transfer trial definitions to HPC
# Single rsync command - only one authentication needed

HPC_USER="eb2007"
HPC_HOST="login.hpc.cam.ac.uk"
HPC_PROJECT_DIR="/home/eb2007/mr_ts_play"

LOCAL_PROJECT_PATH="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "Transferring trial definitions to HPC..."
echo "Note: You'll need to authenticate once (password + TOTP)"
echo ""

# Single rsync command - creates directories automatically
# Only ONE authentication prompt
rsync -avz --progress \
    "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_train.json" \
    "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_val.json" \
    $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/data/trial_definitions/

echo ""
echo "Transfer complete!"
echo "Verify on HPC: ls -la ~/mr_ts_play/data/trial_definitions/eu_emotion_*.json"
