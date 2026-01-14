#!/bin/bash
# Transfer EU Emotion video fine-tuning code to CSD3 (HPC)
#
# Goal: ONE authentication prompt (password + TOTP) by using a SINGLE rsync call.
# Note: If you do NOT have SSH keys set up, rsync will still prompt once.

set -euo pipefail

HPC_USER="eb2007"
HPC_HOST="login.hpc.cam.ac.uk"
HPC_PROJECT_DIR="/home/eb2007/mr_ts_play"

LOCAL_PROJECT_PATH="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "=========================================="
echo "Transferring video fine-tuning code to HPC"
echo "=========================================="
echo "Source:      $LOCAL_PROJECT_PATH"
echo "Destination: $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR"
echo ""
echo "This uses ONE rsync command => you should authenticate ONCE (password + TOTP)."
echo ""

# Transfer the whole experiment folder (small) + trial defs, excluding caches.
rsync -avz --progress \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$LOCAL_PROJECT_PATH/experiments/eu_emotion_model_comparison/" \
  "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_train.json" \
  "$LOCAL_PROJECT_PATH/data/trial_definitions/eu_emotion_val.json" \
  $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/

echo ""
echo "✅ Transfer complete."
echo "On HPC, verify:"
echo "  ls -la ~/mr_ts_play/experiments/eu_emotion_model_comparison/training/"
echo "  ls -la ~/mr_ts_play/data/trial_definitions/eu_emotion_{train,val}.json"

