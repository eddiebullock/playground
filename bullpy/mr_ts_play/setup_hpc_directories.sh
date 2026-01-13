#!/bin/bash
# Create necessary directories on HPC for video fine-tuning

HPC_USER="eb2007"
HPC_HOST="login.hpc.cam.ac.uk"

echo "Creating directories on HPC..."

ssh $HPC_USER@$HPC_HOST << 'EOF'
mkdir -p ~/mr_ts_play/experiments/eu_emotion_model_comparison/training
mkdir -p ~/mr_ts_play/experiments/eu_emotion_model_comparison/models
mkdir -p ~/mr_ts_play/data/trial_definitions

echo "Directories created:"
ls -la ~/mr_ts_play/experiments/eu_emotion_model_comparison/
EOF

echo "Done!"
