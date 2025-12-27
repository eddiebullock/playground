#!/bin/bash
# Transfer CAM dataset and project files to HPC

HPC_USER="eb2007"
HPC_HOST="login-cpu.hpc.cam.ac.uk"
HPC_CAM_DIR="/home/eb2007/data/CAM"
HPC_PROJECT_DIR="/home/eb2007/mr_ts_play"

LOCAL_CAM_PATH="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
LOCAL_PROJECT_PATH="/Users/eb2007/playground/bullpy/mr_ts_play"

echo "=========================================="
echo "Transferring CAM Dataset to HPC"
echo "=========================================="
echo "Source: $LOCAL_CAM_PATH"
echo "Destination: $HPC_USER@$HPC_HOST:$HPC_CAM_DIR"
echo ""
echo "This may take 30-60 minutes depending on dataset size..."
echo "=========================================="
echo ""

# Transfer CAM dataset
rsync -avz --progress \
    "$LOCAL_CAM_PATH/" \
    $HPC_USER@$HPC_HOST:$HPC_CAM_DIR/

echo ""
echo "=========================================="
echo "CAM Dataset Transfer Complete!"
echo "=========================================="
echo ""
echo "Next: Transfer project code with:"
echo "  ./transfer_project_to_hpc.sh"
echo ""








