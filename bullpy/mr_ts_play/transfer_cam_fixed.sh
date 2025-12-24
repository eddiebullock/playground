#!/bin/bash
# Fixed transfer script for CAM dataset

# Properly quoted paths
LOCAL_PATH="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/CAM"

echo "=========================================="
echo "Transferring CAM Dataset to HPC"
echo "=========================================="
echo "Source: $LOCAL_PATH"
echo "Destination: $HPC_DEST"
echo ""
echo "Note: You'll need to enter your password and TOTP code"
echo "=========================================="
echo ""

# Transfer with proper quoting and resume capability
rsync -avz --progress --partial \
    "$LOCAL_PATH/" \
    "$HPC_DEST/"

echo ""
echo "Transfer complete!"







