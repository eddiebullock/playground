#!/bin/bash
# Optimized EU-Emotion transfer to HPC

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions"

echo "=========================================="
echo "Optimized EU-Emotion Transfer"
echo "=========================================="
echo ""
echo "Using optimized rsync flags:"
echo "  --inplace     : Faster for large files (writes in place)"
echo "  --bwlimit=0   : Remove bandwidth limit"
echo "  --partial     : Resume interrupted transfers"
echo "  --progress    : Show progress"
echo ""

# Create destination directory
ssh eb2007@login-cpu.hpc.cam.ac.uk "mkdir -p /home/eb2007/data/EU_emotions"

echo "Starting optimized transfer..."
echo ""

# Optimized rsync command
rsync -avh --progress --partial --inplace --bwlimit=0 \
  "$SOURCE/" \
  "$HPC_DEST/"

echo ""
echo "Transfer complete!"





