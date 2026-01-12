#!/bin/bash
# Transfer EU-Emotion dataset to HPC

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions"

echo "=========================================="
echo "Transferring EU-Emotion Dataset to HPC"
echo "=========================================="
echo ""
echo "Source: $SOURCE"
echo "Destination: $HPC_DEST"
echo ""
echo "This will COPY files (originals will remain on your laptop)"
echo ""

# Create destination directory on HPC first
echo "Creating destination directory on HPC..."
ssh eb2007@login-cpu.hpc.cam.ac.uk "mkdir -p /home/eb2007/data/EU_emotions"

echo ""
echo "Starting transfer..."
echo "This may take a while (213GB to transfer)..."
echo ""

# Transfer with progress
rsync -avh --progress --partial \
  "$SOURCE/" \
  "$HPC_DEST/"

echo ""
echo "=========================================="
echo "Transfer Complete!"
echo "=========================================="
echo ""
echo "To verify on HPC, run:"
echo "  ssh eb2007@login-cpu.hpc.cam.ac.uk"
echo "  du -sh ~/data/EU_emotions"
echo "  find ~/data/EU_emotions -type f | wc -l"









