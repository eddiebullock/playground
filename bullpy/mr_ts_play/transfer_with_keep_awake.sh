#!/bin/bash
# Transfer EU-Emotion with laptop kept awake

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions"

echo "=========================================="
echo "EU-Emotion Transfer (Laptop Stays Awake)"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Keep your laptop awake (no sleep)"
echo "  2. Run transfer in background"
echo "  3. Allow you to close laptop lid (if on AC power)"
echo ""
echo "To stop: Press Ctrl+C"
echo ""

# Create destination directory
ssh eb2007@login-cpu.hpc.cam.ac.uk "mkdir -p /home/eb2007/data/EU_emotions"

# Run transfer with caffeinate to prevent sleep
# caffeinate will keep system awake while rsync runs
caffeinate -d -i -m -s rsync -avh --progress --partial --inplace --bwlimit=0 \
  "$SOURCE/" \
  "$HPC_DEST/"

echo ""
echo "Transfer complete! Laptop can now sleep normally."







