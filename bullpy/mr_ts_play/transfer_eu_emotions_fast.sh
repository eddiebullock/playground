#!/bin/bash
# Fast EU-Emotion transfer with all optimizations

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions"

echo "=========================================="
echo "Fast EU-Emotion Transfer (Optimized)"
echo "=========================================="
echo ""
echo "Your upload speed: ~18.73 Mbps (~2.3 MB/s)"
echo "Expected transfer time: ~25 hours (not 8 days!)"
echo ""
echo "Optimizations applied:"
echo "  --inplace      : Faster for large files (writes in place)"
echo "  --bwlimit=0    : No bandwidth limit"
echo "  --partial      : Resume if interrupted"
echo "  --whole-file   : Transfer whole files (faster for large files)"
echo ""

# Create destination directory
ssh eb2007@login-cpu.hpc.cam.ac.uk "mkdir -p /home/eb2007/data/EU_emotions"

echo "Starting optimized transfer..."
echo "Press Ctrl+C to cancel"
echo ""

# Optimized rsync - --whole-file is faster for large files over network
# --inplace writes directly to destination (faster)
rsync -avh --progress --partial --inplace --whole-file --bwlimit=0 \
  "$SOURCE/" \
  "$HPC_DEST/"

echo ""
echo "=========================================="
echo "Transfer Complete!"
echo "=========================================="

