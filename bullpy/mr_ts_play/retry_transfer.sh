#!/bin/bash
# Retry transfer - will only transfer missing/failed files

echo "=========================================="
echo "Retrying CAM Dataset Transfer"
echo "=========================================="
echo "This will only transfer files that failed or are missing"
echo "Files already transferred will be skipped"
echo "=========================================="
echo ""

rsync -avz --progress --partial \
    "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions/" \
    eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/CAM/

echo ""
echo "=========================================="
echo "Retry Transfer Complete!"
echo "=========================================="
echo ""
echo "Check for any remaining errors above."
echo "If there are still failures, they may be due to:"
echo "  - Network issues"
echo "  - File permissions"
echo "  - Corrupted source files"
echo ""







