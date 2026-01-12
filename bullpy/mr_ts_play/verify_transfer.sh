#!/bin/bash
# Verify CAM dataset transfer to HPC

LOCAL_PATH="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
HPC_PATH="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/CAM"

echo "=========================================="
echo "Verifying CAM Dataset Transfer"
echo "=========================================="
echo ""

echo "Local dataset:"
echo "  Path: $LOCAL_PATH"
echo ""

# Count files locally
echo "Counting local files..."
LOCAL_COUNT=$(find "$LOCAL_PATH" -type f | wc -l | tr -d ' ')
LOCAL_SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)

echo "  Files: $LOCAL_COUNT"
echo "  Size: $LOCAL_SIZE"
echo ""

echo "HPC dataset:"
echo "  Path: $HPC_PATH"
echo ""

# Count files on HPC
echo "Counting HPC files..."
HPC_COUNT=$(ssh eb2007@login-cpu.hpc.cam.ac.uk "find /home/eb2007/data/CAM -type f | wc -l" | tr -d ' ')
HPC_SIZE=$(ssh eb2007@login-cpu.hpc.cam.ac.uk "du -sh /home/eb2007/data/CAM | cut -f1")

echo "  Files: $HPC_COUNT"
echo "  Size: $HPC_SIZE"
echo ""

echo "=========================================="
echo "Comparison"
echo "=========================================="
echo "Local files:  $LOCAL_COUNT"
echo "HPC files:    $HPC_COUNT"
echo ""

if [ "$LOCAL_COUNT" -eq "$HPC_COUNT" ]; then
    echo "✅ SUCCESS: File counts match!"
else
    DIFF=$((LOCAL_COUNT - HPC_COUNT))
    echo "⚠️  WARNING: File count mismatch"
    echo "   Missing: $DIFF files"
    echo "   You may need to retry the transfer"
fi

echo ""
echo "=========================================="
echo "To check on HPC directly, run:"
echo "  ssh eb2007@login-cpu.hpc.cam.ac.uk"
echo "  find ~/data/CAM -type f | wc -l"
echo "  du -sh ~/data/CAM"
echo "=========================================="









