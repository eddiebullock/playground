#!/bin/bash
# Check what files are actually on HPC

echo "Checking files on HPC..."
echo ""

ssh eb2007@login-cpu.hpc.cam.ac.uk << 'EOF'
echo "Total size:"
du -sh ~/data/EU_emotions 2>/dev/null || echo "Directory not found"

echo ""
echo "File count:"
find ~/data/EU_emotions -type f 2>/dev/null | wc -l

echo ""
echo "Files present:"
ls -lh ~/data/EU_emotions 2>/dev/null | head -20

echo ""
echo "Largest files:"
find ~/data/EU_emotions -type f -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -10
EOF






