#!/bin/bash
# Check if EU-Emotion transfer is still running and making progress

echo "=========================================="
echo "EU-Emotion Transfer Status Check"
echo "=========================================="
echo ""

# Check if rsync process is running
if ps aux | grep -E "rsync.*EU_emotions" | grep -v grep > /dev/null; then
    echo "✅ rsync process is RUNNING"
    ps aux | grep -E "rsync.*EU_emotions" | grep -v grep | head -1
else
    echo "❌ rsync process NOT found - transfer may have stopped"
fi

echo ""
echo "To check progress on HPC, run:"
echo "  ssh eb2007@login-cpu.hpc.cam.ac.uk 'du -sh ~/data/EU_emotions'"
echo ""
echo "If the size increases over time, transfer is working!"
echo ""
echo "To see live progress, check the rsync terminal window"
echo "or press Enter in that window to see if it updates"





