#!/bin/bash
# Investigate why HPC storage decreased

echo "=========================================="
echo "Investigating HPC Storage Decrease"
echo "=========================================="
echo ""

echo "1. Current size on HPC:"
ssh eb2007@login-cpu.hpc.cam.ac.uk "du -sh ~/data/EU_emotions 2>/dev/null"

echo ""
echo "2. File count:"
ssh eb2007@login-cpu.hpc.cam.ac.uk "find ~/data/EU_emotions -type f 2>/dev/null | wc -l"

echo ""
echo "3. Files present (first 20):"
ssh eb2007@login-cpu.hpc.cam.ac.uk "ls -lh ~/data/EU_emotions 2>/dev/null | head -20"

echo ""
echo "4. Check for partial/incomplete files:"
ssh eb2007@login-cpu.hpc.cam.ac.uk "find ~/data/EU_emotions -type f -name '*.partial' -o -name '*.tmp' 2>/dev/null"

echo ""
echo "5. Check rsync process status:"
ps aux | grep -E "rsync.*EU_emotions" | grep -v grep || echo "No rsync process found"

echo ""
echo "=========================================="
echo "Possible causes:"
echo "=========================================="
echo "1. rsync --inplace might have issues with large files"
echo "2. Transfer was restarted and cleaned up partial files"
echo "3. HPC filesystem issue"
echo ""
echo "Recommendation: Check if transfer is still working"
echo "  Wait 5 minutes, then check size again:"
echo "  ssh eb2007@login-cpu.hpc.cam.ac.uk 'du -sh ~/data/EU_emotions'"









