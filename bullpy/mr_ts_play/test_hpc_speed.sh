#!/bin/bash
# Test network speed to HPC

echo "Testing network speed to HPC..."
echo "Creating a 10MB test file..."

# Create a test file
dd if=/dev/zero of=/tmp/hpc_speed_test.dat bs=1M count=10 2>/dev/null

echo "Uploading 10MB test file to HPC..."
time scp /tmp/hpc_speed_test.dat eb2007@login-cpu.hpc.cam.ac.uk:/tmp/hpc_speed_test.dat

echo ""
echo "Cleaning up..."
rm /tmp/hpc_speed_test.dat
ssh eb2007@login-cpu.hpc.cam.ac.uk "rm /tmp/hpc_speed_test.dat" 2>/dev/null

echo ""
echo "Speed test complete. Check the time above to estimate transfer speed."







