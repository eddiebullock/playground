#!/bin/bash
# Diagnose why transfer to HPC is so slow

echo "=========================================="
echo "Network Transfer Diagnostics"
echo "=========================================="
echo ""

echo "1. Checking network interface..."
ifconfig | grep -A 5 "inet " | grep -E "inet |status|media" | head -10
echo ""

echo "2. Checking for VPN/Proxy..."
if ps aux | grep -i "vpn\|tunnel\|proxy" | grep -v grep > /dev/null; then
    echo "  ⚠️  VPN/Proxy detected:"
    ps aux | grep -i "vpn\|tunnel\|proxy" | grep -v grep | head -3
else
    echo "  ✅ No VPN/Proxy detected"
fi
echo ""

echo "3. Testing latency to HPC..."
ping -c 3 login-cpu.hpc.cam.ac.uk 2>&1 | tail -3
echo ""

echo "4. Checking rsync options..."
echo "  Current rsync flags: -avh --progress --partial"
echo "  Missing optimization flags that could help:"
echo "    --bwlimit=0 (remove bandwidth limit)"
echo "    --compress (enable compression - but ZIPs already compressed)"
echo "    --inplace (faster for large files)"
echo ""

echo "5. Testing with optimized rsync..."
echo "  Try this command instead:"
echo ""
echo "  rsync -avh --progress --partial --inplace --bwlimit=0 \\"
echo "    \"/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/\" \\"
echo "    eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions/"
echo ""

echo "6. Alternative: Use compression (may help if network is bottleneck):"
echo "  rsync -avhz --progress --partial --inplace --bwlimit=0 \\"
echo "    \"/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/\" \\"
echo "    eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions/"
echo "  (Note: -z enables compression, but ZIPs are already compressed)"
echo ""

echo "7. Check if you're on WiFi vs Ethernet:"
ifconfig | grep -E "status: active|media:" | head -5
echo ""

echo "=========================================="
echo "Recommendations:"
echo "=========================================="
echo "1. If on WiFi, try Ethernet cable (much faster)"
echo "2. Check if university network has transfer limits"
echo "3. Try --inplace flag (faster for large files)"
echo "4. Check if you're behind a proxy/firewall"
echo "5. Consider transferring from a different network (home vs university)"
echo ""








