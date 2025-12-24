#!/bin/bash
# Check if OneDrive files are actually synced locally

FOLDER="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"

echo "=========================================="
echo "OneDrive Sync Status Check"
echo "=========================================="
echo ""

# Test reading a few files
echo "Testing file accessibility..."
TEST_FILES=(
    "emotions-20251220T073354Z-3-001.zip"
    "emotions-20251220T073354Z-3-002.zip"
    "EU_Emotion_Stimulus_Set.zip-001.006"
)

LOCAL_COUNT=0
CLOUD_COUNT=0

for file in "${TEST_FILES[@]}"; do
    filepath="$FOLDER/$file"
    if [ -f "$filepath" ]; then
        # Try to read first 1KB
        if head -c 1024 "$filepath" > /dev/null 2>&1; then
            echo "  ✅ $file - Accessible locally"
            ((LOCAL_COUNT++))
        else
            echo "  ❌ $file - Cloud-only (needs download)"
            ((CLOUD_COUNT++))
        fi
    else
        echo "  ⚠️  $file - Not found"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Local files: $LOCAL_COUNT"
echo "Cloud-only files: $CLOUD_COUNT"
echo ""

if [ $CLOUD_COUNT -gt 0 ]; then
    echo "⚠️  Some files are still cloud-only!"
    echo ""
    echo "To fix:"
    echo "1. Open Finder"
    echo "2. Navigate to: $FOLDER"
    echo "3. Right-click folder → 'Always Keep on This Device'"
    echo "4. Wait for OneDrive to finish syncing (check menu bar icon)"
    echo ""
    echo "Or force download all files:"
    echo "  cd '$FOLDER'"
    echo "  for f in *.zip; do head -c 1 \"\$f\" > /dev/null; done"
else
    echo "✅ All test files are local!"
    echo ""
    echo "If transfer is still slow, it's likely network bandwidth to HPC."
    echo "Current speed: ~500 kB/s = ~4 hours per GB"
    echo "For 213GB: ~850 hours = ~35 days"
    echo ""
    echo "Consider:"
    echo "- Let it run in background (screen/tmux)"
    echo "- Check network connection"
    echo "- Transfer during off-peak hours"
fi






