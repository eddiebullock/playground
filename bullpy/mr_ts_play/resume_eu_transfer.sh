#!/bin/bash
# Resume EU-Emotion transfer (handles connection drops)

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions"

echo "=========================================="
echo "Resuming EU-Emotion Transfer"
echo "=========================================="
echo ""
echo "Using --partial flag: will resume from where it left off"
echo "If connection drops, just run this script again"
echo ""

# Keep retrying if connection drops
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    echo "Starting transfer..."
    echo ""
    
    # Run rsync with retry logic
    rsync -avh --progress --partial --bwlimit=0 \
      "$SOURCE/" \
      "$HPC_DEST/"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Transfer completed successfully!"
        break
    elif [ $EXIT_CODE -eq 12 ] || [ $EXIT_CODE -eq 255 ]; then
        # Connection error - retry
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo ""
        echo "⚠️  Connection interrupted (exit code: $EXIT_CODE)"
        echo "Waiting 10 seconds before retry..."
        sleep 10
        echo ""
    else
        echo ""
        echo "❌ Transfer failed with exit code: $EXIT_CODE"
        echo "Check the error message above"
        break
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo ""
    echo "⚠️  Reached maximum retries. Transfer may need manual intervention."
fi


