#!/bin/bash
# Monitor EU-Emotion transfer progress

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
DEST="/Volumes/LaCie/EU_emotions"

echo "=========================================="
echo "EU-Emotion Transfer Monitor"
echo "=========================================="
echo ""

while true; do
    if [ -d "$DEST" ]; then
        DEST_SIZE=$(du -sh "$DEST" 2>/dev/null | cut -f1)
        DEST_BYTES=$(du -sb "$DEST" 2>/dev/null | cut -f1)
    else
        DEST_SIZE="0"
        DEST_BYTES=0
    fi
    
    if [ -d "$SOURCE" ]; then
        SOURCE_SIZE=$(du -sh "$SOURCE" 2>/dev/null | cut -f1)
        SOURCE_BYTES=$(du -sb "$SOURCE" 2>/dev/null | cut -f1)
    else
        SOURCE_SIZE="0"
        SOURCE_BYTES=0
    fi
    
    # Check if rsync is still running
    if pgrep -f "rsync.*EU_emotions" > /dev/null; then
        STATUS="🔄 Transferring..."
    else
        STATUS="✅ Transfer complete (or stopped)"
    fi
    
    # Calculate percentage if we have both sizes
    if [ "$SOURCE_BYTES" -gt 0 ] && [ "$DEST_BYTES" -gt 0 ]; then
        PERCENT=$((DEST_BYTES * 100 / SOURCE_BYTES))
        if [ "$PERCENT" -gt 100 ]; then
            PERCENT=100
        fi
        echo -ne "\r$STATUS | Destination: $DEST_SIZE | Source remaining: $SOURCE_SIZE | Progress: ${PERCENT}%"
    else
        echo -ne "\r$STATUS | Destination: $DEST_SIZE | Source: $SOURCE_SIZE"
    fi
    
    # If source is gone or very small, transfer is likely complete
    if [ ! -d "$SOURCE" ] || [ "$SOURCE_BYTES" -lt 1000000 ]; then
        echo ""
        echo ""
        echo "✅ Transfer appears complete!"
        echo "   Final size on LaCie: $DEST_SIZE"
        break
    fi
    
    sleep 5
done









