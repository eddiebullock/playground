#!/bin/bash
# Monitor transfer progress in real-time

echo "Monitoring EU-Emotion transfer progress..."
echo "Press Ctrl+C to stop"
echo ""

PREV_SIZE=0
PREV_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    CURRENT_SIZE=$(ssh eb2007@login-cpu.hpc.cam.ac.uk "du -sb ~/data/EU_emotions 2>/dev/null" 2>/dev/null | awk '{print $1}')
    
    if [ -z "$CURRENT_SIZE" ] || [ "$CURRENT_SIZE" = "0" ]; then
        echo "Waiting for transfer to start..."
        sleep 10
        continue
    fi
    
    CURRENT_SIZE_MB=$((CURRENT_SIZE / 1024 / 1024))
    CURRENT_SIZE_GB=$(echo "scale=2; $CURRENT_SIZE_MB / 1024" | bc)
    
    if [ "$PREV_SIZE" -gt 0 ]; then
        ELAPSED=$((CURRENT_TIME - PREV_TIME))
        if [ $ELAPSED -gt 0 ]; then
            SIZE_DIFF=$((CURRENT_SIZE - PREV_SIZE))
            SIZE_DIFF_MB=$((SIZE_DIFF / 1024 / 1024))
            SPEED_MB=$(echo "scale=2; $SIZE_DIFF_MB / $ELAPSED" | bc)
            
            # Calculate progress
            TOTAL_MB=218112  # 213GB in MB
            PROGRESS=$(echo "scale=1; $CURRENT_SIZE_MB * 100 / $TOTAL_MB" | bc)
            REMAINING_MB=$((TOTAL_MB - CURRENT_SIZE_MB))
            if [ $(echo "$SPEED_MB > 0" | bc) -eq 1 ]; then
                REMAINING_SEC=$((REMAINING_MB * 1024 / $(echo "$SPEED_MB * 1024" | bc)))
                REMAINING_HOURS=$(echo "scale=1; $REMAINING_SEC / 3600" | bc)
            else
                REMAINING_HOURS="calculating..."
            fi
            
            echo -ne "\rSize: ${CURRENT_SIZE_GB}GB (${PROGRESS}%) | Speed: ${SPEED_MB} MB/s | Remaining: ~${REMAINING_HOURS} hours"
        fi
    else
        echo -ne "\rSize: ${CURRENT_SIZE_GB}GB | Calculating speed..."
    fi
    
    PREV_SIZE=$CURRENT_SIZE
    PREV_TIME=$CURRENT_TIME
    
    sleep 30  # Check every 30 seconds
done









