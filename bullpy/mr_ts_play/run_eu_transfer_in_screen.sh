#!/bin/bash
# Run EU-Emotion transfer in a screen session so it continues in background

SOURCE="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
HPC_DEST="eb2007@login-cpu.hpc.cam.ac.uk:/home/eb2007/data/EU_emotions"
SCREEN_NAME="eu_emotion_transfer"

echo "=========================================="
echo "EU-Emotion Transfer in Screen Session"
echo "=========================================="
echo ""
echo "This will start the transfer in a screen session named: $SCREEN_NAME"
echo "You can detach with: Ctrl+A then D"
echo "Reattach with: screen -r $SCREEN_NAME"
echo ""

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "⚠️  'screen' not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install screen
    else
        echo "❌ Please install screen: brew install screen"
        exit 1
    fi
fi

# Create destination directory on HPC
echo "Creating destination directory on HPC..."
ssh eb2007@login-cpu.hpc.cam.ac.uk "mkdir -p /home/eb2007/data/EU_emotions"

echo ""
echo "Starting transfer in screen session..."
echo ""

# Start screen session with transfer command
screen -dmS "$SCREEN_NAME" bash -c "
    echo '==========================================';
    echo 'EU-Emotion Transfer Started';
    echo '==========================================';
    echo '';
    echo 'Started: $(date)';
    echo 'Source: $SOURCE';
    echo 'Destination: $HPC_DEST';
    echo '';
    echo 'To detach: Ctrl+A then D';
    echo 'To reattach: screen -r $SCREEN_NAME';
    echo '';
    echo 'Starting rsync...';
    echo '';
    rsync -avh --progress --partial \
      \"$SOURCE/\" \
      \"$HPC_DEST/\";
    echo '';
    echo '==========================================';
    echo 'Transfer Complete!';
    echo 'Finished: $(date)';
    echo '==========================================';
    echo '';
    echo 'Press Enter to close this window...';
    read
"

echo "✅ Transfer started in screen session: $SCREEN_NAME"
echo ""
echo "Commands:"
echo "  Detach:     screen -d $SCREEN_NAME"
echo "  Reattach:   screen -r $SCREEN_NAME"
echo "  List:       screen -ls"
echo "  Kill:       screen -X -S $SCREEN_NAME quit"
echo ""
echo "To check progress on HPC (in another terminal):"
echo "  ssh eb2007@login-cpu.hpc.cam.ac.uk 'du -sh ~/data/EU_emotions'"





