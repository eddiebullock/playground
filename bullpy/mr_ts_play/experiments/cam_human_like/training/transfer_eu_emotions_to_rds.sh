#!/bin/bash
# Transfer EU Emotions library directly to RDS storage
# This avoids /home quota issues

set -e

echo "============================================================"
echo "Transferring EU Emotions Library to RDS Storage"
echo "ONLY Face and Voice Data (~45GB)"
echo "============================================================"
echo ""

# Local EU emotions path
LOCAL_EU_EMOTIONS="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"

# RDS destination - using the autism-research project RDS
# This is project 90416 (rds-ePtR33Nsgi4) where you created the data folder
# Try different path formats (user created: ~/rds/rds-autism-research-ePtR33Nsgi4/data)
RDS_PROJECT_DEFAULT="${HOME}/rds/rds-autism-research-ePtR33Nsgi4"
RDS_DATA_DIR_DEFAULT="${RDS_PROJECT_DEFAULT}/data"
RDS_EU_EMOTIONS_DEFAULT="${RDS_DATA_DIR_DEFAULT}/EU_emotions"

# Alternative: Try /rds-d7/project/45718 if the above doesn't work
# RDS_PROJECT_ALT="/rds-d7/project/45718"
# RDS_DATA_DIR_ALT="${RDS_PROJECT_ALT}/users/eb2007/data"

# HPC host
HPC_HOST="eb2007@login-cpu.hpc.cam.ac.uk"

# Check if local path exists
if [ ! -d "$LOCAL_EU_EMOTIONS" ]; then
    echo "❌ Error: Local EU emotions directory not found: $LOCAL_EU_EMOTIONS"
    exit 1
fi

echo "Step 1: Checking RDS access on HPC..."
# Check RDS path - user created: ~/rds/rds-autism-research-ePtR33Nsgi4/data
# Try to find the correct path format
RDS_PROJECT=""
RDS_DATA_DIR=""
RDS_EU_EMOTIONS=""

# Check RDS path - need to expand ~ properly for SSH
if ssh $HPC_HOST '[ -d ~/rds/rds-autism-research-ePtR33Nsgi4/data ]' 2>/dev/null; then
    # Use absolute path by getting it from remote
    RDS_DATA_DIR=$(ssh $HPC_HOST 'echo ~/rds/rds-autism-research-ePtR33Nsgi4/data')
    RDS_EU_EMOTIONS="${RDS_DATA_DIR}/EU_emotions"
    echo "✅ Found RDS at: $RDS_DATA_DIR"
elif ssh $HPC_HOST '[ -d /rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data ]' 2>/dev/null; then
    RDS_DATA_DIR="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data"
    RDS_EU_EMOTIONS="/rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"
    echo "✅ Found RDS at: $RDS_DATA_DIR"
else
    # Use default - expand ~ on remote
    RDS_DATA_DIR=$(ssh $HPC_HOST 'echo ~/rds/rds-autism-research-ePtR33Nsgi4/data')
    RDS_EU_EMOTIONS="${RDS_DATA_DIR}/EU_emotions"
    echo "⚠️  Using default RDS path: $RDS_DATA_DIR"
    echo "   Will create if needed"
fi

echo "✅ RDS accessible"
echo ""

# Now display the paths (after they're set)
echo "Source: $LOCAL_EU_EMOTIONS"
echo "Destination: $RDS_EU_EMOTIONS"
echo ""
echo "Note: CAM data is separate at: /home/eb2007/data/CAM"
echo "      EU emotions goes to RDS: $RDS_EU_EMOTIONS"
echo ""

echo "Step 2: Creating RDS data directory..."
# Create the data directory if it doesn't exist
# RDS_DATA_DIR should now be an absolute path
ssh $HPC_HOST "mkdir -p '$RDS_DATA_DIR'" || {
    echo "⚠️  Could not create directory. Checking if it already exists..."
    ssh $HPC_HOST "ls -la '$RDS_DATA_DIR'" || {
        echo "❌ Error: Cannot access or create RDS data directory"
        echo "   Please check on HPC: ls -la ~/rds/rds-autism-research-ePtR33Nsgi4/data"
        exit 1
    }
}
echo "✅ Using RDS directory: $RDS_DATA_DIR"
echo ""

echo "Step 3: Starting transfer to RDS..."
echo "This will transfer ONLY face and voice data (~45GB)"
echo "Excluding: body gestures, social scenes, and other modalities"
echo "Press Ctrl+C to pause/resume later"
echo ""

# Transfer ONLY face and voice data
# Face videos: emotions*/HD Version - Face, Body, Social/Faces - HD Version/
# Voice files: EU Emotion - UK Voices/Original/[EmotionName]/*.mp3

# Use rsync with include/exclude patterns to only transfer what we need
# Note: rsync processes patterns in order, so we need to be careful
rsync -avh --progress --partial \
    --include='emotions*/' \
    --include='emotions*/HD Version - Face, Body, Social/' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/**/' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/**/*.mp4' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/**/*.mov' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/**/' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/**/*.mp4' \
    --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/**/*.mov' \
    --include='EU Emotion - UK Voices/' \
    --include='EU Emotion - UK Voices/Original/' \
    --include='EU Emotion - UK Voices/Original/**/' \
    --include='EU Emotion - UK Voices/Original/**/*.mp3' \
    --include='EU Emotion - UK Voices/Original/**/*.wav' \
    --exclude='emotions*/HD Version - Face, Body, Social/Body*/' \
    --exclude='emotions*/HD Version - Face, Body, Social/Social*/' \
    --exclude='emotions*/HD Version - Face, Body, Social/Scenes*/' \
    --exclude='.DS_Store' \
    --exclude='*.tmp' \
    --exclude='*' \
    "$LOCAL_EU_EMOTIONS/" \
    "${HPC_HOST}:${RDS_EU_EMOTIONS}/"

echo ""
echo "============================================================"
echo "Transfer Complete!"
echo "============================================================"
echo ""
echo "EU Emotions data location on HPC:"
echo "  $RDS_EU_EMOTIONS"
echo ""
echo "To check what was transferred:"
echo "  ssh $HPC_HOST \"du -sh '$RDS_EU_EMOTIONS'\""
echo "  ssh $HPC_HOST \"ls -lh '$RDS_EU_EMOTIONS' | head -20\""
echo ""

