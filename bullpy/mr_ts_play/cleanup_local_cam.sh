#!/bin/bash
# Cleanup script to delete large CAM media files from local laptop
# Keeps metadata files (trial definitions, splits) for local development

CAM_DATA_ROOT="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"

echo "=========================================="
echo "CAM Dataset Local Cleanup"
echo "=========================================="
echo ""
echo "This script will delete large media files from:"
echo "  $CAM_DATA_ROOT"
echo ""
echo "Files to DELETE (large media):"
echo "  - Video files (*.mov, *.mp4, *.avi)"
echo "  - Audio files (*.aif, *.wav, *.mp3)"
echo ""
echo "Files to KEEP (metadata):"
echo "  - Trial definitions (data/cam_trial_definitions_*.json)"
echo "  - Train/val/test splits (data/splits/*.csv)"
echo "  - Project code and configs"
echo "  - Model checkpoints"
echo ""
read -p "Are you sure you want to delete the media files? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled. No files deleted."
    exit 0
fi

echo ""
echo "Counting files to delete..."
VIDEO_COUNT=$(find "$CAM_DATA_ROOT" -type f \( -name "*.mov" -o -name "*.mp4" -o -name "*.avi" \) | wc -l | tr -d ' ')
AUDIO_COUNT=$(find "$CAM_DATA_ROOT" -type f \( -name "*.aif" -o -name "*.wav" -o -name "*.mp3" \) | wc -l | tr -d ' ')
TOTAL_MEDIA=$((VIDEO_COUNT + AUDIO_COUNT))

echo "  Video files: $VIDEO_COUNT"
echo "  Audio files: $AUDIO_COUNT"
echo "  Total media files: $TOTAL_MEDIA"
echo ""

# Calculate size before deletion
MEDIA_SIZE=$(find "$CAM_DATA_ROOT" -type f \( -name "*.mov" -o -name "*.mp4" -o -name "*.avi" -o -name "*.aif" -o -name "*.wav" -o -name "*.mp3" \) -exec du -ch {} + | tail -1 | cut -f1)
echo "Total size to free: $MEDIA_SIZE"
echo ""

read -p "Proceed with deletion? (yes/no): " final_confirm

if [ "$final_confirm" != "yes" ]; then
    echo "Cancelled. No files deleted."
    exit 0
fi

echo ""
echo "Deleting media files..."
find "$CAM_DATA_ROOT" -type f \( -name "*.mov" -o -name "*.mp4" -o -name "*.avi" -o -name "*.aif" -o -name "*.wav" -o -name "*.mp3" \) -delete

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Deleted: $TOTAL_MEDIA media files"
echo "Freed: ~$MEDIA_SIZE"
echo ""
echo "Note: Files are still in OneDrive cloud storage as backup."
echo "      Metadata files (trial definitions, splits) are preserved."
echo ""









