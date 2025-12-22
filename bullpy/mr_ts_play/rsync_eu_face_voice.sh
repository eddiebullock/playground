#!/bin/bash
# Rsync only face and voice files from EU_emotions to RDS storage on HPC

SOURCE_DIR="/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions"
DEST_HOST="eb2007@login-cpu.hpc.cam.ac.uk"
# Transfer to RDS storage (not regular HPC storage)
# RDS path: /home/eb2007/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions
# (or /rds/user/eb2007/rds-autism-research-ePtR33Nsgi4/data/EU_emotions)
DEST_DIR="/home/eb2007/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions"

# Check for dry-run flag
DRY_RUN=""
if [[ "$1" == "--dry-run" ]] || [[ "$1" == "-n" ]]; then
    DRY_RUN="--dry-run"
    echo "DRY RUN MODE - No files will be transferred"
    echo ""
fi

echo "============================================================"
echo "Rsyncing EU-Emotion Face and Voice Files to RDS Storage"
echo "============================================================"
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_HOST:$DEST_DIR"
echo ""
echo "Transfer Summary:"
echo "  - Face files: 492 video files (~45.49 GB)"
echo "  - Voice files: 695 audio files (~0.06 GB)"
echo "  - Total: 1,187 files (~45.55 GB)"
echo ""
echo "Estimated transfer time:"
echo "  - Fast connection (100 Mbps): ~1 hour"
echo "  - Medium connection (50 Mbps): ~2 hours"
echo "  - Slow connection (10 Mbps): ~10 hours"
echo ""
echo "Including:"
echo "  - Face files: emotions*/HD Version - Face, Body, Social/Faces - HD Version/**"
echo "  - Voice files: EU Emotion - UK Voices/Original/**"
echo ""
# Skip prompt if running non-interactively, if SKIP_PROMPT is set, or if resuming
if [[ -t 0 ]] && [[ -z "$SKIP_PROMPT" ]] && [[ "$1" != "--resume" ]]; then
    read -p "Press Enter to continue or Ctrl+C to cancel..."
fi

# If resuming, add a note
if [[ "$1" == "--resume" ]]; then
    echo "🔄 RESUME MODE: Rsync will skip already-transferred files"
    echo ""
fi
echo ""
echo "Starting rsync..."
echo ""

rsync -avz --progress $DRY_RUN \
  --include='emotions*/' \
  --include='emotions*/HD Version - Face, Body, Social/' \
  --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/' \
  --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/' \
  --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/**' \
  --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/' \
  --include='emotions*/HD Version - Face, Body, Social/Faces - HD Version/Original/**' \
  --include='EU Emotion - UK Voices/' \
  --include='EU Emotion - UK Voices/Original/' \
  --include='EU Emotion - UK Voices/Original/**' \
  --exclude='*' \
  "$SOURCE_DIR/" "$DEST_HOST:$DEST_DIR/"

echo ""
echo "============================================================"
if [[ -n "$DRY_RUN" ]]; then
    echo "Dry run complete! Review above and run without --dry-run to transfer."
else
    echo "Rsync complete!"
fi
echo "============================================================"
