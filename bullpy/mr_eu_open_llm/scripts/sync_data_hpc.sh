#!/usr/bin/env bash
# Sync EU-Emotions UK Voices + Mindreading item folders to CSD3 (not Emotions/Audio/).
#
# Usage (from Mac, with OneDrive paths as defaults):
#   bash scripts/sync_data_hpc.sh
#
# Override sources:
#   EU_VOICES_SRC="/path/to/EU Emotion - UK Voices" \
#   MR_EMOTIONS_SRC="/path/to/MindReading/Emotions" \
#   bash scripts/sync_data_hpc.sh

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-eb2007}"
REMOTE_HOST="${REMOTE_HOST:-login.hpc.cam.ac.uk}"
REMOTE_BASE="${REMOTE_BASE:-~/rds/hpc-work/study2}"

EU_VOICES_SRC="${EU_VOICES_SRC:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions/EU Emotion - UK Voices}"
MR_EMOTIONS_SRC="${MR_EMOTIONS_SRC:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions}"

REMOTE_EU="${REMOTE_BASE}/data/eu_emotions_118"
REMOTE_MR="${REMOTE_BASE}/data/mindreading"

echo "=== EU UK Voices -> ${REMOTE_HOST}:${REMOTE_EU}/EU Emotion - UK Voices/ ==="
rsync -av --progress \
  "${EU_VOICES_SRC}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EU}/EU Emotion - UK Voices/"

echo ""
echo "=== Mindreading item folders (V + T .mov; excludes Emotions/Audio) -> ${REMOTE_MR}/ ==="
rsync -av --progress \
  --exclude 'Audio/' \
  --exclude 'Audio/**' \
  "${MR_EMOTIONS_SRC}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MR}/"

echo ""
echo "Done. On CSD3 verify:"
echo "  ls \"${REMOTE_EU}/EU Emotion - UK Voices/Fixed - amplified volume\" | head"
echo "  find ${REMOTE_MR} -name '*T*.mov' | head -3"
