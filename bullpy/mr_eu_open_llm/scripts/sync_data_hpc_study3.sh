#!/usr/bin/env bash
# Sync EU Faces videos + Fixed UK Voices (+ optional Mindreading) to CSD3 study3.
#
# Defaults target /home/$USER/rds/hpc-work/study3 (does NOT touch study2).
# study2 sync remains: scripts/sync_data_hpc.sh
#
# Scope (study3):
#   - ALL emotions* packs, but ONLY "Faces - HD Version/" (Original + EDITED)
#   - UK Voices: "Fixed - amplified volume/" only
#   - Excludes: Body Gestures, Social Scenes, Still Images zip
#
# Auth: opens one SSH ControlMaster (password + TOTP once); mkdir/rsync reuse it.
#
# Usage (from Mac repo root; will prompt for SSH/TOTP once):
#   bash scripts/sync_data_hpc_study3.sh
#
# Sync only voices (skip face packs):
#   SYNC_EU_PACKS=0 bash scripts/sync_data_hpc_study3.sh
#
# Sync Mindreading from Mac (usually unnecessary if study2→study3 copy already done):
#   SYNC_MINDREADING=1 bash scripts/sync_data_hpc_study3.sh
#
# Overrides:
#   REMOTE_BASE=/home/eb2007/rds/hpc-work/study3   # use absolute path (not ~)
#   EU_ROOT=.../EU_Emotions
#   EU_VOICES_SRC=.../Fixed - amplified volume
#   MR_EMOTIONS_SRC=.../MindReading/Emotions

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-eb2007}"
REMOTE_HOST="${REMOTE_HOST:-login.hpc.cam.ac.uk}"
# Absolute path required: quoted '~' does not expand over ssh and mkdir tries
# to create a literal '~' on the small home FS (often full) instead of RDS.
REMOTE_BASE="${REMOTE_BASE:-/home/${REMOTE_USER}/rds/hpc-work/study3}"
# Allow callers to pass ~/... ; expand locally before remote use.
if [[ "${REMOTE_BASE}" == ~* ]]; then
  REMOTE_BASE="/home/${REMOTE_USER}${REMOTE_BASE:1}"
fi

EU_ROOT="${EU_ROOT:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_Emotions}"
# UK Voices live under the separate EU_emotions_faces tree, not under EU_ROOT.
EU_VOICES_ROOT="${EU_VOICES_ROOT:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions_faces/audio}"
EU_VOICES_SRC="${EU_VOICES_SRC:-${EU_VOICES_ROOT}/Fixed - amplified volume}"
MR_EMOTIONS_SRC="${MR_EMOTIONS_SRC:-/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions}"

SYNC_EU_PACKS="${SYNC_EU_PACKS:-1}"
SYNC_VOICES="${SYNC_VOICES:-1}"
SYNC_MINDREADING="${SYNC_MINDREADING:-0}"

REMOTE_EU="${REMOTE_BASE}/data/eu_emotions"
REMOTE_MR="${REMOTE_BASE}/data/mindreading"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

if [[ ! -d "${EU_ROOT}" ]]; then
  echo "ERROR: EU_ROOT not found: ${EU_ROOT}" >&2
  exit 1
fi

CTRL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/study3_ssh.XXXXXX")"
CTRL_PATH="${CTRL_DIR}/mux"
cleanup_ssh() {
  ssh -S "${CTRL_PATH}" -O exit "${REMOTE}" 2>/dev/null || true
  rm -rf "${CTRL_DIR}"
}
trap cleanup_ssh EXIT

echo "Remote base: ${REMOTE}:${REMOTE_BASE}"
echo "Opening SSH master (login once for this run)..."
ssh -M -S "${CTRL_PATH}" -o ControlPersist=yes -fnN "${REMOTE}"
RSYNC_RSH="ssh -S ${CTRL_PATH} -o ControlMaster=no"

echo "Creating remote dirs (RDS, not home)..."
ssh -S "${CTRL_PATH}" -o ControlMaster=no "${REMOTE}" \
  "mkdir -p \"${REMOTE_EU}/EU Emotion - UK Voices/Fixed - amplified volume\" \"${REMOTE_MR}\""

if [[ "${SYNC_VOICES}" == "1" ]]; then
  if [[ ! -d "${EU_VOICES_SRC}" ]]; then
    echo "ERROR: EU_VOICES_SRC not found: ${EU_VOICES_SRC}" >&2
    exit 1
  fi
  # OneDrive placeholders rsync as empty files; refuse to sync a dehydrated tree.
  voice_files=$(find "${EU_VOICES_SRC}" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' \) | wc -l | tr -d ' ')
  if [[ "${voice_files}" -lt 100 ]]; then
    echo "ERROR: only ${voice_files} audio files under ${EU_VOICES_SRC}; hydrate OneDrive first." >&2
    exit 1
  fi
  echo "UK Voices source OK: ${voice_files} audio files."
  echo ""
  echo "=== UK Voices (Fixed - amplified volume only) -> ${REMOTE_EU}/EU Emotion - UK Voices/Fixed - amplified volume/ ==="
  rsync -av --progress -e "${RSYNC_RSH}" \
    --exclude '.DS_Store' \
    "${EU_VOICES_SRC}/" \
    "${REMOTE}:${REMOTE_EU}/EU Emotion - UK Voices/Fixed - amplified volume/"
fi

if [[ "${SYNC_EU_PACKS}" == "1" ]]; then
  echo ""
  echo "=== emotions* Faces only (~45G) -> ${REMOTE_EU}/ ==="
  echo "Includes: Faces - HD Version (Original + EDITED). Excludes: Body Gestures, Social Scenes."
  shopt -s nullglob
  packs=("${EU_ROOT}"/emotions*)
  if [[ ${#packs[@]} -eq 0 ]]; then
    echo "ERROR: no emotions* dirs under ${EU_ROOT}" >&2
    exit 1
  fi
  echo "Pack count: ${#packs[@]} (expect 68)"
  rsync -av --progress -e "${RSYNC_RSH}" \
    --exclude '.DS_Store' \
    --exclude 'Body Gestures - HD Version/' \
    --exclude 'Body Gestures - HD Version/**' \
    --exclude 'Social Scenes - HD Version/' \
    --exclude 'Social Scenes - HD Version/**' \
    "${packs[@]}" \
    "${REMOTE}:${REMOTE_EU}/"
fi

if [[ "${SYNC_MINDREADING}" == "1" ]]; then
  if [[ ! -d "${MR_EMOTIONS_SRC}" ]]; then
    echo "ERROR: MR_EMOTIONS_SRC not found: ${MR_EMOTIONS_SRC}" >&2
    exit 1
  fi
  echo ""
  echo "=== Mindreading Emotions (excludes Audio/) -> ${REMOTE_MR}/ ==="
  rsync -av --progress -e "${RSYNC_RSH}" \
    --exclude 'Audio/' \
    --exclude 'Audio/**' \
    --exclude '.DS_Store' \
    "${MR_EMOTIONS_SRC}/" \
    "${REMOTE}:${REMOTE_MR}/"
fi

echo ""
echo "Done. NEXT: build a full-EU manifest (study2 eu_emotions_118_manifest.json is NOT sufficient)."
echo "Then point config/Slurm PROJECT_ROOT at study3."
echo ""
echo "On CSD3 verify:"
echo "  ls \"\${HOME}/rds/hpc-work/study3/data/eu_emotions\" | grep -c '^emotions'"
echo "  ls \"\${HOME}/rds/hpc-work/study3/data/eu_emotions/EU Emotion - UK Voices/Fixed - amplified volume\" | grep -cv '\\.'   # ~27"
echo "  find \"\${HOME}/rds/hpc-work/study3/data/eu_emotions\" -path '*/Faces*/EDITED/*' \\( -name '*.mp4' -o -name '*.mov' \\) | wc -l"
echo "  find \"\${HOME}/rds/hpc-work/study3/data/eu_emotions\" -path '*Body Gestures*' | head   # should be empty after cleanup"
echo "  find \"\${HOME}/rds/hpc-work/study3/data/mindreading\" -mindepth 2 -maxdepth 2 -type d | wc -l   # ~412 under 01-24"
echo "  df -h \"\${HOME}/rds/hpc-work\""
