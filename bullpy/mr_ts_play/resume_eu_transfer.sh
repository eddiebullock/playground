#!/bin/bash
# Resume EU_emotions transfer - rsync will automatically skip already-transferred files

echo "============================================================"
echo "Resuming EU-Emotion Face and Voice Files Transfer to RDS"
echo "============================================================"
echo ""
echo "Destination: RDS Storage (~/rds/rds-autism-research-ePtR33Nsgi4/data/EU_emotions)"
echo ""
echo "Rsync will automatically:"
echo "  ✓ Skip files that are already transferred"
echo "  ✓ Only transfer missing or incomplete files"
echo "  ✓ Preserve directory structure"
echo ""
echo "Running rsync with same parameters as before..."
echo ""

cd /Users/eb2007/playground/bullpy/mr_ts_play
SKIP_PROMPT=1 ./rsync_eu_face_voice.sh --resume
