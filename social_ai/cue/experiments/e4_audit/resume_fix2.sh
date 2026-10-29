#!/bin/zsh
# Resume the fix-2 replies audit after the 2026-10-25 restart: seeds 1-2
# (seed 0 is done).
cd /Users/eb2007/playground/social_ai/gloss/app
for s in 1 2; do
  node --experimental-strip-types scripts/audit_replies.ts qwen3:8b $s > ../experiments/e4_audit/replies_fix2_seed$s.json 2> ../experiments/e4_audit/replies_fix2_seed$s.log
done
echo FIX2_DONE >> ../experiments/e4_audit/replies_fix2_seed2.log
