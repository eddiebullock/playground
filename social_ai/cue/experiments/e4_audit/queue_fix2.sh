#!/bin/zsh
# Replies audit after fix 2 (not-replying option tied to the message), seeds 0-2.
cd /Users/eb2007/playground/social_ai/gloss
cd app
for s in 0 1 2; do
  node --experimental-strip-types scripts/audit_replies.ts qwen3:8b $s > ../experiments/e4_audit/replies_fix2_seed$s.json 2> ../experiments/e4_audit/replies_fix2_seed$s.log
done
echo FIX2_DONE >> ../experiments/e4_audit/replies_fix2_seed2.log
