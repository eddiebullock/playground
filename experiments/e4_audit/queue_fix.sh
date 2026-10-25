#!/bin/zsh
# After the latency run: replies audit with the engine as committed (v2 +
# reply-prompt fix for the anxiety drift), seeds 0-2.
cd /Users/eb2007/playground/social_ai/gloss
until grep -q LATENCY_DONE experiments/e4_product/latency_v2.log 2>/dev/null; do sleep 60; done
cd app
for s in 0 1 2; do
  node --experimental-strip-types scripts/audit_replies.ts qwen3:8b $s > ../experiments/e4_audit/replies_fix_seed$s.json 2> ../experiments/e4_audit/replies_fix_seed$s.log
done
echo FIX_DONE >> ../experiments/e4_audit/replies_fix_seed2.log
