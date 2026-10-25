#!/bin/zsh
# Runs after the baseline E4 run: the replies stereotype audit, then E4 on
# the current engine. One model at a time, sequentially.
cd /Users/eb2007/playground/social_ai/gloss/app
LOG=/private/tmp/claude-502/-Users-eb2007/c14745e3-236c-452c-a1e5-68d52d97d5c2/scratchpad/e4base/run_e4.log
until grep -q E4_BASE_DONE $LOG 2>/dev/null; do sleep 60; done
node --experimental-strip-types scripts/audit_replies.ts qwen3:8b > ../experiments/e4_audit/replies_qwen3-8b.json 2> ../experiments/e4_audit/replies_qwen3-8b.log
echo REPLIES_DONE
npm run eval -- qwen3:8b > ../experiments/e4_product/run_current.log 2>&1
echo E4_CURRENT_DONE
