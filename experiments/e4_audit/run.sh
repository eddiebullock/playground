#!/bin/zsh
# Runs after the E1/E2 queue so only one model is resident.
while pgrep -f "e1_pragmatics/queue.sh" >/dev/null; do sleep 30; done
cd /Users/eb2007/playground/social_ai/gloss/app
for m in qwen3:4b qwen3:8b; do
  node --experimental-strip-types scripts/audit.ts $m > ../experiments/e4_audit/${m/:/-}.json 2> ../experiments/e4_audit/${m/:/-}.log
done
echo AUDIT_DONE
