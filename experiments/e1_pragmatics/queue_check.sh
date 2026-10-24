#!/bin/zsh
cd /Users/eb2007/playground/gloss/experiments/e1_pragmatics
while pgrep -f "e1_pragmatics/queue.sh|e4_audit/run.sh" >/dev/null; do sleep 30; done
for m in qwen3:8b qwen3:4b; do python3 run.py --model $m --task hu --mode check --seeds 1; done
echo CHECK_DONE
