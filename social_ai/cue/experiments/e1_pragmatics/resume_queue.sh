#!/bin/zsh
# Resume after the 2026-10-24 interruption (disk full during repo move).
# run.py skips finished (id, seed) pairs, so reruns only fill gaps.
cd /Users/eb2007/playground/social_ai/gloss/experiments/e1_pragmatics
python3 run.py --model qwen3:8b --task mustard --mode stance
python3 run.py --model qwen3:4b --task sincere; python3 run.py --model qwen3:4b --task sincere --mode stance
for m in qwen3:8b qwen3:4b; do python3 run.py --model $m --task hu --mode check --seeds 1; done
echo E1_RESUME_DONE
cd ../../app
for m in qwen3:4b qwen3:8b; do
  node --experimental-strip-types scripts/audit.ts $m > ../experiments/e4_audit/${m/:/-}.json 2> ../experiments/e4_audit/${m/:/-}.log
done
echo AUDIT_DONE
