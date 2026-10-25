#!/bin/zsh
# After the current-engine E4 run: apply generation prompt v2 (uncommitted)
# and run it on the held-out even half, judge off (hand-labelled later).
cd /Users/eb2007/playground/social_ai/gloss
until grep -q E4_CURRENT_DONE experiments/e4_product/run_current.log; do sleep 60; done
python3 experiments/e4_product/gen_v2_patch.py
cd app && npx tsc --noEmit && echo tsc ok
E4_TAG=v2 E4_HALF=even E4_JUDGE=0 npm run eval -- qwen3:8b > ../experiments/e4_product/run_v2.log 2>&1
echo E4_V2_DONE >> ../experiments/e4_product/run_v2.log
