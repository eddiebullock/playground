#!/bin/zsh
# Sequential, priority order: one model resident at a time on 16 GB.
cd /Users/eb2007/playground/social_ai/gloss/experiments/e1_pragmatics
python3 run.py --model qwen3:8b --task hu
python3 run.py --model qwen3:4b --task mustard
python3 run.py --model qwen3:8b --task mustard
for m in qwen3:4b qwen3:8b; do python3 run.py --model $m --task hu --mode stance; done
for m in qwen3:4b qwen3:8b; do python3 run.py --model $m --task mustard --mode stance; done
for m in qwen3:4b qwen3:8b; do python3 run.py --model $m --task hu --mode reason --seeds 1; done
echo QUEUE_DONE
