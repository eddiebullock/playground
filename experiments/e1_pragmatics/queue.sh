#!/bin/zsh
# Sequential so only one model is resident at a time on 16 GB.
cd /Users/eb2007/playground/gloss/experiments/e1_pragmatics
while pgrep -f "run.py --model qwen3:4b --task" >/dev/null; do sleep 30; done
python3 run.py --model gemma3:4b --task hu > log_gemma_hu.txt 2>&1
python3 run.py --model gemma3:4b --task mustard > log_gemma_mustard.txt 2>&1
python3 run.py --model qwen3:4b --task hu --mode reason --seeds 1 > log_qwen_hu_reason.txt 2>&1
python3 run.py --model gemma3:4b --task hu --mode reason --seeds 1 > log_gemma_hu_reason.txt 2>&1
python3 run.py --model qwen3:4b --task hu > log_qwen_hu_resume.txt 2>&1
python3 run.py --model qwen3:4b --task mustard > log_qwen_mustard_resume.txt 2>&1
echo QUEUE_DONE
