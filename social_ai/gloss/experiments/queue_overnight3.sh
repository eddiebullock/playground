#!/bin/zsh
# Third queue: pre-registered "fit" stance on Qwen (E1 Hu items + all pairs).
G=/Users/eb2007/playground/social_ai/gloss
L=$G/experiments/overnight.log
until grep -q QUEUE2_DONE $L 2>/dev/null; do sleep 60; done
cd $G/experiments/e1_pragmatics
for m in qwen3:8b qwen3:4b; do for t in hu sincere sincere_all; do
  python3 run.py --model $m --task $t --mode fit >> overnight_e1.log 2>&1; echo "[$(date '+%H:%M')] e1 $m $t fit" >> $L
done; done
echo "[$(date '+%H:%M')] QUEUE3_DONE" >> $L
