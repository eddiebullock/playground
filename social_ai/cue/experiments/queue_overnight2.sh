#!/bin/zsh
# Second queue: starts when the first logs OVERNIGHT_DONE. E9 safety red team
# (with ideas for replying) and E10 two-way drafts, shipped engine, qwen3:8b.
G=/Users/eb2007/playground/social_ai/gloss
L=$G/experiments/overnight.log
until grep -q OVERNIGHT_DONE $L 2>/dev/null; do sleep 60; done
cd $G/app
node --experimental-strip-types scripts/run_set.ts $G/experiments/e9_safety/items.json $G/experiments/e9_safety/results_qwen3-8b.jsonl --replies 2> $G/experiments/e9_safety/run.log
echo "[$(date '+%H:%M')] e9 safety" >> $L
node --experimental-strip-types scripts/run_set.ts $G/experiments/e10_twoway/items.json $G/experiments/e10_twoway/results_qwen3-8b.jsonl 2> $G/experiments/e10_twoway/run.log
echo "[$(date '+%H:%M')] e10 twoway" >> $L
echo "[$(date '+%H:%M')] QUEUE2_DONE" >> $L
