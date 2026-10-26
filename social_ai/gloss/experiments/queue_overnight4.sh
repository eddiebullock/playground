#!/bin/zsh
# Fourth queue: model-based safety check scores (safety_model.ts), qwen3:8b.
# E9b (threshold setting), false-positive sets, then E9c (test) once written.
G=/Users/eb2007/playground/social_ai/gloss
L=$G/experiments/overnight.log
until grep -q QUEUE3_DONE $L 2>/dev/null; do sleep 60; done
cd $G/app
for s in e9b e9 audit e10 e4 e8; do
  node --experimental-strip-types scripts/safety_model_eval.ts $s 2>> $G/experiments/e9_safety/model_scores/run.log; echo "[$(date '+%H:%M')] safety scores $s" >> $L
done
until [ -f $G/experiments/e9c_safety_test/items.json ]; do sleep 60; done
node --experimental-strip-types scripts/safety_model_eval.ts e9c 2>> $G/experiments/e9_safety/model_scores/run.log; echo "[$(date '+%H:%M')] safety scores e9c" >> $L
echo "[$(date '+%H:%M')] QUEUE4_DONE" >> $L
