#!/bin/zsh
# After the replies seed runs: adopt prompt v2 in the engine, then measure
# Reflect latency (time to first streamed reading, total) on 20 items.
cd /Users/eb2007/playground/social_ai/gloss
until grep -q SEEDS_DONE experiments/e4_audit/replies_qwen3-8b_seed2.log 2>/dev/null; do sleep 60; done
git apply experiments/e4_product/gen_v2.diff && echo V2_APPLIED
cd app && node --experimental-strip-types scripts/latency.ts qwen3:8b 20 > ../experiments/e4_product/latency_v2.jsonl 2> ../experiments/e4_product/latency_v2.log
echo LATENCY_DONE >> ../experiments/e4_product/latency_v2.log
