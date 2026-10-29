#!/bin/zsh
# After the v2 E4 run: put reflect.ts back to the committed engine (keeping
# the v2 diff), then repeat the replies audit with seeds 1 and 2 to measure
# run-to-run noise against the seed-0 run.
cd /Users/eb2007/playground/social_ai/gloss
until grep -q E4_V2_DONE experiments/e4_product/run_v2.log 2>/dev/null; do sleep 60; done
git diff app/src/engine/reflect.ts > experiments/e4_product/gen_v2.diff
git checkout app/src/engine/reflect.ts
cd app
for s in 1 2; do
  node --experimental-strip-types scripts/audit_replies.ts qwen3:8b $s > ../experiments/e4_audit/replies_qwen3-8b_seed$s.json 2> ../experiments/e4_audit/replies_qwen3-8b_seed$s.log
done
echo SEEDS_DONE >> ../experiments/e4_audit/replies_qwen3-8b_seed2.log
