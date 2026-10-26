#!/bin/zsh
# Overnight queue, 2026-10-25. One model resident at a time on 16 GB, so every
# step runs in sequence. Product questions first, then Paper 1 breadth, then
# the audit across models. Each step logs; a failed step does not stop the queue.
G=/Users/eb2007/playground/social_ai/gloss
L=$G/experiments/overnight.log
log() { echo "[$(date '+%H:%M')] $*" >> $L; }
cd $G/app
until grep -q FIX2_DONE $G/experiments/e4_audit/replies_fix2_seed2.log 2>/dev/null; do sleep 60; done
log "fix2 done; queue starts"

# 1. Readings audit, 48 scenarios x 6 disclosures, scorer as shipped vs words-not-reader.
for v in stance words; do
  node --experimental-strip-types scripts/audit_v2.ts qwen3:8b $v > $G/experiments/e4_audit/v2_qwen3-8b_$v.json 2> $G/experiments/e4_audit/v2_qwen3-8b_$v.log; log "audit v2 qwen3:8b $v"
done

# 2. New minimal pairs on Qwen (fast).
cd $G/experiments/e1_pragmatics
for m in qwen3:8b qwen3:4b; do for mode in direct stance; do
  python3 run.py --model $m --task sincere_all --mode $mode >> $G/experiments/e1_pragmatics/overnight_e1.log 2>&1; log "sincere_all $m $mode"
done; done

# 3. Write more, show fewer: development half of E4 (odd items).
cd $G/app
E4_TAG=v2wide E4_WIDE=6 E4_HALF=odd E4_JUDGE=0 node --experimental-strip-types scripts/eval.ts qwen3:8b 2> $G/experiments/e4_product/run_v2wide_odd.log; log "e4 v2wide odd"

# 4. External held-out: real tweets (iSarcasmEval), shipped engine then wide.
E4_ITEMS=$G/experiments/e8_isarcasm/items.json E4_TAG=v2 E4_JUDGE=0 node --experimental-strip-types scripts/eval.ts qwen3:8b 2> $G/experiments/e8_isarcasm/run_v2.log; log "e8 v2"
E4_ITEMS=$G/experiments/e8_isarcasm/items.json E4_TAG=v2wide E4_WIDE=6 E4_JUDGE=0 node --experimental-strip-types scripts/eval.ts qwen3:8b 2> $G/experiments/e8_isarcasm/run_v2wide.log; log "e8 v2wide"

# 5. Paper 1 breadth: other families, direct and stance, all pairs, MUStARD.
cd $G/experiments/e1_pragmatics
for m in llama3.1:8b gemma3:12b gemma3:4b; do
  ollama list | grep -q "^$m" || { log "skip $m (not pulled)"; continue; }
  for mode in direct stance; do
    for t in hu sincere sincere_all; do
      python3 run.py --model $m --task $t --mode $mode >> $G/experiments/e1_pragmatics/overnight_e1.log 2>&1; log "e1 $m $t $mode"
    done
  done
  python3 run.py --model $m --task mustard >> $G/experiments/e1_pragmatics/overnight_e1.log 2>&1; log "e1 $m mustard"
done

# 6. Readings audit across models (scorer as shipped).
cd $G/app
for m in qwen3:4b llama3.1:8b gemma3:12b gemma3:4b; do
  ollama list | grep -q "^$m" || continue
  f=$G/experiments/e4_audit/v2_${m//:/-}_stance
  node --experimental-strip-types scripts/audit_v2.ts $m stance > $f.json 2> $f.log; log "audit v2 $m"
done
log OVERNIGHT_DONE
