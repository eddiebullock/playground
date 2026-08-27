#!/usr/bin/env bash
# Commercial API panel for study4 RMET (gpt5, claude_opus, gemini_flash).
# Runs locally (not HPC). Same stimuli/prompt/schema as open-weight eval.
#
# Usage (from mr_eu_open_llm repo root):
#   ./study4_rmet/scripts/run_rmet_api_panel.sh              # full 36 items × 3 models
#   ./study4_rmet/scripts/run_rmet_api_panel.sh smoke        # 3 items each
#   ./study4_rmet/scripts/run_rmet_api_panel.sh full gpt5    # one model only
#
# After completion, re-run alignment (includes commercial + open models):
#   /Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python \
#     study4_rmet/scripts/alignment_analyses.py

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${STUDY4_PYTHON:-/Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi

MODE="${1:-full}"
ONLY_MODEL="${2:-}"

TAG="full"
MAX_ITEMS=""
if [[ "$MODE" == "smoke" ]]; then
  TAG="smoke3"
  MAX_ITEMS="3"
fi

MODELS=(gpt5 claude_opus gemini_flash)
if [[ -n "$ONLY_MODEL" ]]; then
  MODELS=("$ONLY_MODEL")
fi

echo "python=$PY mode=$MODE models=${MODELS[*]}"
echo "Outputs -> study4_rmet/results/model/<model>/rmet_eval_<model>_${TAG}_seed42.json"

for m in "${MODELS[@]}"; do
  echo "======== $(date '+%F %T') starting $m ========"
  OUT="study4_rmet/results/model/${m}/rmet_eval_${m}_${TAG}_seed42.json"
  CMD=("$PY" study4_rmet/scripts/evaluate_rmet_api.py --model "$m" --output "$OUT" --seed 42 --n_samples 10)
  if [[ -n "$MAX_ITEMS" ]]; then
    CMD+=(--max_items "$MAX_ITEMS")
  fi
  "${CMD[@]}"
  echo "======== $(date '+%F %T') finished $m ========"
done

echo "Done. Run alignment_analyses.py to refresh A1/A3 (A2 activations remain open-weight only)."
