#!/usr/bin/env bash
# Run PROTOCOL_V2 Phases 1–4 (CPU analyses) using mr_ts_play venv when present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${STUDY4_PYTHON:-/Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python}"
[[ -x "$PY" ]] || PY="${PYTHON:-python3}"

echo "== Phase 1: CARD structure =="
"$PY" study4_rmet/scripts/build_card_rmet_structure.py

echo "== Phase 2: behavioural B1/B2 =="
"$PY" study4_rmet/scripts/behavioural_profile_alignment.py

echo "== Phase 3: RSA + probes =="
"$PY" study4_rmet/scripts/rsa_probe_card_axes.py

echo "== Phase 4: causal axis geometry (CPU) =="
"$PY" study4_rmet/scripts/causal_rmet_axes.py

echo "== Phase 4b: steer protocol (no GPU => planned_only) =="
"$PY" study4_rmet/scripts/steer_rmet_axes.py --model qwen3vl --layer 4 --protocol_only

echo "== Contamination limitations (offline) =="
"$PY" study4_rmet/scripts/contamination_option_order_smoke.py --offline_report

echo "== tests =="
"$PY" -m pytest study4_rmet/scripts/tests/test_card_structure.py -q || true

echo "Done. See results/card_structure/, results/behavioural_v2/, results/mech/"
