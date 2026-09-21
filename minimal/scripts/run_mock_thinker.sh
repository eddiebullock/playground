#!/usr/bin/env bash
# Start the Thinker mock on the laptop (no GPU, not Qwen).
# Use this to prove the HTTP contract before the HPC session.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
cd "$ROOT/thinker"
exec "$PY" server.py --mock --host 127.0.0.1 --port 8000
