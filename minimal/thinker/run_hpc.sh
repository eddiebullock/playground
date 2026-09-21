#!/usr/bin/env bash
# Run on the CSD3 GPU node (Wilkes3 interactive session), NOT the login node.
# Copy thinker/ here first. See HPC_TRANSFER.md.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export HF_HOME="${HF_HOME:-$HOME/rds/hpc-work/hf-cache}"
export TOKENIZERS_PARALLELISM=false
python server.py --host 127.0.0.1 --port 8000 \
  --model-id "${THINKER_MODEL_ID:-Qwen/Qwen2-VL-2B-Instruct}"
