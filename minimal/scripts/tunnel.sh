#!/usr/bin/env bash
# Laptop: open an SSH tunnel so http://127.0.0.1:8000 reaches the Thinker
# on the GPU node through the CSD3 login node.
#
# Usage:
#   ./scripts/tunnel.sh                 # login node only; you still need a
#                                       # second hop if the server is on a GPU node
#   ./scripts/tunnel.sh gpu-q-12        # replace with YOUR sintr hostname
#
# Keep this terminal open while the replay harness runs.
set -euo pipefail
GPU_HOST="${1:-}"
if [[ -z "$GPU_HOST" ]]; then
  echo "Tunneling login.hpc.cam.ac.uk:8000 -> laptop:8000"
  echo "If the Thinker is on a GPU node, re-run: $0 <gpu-hostname>"
  exec ssh -N -L 8000:127.0.0.1:8000 eb2007@login.hpc.cam.ac.uk
else
  echo "Tunneling ${GPU_HOST}:8000 -> laptop:8000 via login.hpc.cam.ac.uk"
  exec ssh -N -L "8000:${GPU_HOST}:8000" eb2007@login.hpc.cam.ac.uk
fi
