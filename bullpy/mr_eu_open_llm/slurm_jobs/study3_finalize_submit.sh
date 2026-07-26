#!/usr/bin/env bash
# Submit all Study 2/3 final analyses for write-up.
#
# Usage (HPC study2 root):
#   bash slurm_jobs/study3_finalize_submit.sh
#
# Order: CPU probes+SAE (parallel) -> GPU path patching (3 models) ->
#        LLaVA eval+patch repair -> CPU summarize (run locally after pull).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1. Re-run probes with entropy tertiles (all models) ==="
PROBE_JID=$(sbatch --parsable "${SCRIPT_DIR}/study3_probe_tertiles.sh")
echo "  probes: ${PROBE_JID}"

echo "=== 2. NMF sparse-feature analysis (CPU, one job per model) ==="
SAE_JIDS=()
for MODEL in gemma4 qwen2vl llavanext; do
  jid=$(sbatch --parsable --export=MODEL="${MODEL}" "${SCRIPT_DIR}/study3_sae.sh")
  SAE_JIDS+=("${jid}")
  echo "  sae ${MODEL}: ${jid}"
done

echo "=== 3. Path patching at each layer (GPU, 30 trials/model) ==="
PATH_JIDS=()
for MODEL in gemma4 qwen2vl llavanext; do
  jid=$(sbatch --parsable --dependency=afterok:"${PROBE_JID}" \
    --export=MODEL="${MODEL}",MAX_TRIALS=30 \
    "${SCRIPT_DIR}/study3_path_patch.sh")
  PATH_JIDS+=("${jid}")
  echo "  path_patch ${MODEL}: ${jid}"
done

echo "=== 4. LLaVA PeftModel eval + full patching repair ==="
bash "${SCRIPT_DIR}/study3_llavanext_repair.sh"

echo ""
echo "Study 3 finalize submitted."
echo "  probes:     ${PROBE_JID}"
echo "  sae:        ${SAE_JIDS[*]}"
echo "  path_patch: ${PATH_JIDS[*]}"
echo ""
echo "After jobs complete:"
echo "  ./sync.sh pull-study3"
echo "  python -m scripts.study3_summarize"
echo "  python -m scripts.study3_figures"
