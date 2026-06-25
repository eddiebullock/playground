#!/usr/bin/env bash
#SBATCH -J msr_audio_audit
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o logs/audio_audit_%j.out
#SBATCH -e logs/audio_audit_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm
MAX_AUDIT_TRIALS="${MAX_AUDIT_TRIALS:-118}"

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_ROOT}"
mkdir -p results/audit logs

python << EOF
import json
from pathlib import Path

from scripts.evaluate import load_trials_from_manifest
from scripts.eu_audio_resolver import build_audio_mapping_audit, save_audio_mapping_audit
from scripts.mindreading_audio_resolver import (
    build_audio_mapping_audit as build_mr_audit,
    save_audio_mapping_audit,
)

eu_root = Path("data/eu_emotions_118")
eu_manifest = Path("data/eu_emotions_118_manifest.json")
n = int("${MAX_AUDIT_TRIALS}")

trials, _ = load_trials_from_manifest(eu_manifest, eu_root)
trials = trials[:n]

for cond in ("audio_only", "multimodal"):
    rows = build_audio_mapping_audit(trials, base_data_dir=eu_root, condition=cond, seed=42)
    out = Path(f"results/audit/eu_{cond}_audit.json")
    save_audio_mapping_audit(rows, out)
    ok = sum(1 for r in rows if r.get("resolved_audio_path"))
    print(f"EU {cond}: {ok}/{len(rows)} trials with audio -> {out}")

mr_manifest = Path("data/mindreading_test_manifest.json")
if mr_manifest.is_file():
    mr_root = Path("data/mindreading")
    mt, _ = load_trials_from_manifest(mr_manifest, mr_root)
    mt = mt[:n]
    rows = build_mr_audit(mt, base_data_dir=mr_root)
    out = Path("results/audit/mindreading_audio_audit.json")
    save_audio_mapping_audit(rows, out)
    ok = sum(1 for r in rows if r.get("resolved_audio_path"))
    print(f"Mindreading: {ok}/{len(rows)} trials with audio -> {out}")
else:
    print(f"Skip Mindreading audit (no {mr_manifest})")
EOF

echo "Audio audit complete."
