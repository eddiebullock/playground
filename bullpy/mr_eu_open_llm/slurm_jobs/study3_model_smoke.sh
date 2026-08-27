#!/usr/bin/env bash
#SBATCH -J msr_s3_smoke
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# 5 trials is ~10 min including model load off Lustre. Short walltime backfills
# far better on SL3; override with `sbatch --time=...` for larger --max_trials.
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/study3_smoke_%x_%j.out
#SBATCH -e logs/study3_smoke_%x_%j.err
#
# study3 smoke test: few-trial two-stage eval on the full-EU 6AFC manifest.
# Validates model wiring (loader, processor, frame policy, foils, parsing) before
# committing to a full sweep. Weights are read from study2/models (shared).
#
#   MODEL=qwen3vl sbatch -J s3_smoke_qwen3vl slurm_jobs/study3_model_smoke.sh
#   MODEL=gemma4  sbatch -J s3_smoke_gemma4  slurm_jobs/study3_model_smoke.sh
#
# Pass -J: log filenames use %x, and Slurm cannot expand MODEL in an #SBATCH
# directive, so without it concurrent models all write to msr_s3_smoke_*.
#
# Submit one job per model rather than looping: independent short jobs backfill
# sooner and can run concurrently on different nodes.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"

# Molmo2's remote processing code declares its config flags in `optional_attributes`,
# a ProcessorMixin mechanism Transformers 5 dropped, so it needs Transformers 4.57.
# Gemma 4 needs 5.x for its own model classes, so the two cannot share an install.
# Prefer a --system-site-packages venv holding only Transformers; fall back to a full
# conda clone if that is what exists.
MOLMO2_VENV="${MOLMO2_VENV:-${PROJECT_ROOT}/venvs/molmo2}"
USE_MOLMO2_VENV=0
if [ "${MODEL}" = "molmo2" ]; then
  if [ -d "${MOLMO2_VENV}" ]; then
    USE_MOLMO2_VENV=1
  else
    ENV_NAME="${ENV_NAME:-mr_eu_molmo2}"
  fi
fi
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

CONDITION="${CONDITION:-video_only}"
STAGE="${STAGE:-both}"
MAX_TRIALS="${MAX_TRIALS:-5}"
N_OPTIONS="${N_OPTIONS:-6}"
SEED="${SEED:-42}"
# RQ1.1b: FC_SAMPLES>1 draws the forced choice repeatedly and reports response entropy.
# Off by default so a plain smoke stays fast; 20 is the calibration-run value.
FC_SAMPLES="${FC_SAMPLES:-1}"
FC_TEMPERATURE="${FC_TEMPERATURE:-1.0}"
HUMAN_OPTIONS="${HUMAN_OPTIONS:-data/eu_emotions_human_entropy.json}"

# Env vars are case-sensitive, so `FC_samples=20 sbatch ...` silently runs a 1-draw job
# and wastes the allocation. Fail before the queue rather than after the GPU.
for miscased in FC_samples fc_samples FC_Samples; do
  if [ -n "${!miscased:-}" ] && [ "${FC_SAMPLES}" = "1" ]; then
    echo "ERROR: ${miscased}=${!miscased} is set but FC_SAMPLES is not. Did you mean FC_SAMPLES?" >&2
    exit 2
  fi
done

# sbatch exports the submitting shell's environment, so a venv left active at submit time
# leaks in and shadows the conda interpreter. Gemma 4 under the Molmo2 venv fails with
# "Unrecognized processing class". Drop any inherited venv before selecting the env.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "note: dropping inherited VIRTUAL_ENV=${VIRTUAL_ENV} from the submitting shell"
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
if [ "${USE_MOLMO2_VENV}" = "1" ]; then
  source "${MOLMO2_VENV}/bin/activate"
fi
set -u

cd "${PROJECT_ROOT}"
mkdir -p results/test_runs logs

# Keep caches on project storage; /home is at its quota limit.
HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}/modules" "${HF_CACHE_DIR}/transformers" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/torch"
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "study3 smoke: model=${MODEL} condition=${CONDITION} stage=${STAGE} trials=${MAX_TRIALS} n_options=${N_OPTIONS}"
echo "RQ1.1b     : fc_samples=${FC_SAMPLES} fc_temperature=${FC_TEMPERATURE} $([ "${FC_SAMPLES}" -gt 1 ] && echo '(forced-choice entropy ON)' || echo '(single draw, no H_fc)')"
echo "env        : ${ENV_NAME}$([ "${USE_MOLMO2_VENV}" = "1" ] && echo " + ${MOLMO2_VENV}") | python $(command -v python)"
echo "versions   : transformers $(python -c 'import transformers; print(transformers.__version__)') | torch $(python -c 'import torch; print(torch.__version__)')"

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset eu_emotions \
  --condition "${CONDITION}" \
  --stage "${STAGE}" \
  --max_frames 16 \
  --fps 1 \
  --seed "${SEED}" \
  --n_options "${N_OPTIONS}" \
  --data_root data/eu_emotions \
  --manifest data/eu_emotions_full_manifest.json \
  --human_options "${HUMAN_OPTIONS}" \
  --fc_samples "${FC_SAMPLES}" \
  --fc_temperature "${FC_TEMPERATURE}" \
  --max_trials "${MAX_TRIALS}" \
  --output "results/test_runs/smoke_${MODEL}_eu_emotions_${CONDITION}_${MAX_TRIALS}trials_seed${SEED}.json"

python - "results/test_runs/smoke_${MODEL}_eu_emotions_${CONDITION}_${MAX_TRIALS}trials_seed${SEED}.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("protocol   :", m["protocol_version"], "| n_options:", m["n_options"], "| chance:", round(m["chance_level"], 4))
print("accuracy   :", m["accuracy"], "of", m["n_scored"], "scored")
print("mean H_sem :", m["mean_semantic_entropy"], "(RQ1.1a)")
print("mean H_fc  :", m.get("mean_forced_choice_entropy"), "(RQ1.1b,", m.get("forced_choice_n_samples"), "draws @ T=", m.get("forced_choice_temperature"), ")")
print("options    :", m.get("forced_choice_options_source"))
errs = [(t["trial_id"], t["error"]) for t in m["trials"] if t.get("error")]
print("errors     :", len(errs))
for tid, e in errs[:5]:
    print("   ", tid, "->", e)
for t in m["trials"][:2]:
    s1, s2 = t.get("stage1") or {}, t.get("stage2") or {}
    print("---", t["trial_id"], "| frames:", t.get("n_frames_used"), t.get("multi_frame_strategy"))
    print("    free text :", (s1.get("free_response_text") or "")[:160].replace("\n", " "))
    print("    options   :", s2.get("options"))
    print("    prediction:", s2.get("prediction"), "| correct:", s2.get("correct"))
PY

echo "study3 smoke complete."
