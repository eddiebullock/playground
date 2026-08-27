#!/usr/bin/env bash
#SBATCH -J s3_sweep
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Batched FC (default): ~2h is enough for Gemma/Molmo. Qwen sequential needs more —
# always override with `sbatch --time=` (see below). Shorter requests backfill better.
#SBATCH --time=02:00:00
# Inference is GPU-bound; 2 CPUs / 48G host RAM is enough and queues faster than 4/64G.
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_sweep_%x_%j.out
#SBATCH -e logs/study3_sweep_%x_%j.err
#
# Phase 4: full-manifest benchmarking run (243 trials, 6AFC) for one model.
# Produces accuracy, RQ1.1a semantic entropy and RQ1.1b forced-choice entropy in one pass.
#
#   MODEL=qwen3vl sbatch -J s3_sweep_qwen3vl slurm_jobs/study3_benchmark_sweep.sh
#   MODEL=gemma4  sbatch -J s3_sweep_gemma4  slurm_jobs/study3_benchmark_sweep.sh
#   MODEL=molmo2  sbatch -J s3_sweep_molmo2  slurm_jobs/study3_benchmark_sweep.sh
#
# Qwen3-VL OOM fix (batched 20-way FC blows A100-80GB). Sync this script first, then:
#   MODEL=qwen3vl FC_SEQUENTIAL=1 MAX_FRAMES=8 \
#     sbatch --time=08:00:00 -J s3_sweep_qwen3vl slurm_jobs/study3_benchmark_sweep.sh
# Walltime basis for sequential: ~2s/draw x 20 draws x 243 ≈ 2.7h generate + Stage1/IO;
# 8h leaves margin. If it times out, resubmit with --time=12:00:00 (do not cut FC_SAMPLES).
#
# One job per model: independent jobs backfill sooner than one long multi-model job.

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-qwen3vl}"

MOLMO2_VENV="${MOLMO2_VENV:-${PROJECT_ROOT}/venvs/molmo2}"
USE_MOLMO2_VENV=0
if [ "${MODEL}" = "molmo2" ]; then
  if [ -d "${MOLMO2_VENV}" ]; then
    USE_MOLMO2_VENV=1
  else
    echo "ERROR: molmo2 needs the Transformers 4.57 venv at ${MOLMO2_VENV}" >&2
    exit 2
  fi
fi
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

CONDITION="${CONDITION:-video_only}"
STAGE="${STAGE:-both}"
N_OPTIONS="${N_OPTIONS:-6}"
SEED="${SEED:-42}"
FC_SAMPLES="${FC_SAMPLES:-20}"
FC_TEMPERATURE="${FC_TEMPERATURE:-1.0}"
HUMAN_OPTIONS="${HUMAN_OPTIONS:-data/eu_emotions_human_entropy.json}"
# Qwen3-VL OOMed on A100-80GB with batched FC (20x num_return_sequences + video).
# Override: FC_SEQUENTIAL=1 MAX_FRAMES=8 MODEL=qwen3vl sbatch ...
FC_SEQUENTIAL="${FC_SEQUENTIAL:-0}"
MAX_FRAMES="${MAX_FRAMES:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for miscased in FC_samples fc_samples FC_Samples; do
  if [ -n "${!miscased:-}" ] && [ "${FC_SAMPLES}" = "20" ]; then
    echo "ERROR: ${miscased}=${!miscased} is set but FC_SAMPLES is not. Did you mean FC_SAMPLES?" >&2
    exit 2
  fi
done

# sbatch exports the submitting shell's environment, so an active venv would shadow the
# conda interpreter and silently run the wrong Transformers.
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
OUT_DIR="results/baseline/eu_emotions/${MODEL}"
mkdir -p "${OUT_DIR}" logs

HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}/modules" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/torch"
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
# Default to warnings: `info` prints a full model config per load, which buries the
# traceback in a multi-hour run.
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"

if [ ! -f "${HUMAN_OPTIONS}" ]; then
  echo "ERROR: human option/entropy lookup missing: ${HUMAN_OPTIONS}" >&2
  echo "       build it locally with scripts/human_entropy.py and ./sync.sh push" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "study3 sweep: model=${MODEL} condition=${CONDITION} stage=${STAGE} n_options=${N_OPTIONS} seed=${SEED}"
echo "env         : ${ENV_NAME}$([ "${USE_MOLMO2_VENV}" = "1" ] && echo " + ${MOLMO2_VENV}") | python $(command -v python)"
echo "versions    : transformers $(python -c 'import transformers; print(transformers.__version__)') | torch $(python -c 'import torch; print(torch.__version__)')"
echo "RQ1.1b      : fc_samples=${FC_SAMPLES} fc_temperature=${FC_TEMPERATURE} options=${HUMAN_OPTIONS} sequential=${FC_SEQUENTIAL} max_frames=${MAX_FRAMES}"

OUT_JSON="${OUT_DIR}/eval_v2_eu_emotions_${MODEL}_${CONDITION}_seed${SEED}.json"
EVAL_EXTRA=()
if [ "${FC_SEQUENTIAL}" = "1" ]; then
  EVAL_EXTRA+=(--fc_sequential)
fi

python -m scripts.evaluate \
  --model "${MODEL}" \
  --dataset eu_emotions \
  --condition "${CONDITION}" \
  --stage "${STAGE}" \
  --max_frames "${MAX_FRAMES}" \
  --fps 1 \
  --seed "${SEED}" \
  --n_options "${N_OPTIONS}" \
  --data_root data/eu_emotions \
  --manifest data/eu_emotions_full_manifest.json \
  --human_options "${HUMAN_OPTIONS}" \
  --fc_samples "${FC_SAMPLES}" \
  --fc_temperature "${FC_TEMPERATURE}" \
  "${EVAL_EXTRA[@]}" \
  --output "${OUT_JSON}"

python - "${OUT_JSON}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print()
print("protocol   :", m["protocol_version"], "| n_options:", m["n_options"], "| chance:", round(m["chance_level"], 4))
print("accuracy   :", m["accuracy"], "of", m["n_scored"], "scored")
print("H_sem      :", m["mean_semantic_entropy"], "(RQ1.1a)")
print("H_fc       :", m.get("mean_forced_choice_entropy"), "(RQ1.1b over", m.get("n_forced_choice_scored"), "items,",
      m.get("forced_choice_n_samples"), "draws,", m.get("forced_choice_sampling_mode"), ")")
print("options    :", m.get("forced_choice_options_source"))
errs = [(t["trial_id"], t["error"]) for t in m["trials"] if t.get("error")]
print("errors     :", len(errs))
for tid, e in errs[:5]:
    print("   ", tid, "->", e)
PY

echo "study3 sweep complete: ${OUT_JSON}"
