#!/usr/bin/env bash
#SBATCH -J s4_rmet_steer
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o logs/study4_rmet_steer_%x_%j.out
#SBATCH -e logs/study4_rmet_steer_%x_%j.err
#
# C1 GPU steer/patch smoke then full on qwen3vl.
# Matches env setup of rmet_activation_extract.sh / rmet_model_full.sh.
#
# Submit FROM ~/rds/hpc-work/study4_rmet:
#   unset VIRTUAL_ENV
#   MODEL=qwen3vl LAYER=4 sbatch study4_rmet/slurm_jobs/rmet_causal_steer_smoke.sh
#   MODE=full MODEL=qwen3vl LAYER=4 sbatch study4_rmet/slurm_jobs/rmet_causal_steer_smoke.sh

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study4_rmet
ENV_ROOT=~/rds/hpc-work/study2
MOLMO2_VENV="${MOLMO2_VENV:-~/rds/hpc-work/study3/venvs/molmo2}"
MODEL="${MODEL:-qwen3vl}"
LAYER="${LAYER:-4}"
MODE="${MODE:-smoke}"
SEED="${SEED:-42}"

USE_MOLMO2_VENV=0
if [ "${MODEL}" = "molmo2" ]; then
  MOLMO2_VENV_EXPANDED=$(eval echo "${MOLMO2_VENV}")
  if [ -d "${MOLMO2_VENV_EXPANDED}" ]; then
    USE_MOLMO2_VENV=1
    MOLMO2_VENV="${MOLMO2_VENV_EXPANDED}"
  else
    ENV_NAME="${ENV_NAME:-mr_eu_molmo2}"
  fi
fi
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "note: dropping inherited VIRTUAL_ENV=${VIRTUAL_ENV}"
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
mkdir -p logs study4_rmet/results/mech

HF_CACHE_DIR="${PROJECT_ROOT}/hf_cache"
mkdir -p "${HF_CACHE_DIR}"/{modules,transformers,datasets,torch}
export HF_HOME="${HF_CACHE_DIR}"
export HF_MODULES_CACHE="${HF_CACHE_DIR}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export MR_EU_HPC_PROJECT="study4_rmet"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "cwd=$(pwd)"
echo "which python: $(command -v python)"
echo "env: ${ENV_NAME}$([ "${USE_MOLMO2_VENV}" = "1" ] && echo " + ${MOLMO2_VENV}")"
echo "transformers $(python -c 'import transformers; print(transformers.__version__)')"
echo "C1 steer ${MODE} MODEL=${MODEL} LAYER=${LAYER}"

# Fail loudly with clear paths (avoids opaque FileNotFound later).
need=(
  "study4_rmet/scripts/causal_rmet_axes.py"
  "study4_rmet/scripts/steer_rmet_axes.py"
  "study4_rmet/results/card_structure/item_classes_preregistered.json"
  "study4_rmet/results/card_structure/choice_distributions.json"
  "study4_rmet/results/activations/${MODEL}/full/layer${LAYER}_rmet_seed42.npy"
  "study4_rmet/data/rmet/stimuli/manifest.json"
  "scripts/evaluate.py"
  "config.py"
)
for f in "${need[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "ERROR: missing required path: ${PROJECT_ROOT}/${f}" >&2
    echo "Hint: from laptop run ./study4_rmet/sync.sh push && ./study4_rmet/sync.sh push-repo-readonly" >&2
    exit 2
  fi
done
echo "preflight OK"

python study4_rmet/scripts/causal_rmet_axes.py --models "${MODEL}" --layers "${LAYER}" --seed "${SEED}"

if [[ "${MODE}" == "smoke" ]]; then
  # ~117 gens (3 items x 13 conditions x 3 samples); keep both patch modes for VLM control.
  # Use --alphas=... so argparse does not treat -1 as a flag.
  python study4_rmet/scripts/steer_rmet_axes.py \
    --model "${MODEL}" --layer "${LAYER}" \
    --smoke --patch_modes last_token,all_tokens --alphas=-1,1 --seed "${SEED}"
else
  python study4_rmet/scripts/steer_rmet_axes.py \
    --model "${MODEL}" --layer "${LAYER}" \
    --n_samples 10 --max_items 36 \
    --patch_modes last_token,all_tokens --alphas=-2,-1,1,2 --seed "${SEED}"
fi

echo "Done. See study4_rmet/results/mech/steer_*"
ls -lh study4_rmet/results/mech/steer_*"${MODEL}"*layer"${LAYER}"* 2>/dev/null || ls -lh study4_rmet/results/mech/ | head -40
