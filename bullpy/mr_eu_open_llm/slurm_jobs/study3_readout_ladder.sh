#!/usr/bin/env bash
#SBATCH -J s3_ladder
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -o logs/study3_ladder_%x_%j.out
#SBATCH -e logs/study3_ladder_%x_%j.err
#
# Option-free readout ladder on CREMA-D calibration clips. See scripts/readout_ladder.py.
#   GEOMETRY=1 MODEL=gemma4 sbatch slurm_jobs/study3_readout_ladder.sh
#   MODEL=qwen3vl sbatch --dependency=afterok:JOB slurm_jobs/study3_readout_ladder.sh
#   RUNGS=2,3 FORCE=1 MODEL=gemma4 sbatch ...   later rungs after a pass, for a shared rung

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
MODEL="${MODEL:-gemma4}"
RUNGS="${RUNGS:-1,2,3}"
FORCE="${FORCE:-0}"
GEOMETRY="${GEOMETRY:-0}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi
if [ "${MODEL}" = "molmo2" ]; then
  module load miniconda || module load miniconda3 || true
  export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
  export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate mr_eu_open_llm
  source "${PROJECT_ROOT}/venvs/molmo2/bin/activate"
  set -u
else
  source "${PROJECT_ROOT}/venvs/au_tools/env/bin/activate"
fi

cd "${PROJECT_ROOT}"
mkdir -p logs results/readout_ladder research/reproduction
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONWARNINGS="ignore::UserWarning"

if [ "${GEOMETRY}" = "1" ]; then
  python -m scripts.readout_ladder --step geometry
fi
EXTRA=()
if [ "${FORCE}" = "1" ]; then
  EXTRA+=(--force)
fi
python -m scripts.readout_ladder --step ladder --model "${MODEL}" --rungs "${RUNGS}" "${EXTRA[@]}"
echo "Done: ${MODEL}"
