#!/usr/bin/env bash
#SBATCH -J upgrade_tf
#SBATCH -A BARON-COHEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o logs/upgrade_transformers_%j.out
#SBATCH -e logs/upgrade_transformers_%j.err

set -euo pipefail

PROJECT_ROOT=~/rds/hpc-work/study2
ENV_NAME=mr_eu_open_llm

module load miniconda || module load miniconda3
export CONDA_ENVS_PATH="${PROJECT_ROOT}/conda_envs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

python -m pip install --upgrade "transformers>=4.58" "huggingface_hub>=0.34.0,<1.0"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from transformers import AutoProcessor; print('AutoProcessor ok')"
python -c "from transformers import Gemma4Processor; print('Gemma4Processor ok')" 2>/dev/null \
  || echo "Gemma4Processor not in this transformers build (may still work via AutoProcessor)"

echo "Done."
