#!/usr/bin/env bash
#SBATCH -J a8_equiv_rest
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH -o logs/a8_equiv_rest_%j.out
#SBATCH -e logs/a8_equiv_rest_%j.err
#
# The rest of A8's GPU equivalence test (PREREG_A8 Freeze checklist item 2) after the whole-node `intr` job 40986052
# (`a8_equiv_intr.sh`) hit its one-hour limit (DECISIONS 2026-11-14, "A8's GPU equivalence test finishes in a second
# job"). In both conditions that job's two frozen runs ended with exit 0, and it stopped during the first shard run.
# This job runs each condition's remaining commands, unchanged and in the same order, through
# `a8_equiv_driver_condition_rest.py`: the shard runner for shards 0 and 1, the merge, the comparisons and the report.
# Everything else is `a8_equiv_intr.sh` line for line: the environment, the request for the node's four GPUs, the two
# one-GPU steps run at once and their 59-minute caps. The `gpu2` job 40399722 stays held while this runs.
#
# Before submitting, move each condition's partial shard0of2 folder from job 40986052 out of results/a8/equivalence/,
# unread; the driver stops before any command if a shard, merged or report output exists.
#
# Run from the scratch copy, so that no file in the project tree changes:
#   cd ~/rds/hpc-work/a8_check && sbatch a8_equiv_rest_intr.sh
#
# Exit 0 when both steps ended with exit 0; 1 otherwise. The verdicts are the lines starting "EQUIVALENCE" in the step
# logs and the two equivalence_report.json files under results/a8/equivalence/.
set -euo pipefail
P=~/rds/hpc-work/a8_check
S=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
SRUN="${SRUN:-srun}"
module load miniconda || module load miniconda3 || true
export CONDA_ENVS_PATH="${ENV_ROOT}/conda_envs"
export CONDA_PKGS_DIRS="${ENV_ROOT}/conda_pkgs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mr_eu_open_llm
set -u
cd "$P"
mkdir -p logs
export PYTHONPATH="${S}/venvs/molmo2_cv411:${P}/src"
export HF_HOME="${S}/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM="false"
export A8_EQUIV_MODELS_DIR="${ENV_ROOT}/models"
unset MR_EU_HPC_MODELS_DIR || true
python research/scripts/decoder_preflight.py
python -c "import torch, transformers, cv2, numpy; print('versions', torch.__version__, transformers.__version__, cv2.__version__, numpy.__version__)"

CONDITIONS=("real" "mismatch")
pids=()
for c in "${CONDITIONS[@]}"; do
  log="logs/a8_equiv_rest_${SLURM_JOB_ID:-local}_${c}.out"
  echo "$(date -Is) start ${c}, step limit 59 min, log ${log}"
  "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --mem=120G --time=59 \
    python a8_equiv_driver_condition_rest.py "${c}" > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo "$(date -Is) ${CONDITIONS[$i]}: ended (exit 0)"
  else
    echo "$(date -Is) ${CONDITIONS[$i]}: FAILED or hit its cap (exit ${rc}); see its log" >&2
    status=1
  fi
done
grep -h "^EQUIVALENCE" logs/a8_equiv_rest_"${SLURM_JOB_ID:-local}"_*.out || true
exit "${status}"
