#!/usr/bin/env bash
#SBATCH -J a8_equiv_intr
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH -o logs/a8_equiv_intr_%j.out
#SBATCH -e logs/a8_equiv_intr_%j.err
#
# A8's GPU equivalence test (PREREG_A8 Freeze checklist item 2) as one job under the `intr` QoS (DECISIONS 2026-11-14,
# scheduling only). It replaces the queued `gpu2` job 40399722 (`a8_equiv_job.sh` in the same folder), which is held
# while this runs. The environment below is that job's, line for line. The two conditions run at once, as two job steps
# on one GPU each, through `a8_equiv_driver_condition.py` (the queued job's driver restricted to one condition; same
# commands, order and outputs). Each step is capped at 59 minutes.
#
# Run from the scratch copy, so that no file in the project tree changes:
#   cd ~/rds/hpc-work/a8_check && sbatch a8_equiv_intr.sh
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
  log="logs/a8_equiv_intr_${SLURM_JOB_ID:-local}_${c}.out"
  echo "$(date -Is) start ${c}, step limit 59 min, log ${log}"
  "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --mem=120G --time=59 \
    python a8_equiv_driver_condition.py "${c}" > "${log}" 2>&1 &
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
grep -h "^EQUIVALENCE" logs/a8_equiv_intr_"${SLURM_JOB_ID:-local}"_*.out || true
exit "${status}"
