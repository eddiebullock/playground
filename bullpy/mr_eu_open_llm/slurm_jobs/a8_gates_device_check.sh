#!/usr/bin/env bash
#SBATCH -J a8_gdev_check
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=05:00:00
#SBATCH -o logs/a8_gdev_check_%j.out
#SBATCH -e logs/a8_gdev_check_%j.err
#
# A8, dated deviation (DECISIONS 2026-11-17, row 102): the check of the device-aware gate hooks on the frozen
# Qwen3-VL-8B, before Qwen3-VL-32B's gates run through them. Three runs of the frozen gate command at
# bfloat16 start together, as steps of one whole-node job:
#   frozen_a, frozen_b: the frozen script (python -m scripts.vision_gates), one GPU each;
#   patched: the same arguments through research/scripts/a8_gates_device.py, on two GPUs.
# Then research/scripts/a8_gates_device_check.py compares them, on the CPU, by its rule, which was set before
# any run. The environment is the frozen gates job's (slurm_jobs/study3_vision_gates.sh), with its model
# directory, so --model qwen3vl loads the frozen 8B weights. Every step gets the same CPUs and memory, as the
# A8 equivalence job's steps did. Outputs go under results/a8/gates_device_check. The frozen gate file
# research/reproduction/vision_gates_qwen3vl.json is hashed before and after, and must not change.
#
#   BUDGET=240m sbatch slurm_jobs/a8_gates_device_check.sh
#
# Exit 0 only when all three runs ended with exit 0, the frozen gate file is unchanged and the check passed.

set -euo pipefail

BUDGET="${BUDGET:-240m}"
PROJECT_ROOT=~/rds/hpc-work/study3
ENV_ROOT=~/rds/hpc-work/study2
ENV_NAME="${ENV_NAME:-mr_eu_open_llm}"
SRUN="${SRUN:-srun}"
OUT=results/a8/gates_device_check
FROZEN_FILE=research/reproduction/vision_gates_qwen3vl.json

if ! [[ "${BUDGET}" =~ ^[1-9][0-9]*m$ ]]; then
  echo "Set BUDGET in minutes, for example BUDGET=240m." >&2
  exit 2
fi

# As slurm_jobs/study3_vision_gates.sh.
if [ -n "${VIRTUAL_ENV:-}" ]; then
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
set -u
cd "${PROJECT_ROOT}"
mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/venvs/molmo2_cv411:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export TOKENIZERS_PARALLELISM="false"
export MR_EU_HPC_MODELS_DIR="${ENV_ROOT}/models"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -e "${OUT}" ]; then
  echo "${OUT} exists; the check runs once." >&2
  exit 2
fi
# A8's code pins, as every A8 entry point checks them. They cover the hooks, this job and the check's rule.
python -c "import sys; sys.path.insert(0, 'research/scripts'); import a8_design; print(len(a8_design.check_own_code(require=True)), 'A8 code pins hold')"
python research/scripts/decoder_preflight.py
mkdir -p "${OUT}/frozen_a" "${OUT}/frozen_b" "${OUT}/patched"
before="$(sha256sum "${FROZEN_FILE}")"

# The frozen command's arguments (A8 design, commands.gates, for qwen3vl), at bfloat16 only.
ARGS=(--model qwen3vl --gates "g2,g3,g4" --dtypes bfloat16 --n_g3 12)
start() {
  local name="$1" gpus="$2"
  shift 2
  echo "$(date -Is) start ${name}, ${gpus} GPU(s), budget ${BUDGET}"
  timeout --signal=TERM --kill-after=90s "${BUDGET}" \
    "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:"${gpus}" \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --mem=120G \
      "$@" "${ARGS[@]}" --output "${OUT}/${name}/vision_gates_qwen3vl.json" --float32_max_frames 6 \
      > "logs/a8_gdev_check_${SLURM_JOB_ID:-local}_${name}.out" 2>&1
}
start frozen_a 1 python -m scripts.vision_gates & pid_a=$!
start frozen_b 1 python -m scripts.vision_gates & pid_b=$!
start patched 2 python research/scripts/a8_gates_device.py & pid_p=$!

status=0
for run in frozen_a:${pid_a} frozen_b:${pid_b} patched:${pid_p}; do
  rc=0
  wait "${run#*:}" || rc=$?
  echo "$(date -Is) ${run%%:*} ended (exit ${rc})"
  [ "${rc}" -eq 0 ] || status=1
done

after="$(sha256sum "${FROZEN_FILE}")"
if [ "${before}" != "${after}" ]; then
  echo "The frozen gate file ${FROZEN_FILE} changed." >&2
  exit 1
fi
if [ "${status}" -ne 0 ]; then
  echo "A run did not end with exit 0; the check is not run. See the runs' logs." >&2
  exit 1
fi
python research/scripts/a8_gates_device_check.py \
  --frozen "${OUT}/frozen_a/vision_gates_qwen3vl.json" "${OUT}/frozen_b/vision_gates_qwen3vl.json" \
  --patched "${OUT}/patched/vision_gates_qwen3vl.json" --report "${OUT}/report.json"
echo "$(date -Is) Done."
