#!/usr/bin/env bash
#SBATCH -J s3_intr
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH -o logs/study3_intr_%j.out
#SBATCH -e logs/study3_intr_%j.err
#
# One-hour slices of a frozen, resumable scored run under the `intr` QoS
# (research/DECISIONS.md, 2026-11-13, "A4's scored shards may run in one-hour slices").
#
# Each slot runs one shard through the unchanged job script (slurm_jobs/study3_a4.sh), as its own
# job step on one GPU, and `timeout` stops it before the job's limit. The runner skips cells already
# written and drops a partial last line, so the next slice resumes the shard where this one stopped.
# A slice that ends by the time budget leaves its shard incomplete. Whether a shard is complete is
# decided only by the amendment's own pre-run check (exact planned cells per shard file).
#
# Before a slot runs a shard, the gpu2 array task for that shard must be held, so that two runs never
# write one file at once.
#
#   SLOTS="a4:gemma4:0 a4:gemma4:1 a4:gemma4:2 a4:gemma4:3" sbatch slurm_jobs/study3_intr_slices.sh
#
# Exit 0 when every slot finished its shard or was stopped by the budget; 1 when any slot failed.

set -euo pipefail

SLOTS="${SLOTS:-}"
if [ -z "${SLOTS}" ]; then
  echo "Set SLOTS, for example SLOTS=\"a4:gemma4:0 a4:gemma4:1\"." >&2
  exit 2
fi
BUDGET="${BUDGET:-52m}"
STEP_MEM="${STEP_MEM:-120G}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/rds/hpc-work/study3}"
SRUN="${SRUN:-srun}"

cd "${PROJECT_ROOT}"
mkdir -p logs

read -r -a slots <<< "${SLOTS}"
if [ "${#slots[@]}" -eq 0 ] || [ "${#slots[@]}" -gt "${SLURM_NTASKS:-4}" ]; then
  echo "Need 1 to ${SLURM_NTASKS:-4} slots, got ${#slots[@]}." >&2
  exit 2
fi

seen=" "
for slot in "${slots[@]}"; do
  IFS=: read -r amendment model shard extra <<< "${slot}"
  if [ -n "${extra:-}" ] || [ -z "${shard:-}" ]; then
    echo "Slot ${slot} is not amendment:model:shard." >&2
    exit 2
  fi
  case "${amendment}" in
    a4) ;;
    *) echo "Slot ${slot}: only a4 is set up for slices." >&2; exit 2 ;;
  esac
  case "${model}" in
    gemma4|qwen3vl) ;;
    *) echo "Slot ${slot}: model must be gemma4 or qwen3vl." >&2; exit 2 ;;
  esac
  if ! [[ "${shard}" =~ ^([0-9]|1[01])$ ]]; then
    echo "Slot ${slot}: an A4 shard is 0 to 11." >&2
    exit 2
  fi
  case "${seen}" in
    *" ${model}:${shard} "*) echo "Slot ${slot} appears twice." >&2; exit 2 ;;
  esac
  seen="${seen}${model}:${shard} "
done

pids=()
for slot in "${slots[@]}"; do
  IFS=: read -r amendment model shard <<< "${slot}"
  log="logs/study3_intr_${SLURM_JOB_ID:-local}_${amendment}_${model}_${shard}.out"
  echo "$(date -Is) start ${slot}, budget ${BUDGET}, log ${log}"
  timeout --signal=TERM --kill-after=90s "${BUDGET}" \
    "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
      --mem="${STEP_MEM}" \
      env MODEL="${model}" N_SHARDS=12 A4_FROZEN=1 SLURM_ARRAY_TASK_ID="${shard}" \
      bash slurm_jobs/study3_a4.sh > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  case "${rc}" in
    0) echo "$(date -Is) ${slots[$i]}: the run ended by itself (exit 0)" ;;
    124|137) echo "$(date -Is) ${slots[$i]}: stopped by the time budget (exit ${rc}); the next slice resumes it" ;;
    *) echo "$(date -Is) ${slots[$i]}: FAILED (exit ${rc}); see its log" >&2; status=1 ;;
  esac
done
exit "${status}"
