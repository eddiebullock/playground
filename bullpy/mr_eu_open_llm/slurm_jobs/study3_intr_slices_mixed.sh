#!/usr/bin/env bash
#SBATCH -J s3_intr_mix
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH -o logs/study3_intr_mixed_%j.out
#SBATCH -e logs/study3_intr_mixed_%j.err
#
# One-hour slices under the `intr` QoS that run A2 shards and A9 shards in one job
# (research/DECISIONS.md, 2026-11-13, "A9's smoke and scored shards may run in one-hour slices beside A2's").
#
# a2 slots are handed, unchanged, to A2's own slice wrapper (slurm_jobs/study3_intr_slices_a2.sh, with its
# A2-2 repair), which runs as a child of this job. a9 and a9smoke slots run here, each as its own job step on
# one GPU, through the unchanged slurm_jobs/study3_a9.sh with the settings of the queued arrays:
#   a9:<model>:<shard>   MODEL, N_SHARDS=12, A9_FROZEN=1, the shard (0 to 11) as SLURM_ARRAY_TASK_ID
#   a9smoke:<model>:0    the same with shard 0 and MAX_CELLS=40: the timing smoke, which writes
#                        results/a9/cells_<model>_smoke_0.jsonl beside the scored cells
# `timeout` stops each step at 52 minutes. scripts/a9_run.py resumes a shard: it drops a partial last line
# and skips cells already written.
#
# Before a slot runs a shard, the gpu2 task for that shard must be held, so that two runs never write one
# file at once. Completeness is decided by each amendment's own pre-run check.
#
#   SLOTS="a2:gemma4:0 a2:qwen3vl:2 a9smoke:gemma4:0 a9:qwen3vl:3" \
#     sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4 slurm_jobs/study3_intr_slices_mixed.sh
#
# Exit 0 when every slot finished or was stopped by the budget; 1 when any slot or A2's wrapper failed.

set -euo pipefail

SLOTS="${SLOTS:-}"
if [ -z "${SLOTS}" ]; then
  echo "Set SLOTS, for example SLOTS=\"a2:gemma4:0 a9:qwen3vl:1\"." >&2
  exit 2
fi
BUDGET="${BUDGET:-52m}"
STEP_MEM="${STEP_MEM:-120G}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/rds/hpc-work/study3}"
SRUN="${SRUN:-srun}"
A2_WRAPPER="${A2_WRAPPER:-slurm_jobs/study3_intr_slices_a2.sh}"
export BUDGET STEP_MEM PROJECT_ROOT SRUN

cd "${PROJECT_ROOT}"
mkdir -p logs

read -r -a slots <<< "${SLOTS}"
if [ "${#slots[@]}" -eq 0 ] || [ "${#slots[@]}" -gt "${SLURM_NTASKS:-4}" ]; then
  echo "Need 1 to ${SLURM_NTASKS:-4} slots, got ${#slots[@]}." >&2
  exit 2
fi

seen=" "
a2_slots=()
a9_slots=()
for slot in "${slots[@]}"; do
  IFS=: read -r kind model shard extra <<< "${slot}"
  if [ -n "${extra:-}" ] || [ -z "${shard:-}" ]; then
    echo "Slot ${slot} is not kind:model:shard." >&2
    exit 2
  fi
  case "${model}" in
    gemma4|qwen3vl) ;;
    *) echo "Slot ${slot}: model must be gemma4 or qwen3vl." >&2; exit 2 ;;
  esac
  case "${kind}" in
    a2)
      if ! [[ "${shard}" =~ ^[0-2]$ ]]; then echo "Slot ${slot}: an A2 shard is 0 to 2." >&2; exit 2; fi
      a2_slots+=("${slot}") ;;
    a9)
      if ! [[ "${shard}" =~ ^([0-9]|1[01])$ ]]; then echo "Slot ${slot}: an A9 shard is 0 to 11." >&2; exit 2; fi
      a9_slots+=("${slot}") ;;
    a9smoke)
      if [ "${shard}" != "0" ]; then echo "Slot ${slot}: the A9 smoke runs shard 0 only." >&2; exit 2; fi
      a9_slots+=("${slot}") ;;
    *) echo "Slot ${slot}: kind must be a2, a9 or a9smoke." >&2; exit 2 ;;
  esac
  case "${seen}" in
    *" ${kind}:${model}:${shard} "*) echo "Slot ${slot} appears twice." >&2; exit 2 ;;
  esac
  seen="${seen}${kind}:${model}:${shard} "
done

a2_pid=""
if [ "${#a2_slots[@]}" -gt 0 ]; then
  echo "$(date -Is) A2 slots to ${A2_WRAPPER}: ${a2_slots[*]}"
  SLOTS="${a2_slots[*]}" bash "${A2_WRAPPER}" &
  a2_pid="$!"
fi

pids=()
for slot in "${a9_slots[@]+"${a9_slots[@]}"}"; do
  IFS=: read -r kind model shard <<< "${slot}"
  log="logs/study3_intr_mixed_${SLURM_JOB_ID:-local}_${kind}_${model}_${shard}.out"
  smoke=()
  if [ "${kind}" = "a9smoke" ]; then
    smoke=(MAX_CELLS=40)
  fi
  echo "$(date -Is) start ${slot}, budget ${BUDGET}, log ${log}"
  timeout --signal=TERM --kill-after=90s "${BUDGET}" \
    "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
      --mem="${STEP_MEM}" \
      env MODEL="${model}" N_SHARDS=12 A9_FROZEN=1 "${smoke[@]+"${smoke[@]}"}" SLURM_ARRAY_TASK_ID="${shard}" \
      bash slurm_jobs/study3_a9.sh > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  case "${rc}" in
    0) echo "$(date -Is) ${a9_slots[$i]}: the run ended by itself (exit 0)" ;;
    124|137) echo "$(date -Is) ${a9_slots[$i]}: stopped by the time budget (exit ${rc}); the next slice resumes it" ;;
    *) echo "$(date -Is) ${a9_slots[$i]}: FAILED (exit ${rc}); see its log" >&2; status=1 ;;
  esac
done
if [ -n "${a2_pid}" ]; then
  rc=0
  wait "${a2_pid}" || rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "$(date -Is) A2's wrapper FAILED (exit ${rc}); see its lines above and its slot logs" >&2
    status=1
  else
    echo "$(date -Is) A2's wrapper ended (exit 0)"
  fi
fi
exit "${status}"
