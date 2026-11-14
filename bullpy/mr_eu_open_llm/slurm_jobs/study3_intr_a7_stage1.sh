#!/usr/bin/env bash
#SBATCH -J s3_intr_a7
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH -o logs/study3_intr_a7_%j.out
#SBATCH -e logs/study3_intr_a7_%j.err
#
# A7 Stage 1's route arm as one whole job under the `intr` QoS (research/DECISIONS.md, 2026-11-14, "A7 Stage 1
# runs as one whole intr job"; the user's choice, NEEDS_INPUT 12).
#
# The four tasks of the queued arrays (Gemma 4 40795186, Qwen3-VL 40795187; shards 0 and 1 of each) run at once, as
# four job steps on one GPU each, through the job script named by JOB_SCRIPT (below) with the arrays' settings:
#   A7_FROZEN=1 STEP=run MODEL=<model> ARMS=route DECISION_DATE=2026-11-10 N_SHARDS=2, the shard as SLURM_ARRAY_TASK_ID.
# Each step keeps the arrays' 59-minute cap (srun --time=59). Nothing is sliced: a task runs once, from the start,
# as it would have on gpu2. The arrays' gpu2 tasks must be held before this job runs, so that no task runs twice.
#
#   sbatch slurm_jobs/study3_intr_a7_stage1.sh
#
# JOB_SCRIPT (default slurm_jobs/study3_a7.sh) names the job script each step runs. Since 2026-11-14 the rerun uses
# slurm_jobs/study3_a7_dataroot.sh, the pinned script plus --data_root data/eu_emotions (DECISIONS 2026-11-14,
# "A7 Stage 1 reruns with the clips' data root after a path fault").
#
# Exit 0 when every step ended with exit 0; 1 otherwise (a step that hits its cap, or fails, is reported by name).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/rds/hpc-work/study3}"
SRUN="${SRUN:-srun}"
STEP_TIME="${STEP_TIME:-59}"
STEP_MEM="${STEP_MEM:-120G}"
JOB_SCRIPT="${JOB_SCRIPT:-slurm_jobs/study3_a7.sh}"
TASKS=("gemma4:0" "gemma4:1" "qwen3vl:0" "qwen3vl:1")

cd "${PROJECT_ROOT}"
mkdir -p logs

pids=()
for task in "${TASKS[@]}"; do
  IFS=: read -r model shard <<< "${task}"
  log="logs/study3_intr_a7_${SLURM_JOB_ID:-local}_${model}_${shard}.out"
  echo "$(date -Is) start a7:${model}:${shard}, step limit ${STEP_TIME} min, job script ${JOB_SCRIPT}, log ${log}"
  "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --mem="${STEP_MEM}" --time="${STEP_TIME}" \
    env A7_FROZEN=1 STEP=run MODEL="${model}" ARMS=route DECISION_DATE=2026-11-10 N_SHARDS=2 \
        SLURM_ARRAY_TASK_ID="${shard}" \
    bash "${JOB_SCRIPT}" > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo "$(date -Is) a7:${TASKS[$i]}: ended (exit 0)"
  else
    echo "$(date -Is) a7:${TASKS[$i]}: FAILED or hit its cap (exit ${rc}); see its log" >&2
    status=1
  fi
done
exit "${status}"
