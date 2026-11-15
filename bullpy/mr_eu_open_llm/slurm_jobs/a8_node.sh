#!/usr/bin/env bash
#SBATCH -J a8_node
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH -o logs/a8_node_%j.out
#SBATCH -e logs/a8_node_%j.err
#
# Up to four A8 launcher steps in one whole-node job, each as its own job step on its own GPU, through the
# unchanged slurm_jobs/a8_job.sh. That file sets up the step's environment and calls
# research/scripts/a8_launch.py exactly as a single-GPU A8 job does, so the launcher's checks, commands,
# outputs and records are the same. Only the number of launcher steps that share one Slurm job changes.
# Used only once a dated note says so.
#
# Why: on `ampere`, jobs that ask for every GPU of a node have started within minutes, while jobs that ask
# for one or two have waited for days (DECISIONS 2026-11-14, "A8's GPU equivalence test asks for a whole
# node's four GPUs").
#
# A slot is one launcher step, written as the a8_job.sh variables it sets, comma-separated:
#   STEP=gates,A8_ID=qwen3vl_32b
#   STEP=a1_shard,A8_ID=qwen3vl_32b,ROTATION=0,CONDITION=real,N_SHARDS=4,SHARD=0
#   STEP=a2_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=3
# SHARD reaches a8_job.sh as SLURM_ARRAY_TASK_ID, as an array task would pass it (the A4 slices do the
# same, DECISIONS 2026-11-13). CPU steps (a1_merge, natural_rq1, a4_analyse) do not run here.
#
# GPUS_PER_SLOT (default 1) is every slot's visible GPU count; A8's design rule gpu_settings asks for the
# same count across the shards of a rotation and condition, and across real and mismatch. The slots times
# GPUS_PER_SLOT must be at most 4.
#
# Host memory per slot is STEP_MEM (default 120G), or the slot's own MEM key, which is a run setting and
# does not reach a8_job.sh (for example MEM=240G for the gates of a large model, whose float32 stage the
# frozen loader may place partly in host memory). The slots' memory together must fit the job's memory.
#
# BUDGET (required, in minutes) is how long each slot may run, through `timeout`; it must end before the
# job's limit. A resumable step (a3, a2_run, a4_run) stopped by the budget resumes in a later job. A
# non-resumable step (gates, delivery, ladder, a1_shard) stopped by the budget is a failed attempt, and
# two failed attempts make a Phase A step "not tested" (design rule not_tested_after_retry), so its
# budget is set from its timing with a wide margin.
#
#   SLOTS="STEP=gates,A8_ID=qwen3vl_32b STEP=gates,A8_ID=gemma4_31b" BUDGET=700m sbatch slurm_jobs/a8_node.sh
#   intr:  SLOTS="..." BUDGET=52m sbatch --qos=intr --time=01:00:00 slurm_jobs/a8_node.sh
#
# Exit 0 when every slot ended with exit 0, or was a resumable step stopped by the budget; 1 otherwise.
# Each slot's outcome is in this job's log, and each slot's output in logs/a8_node_<job>_<slot>.out.

set -euo pipefail

SLOTS="${SLOTS:-}"
BUDGET="${BUDGET:-}"
GPUS_PER_SLOT="${GPUS_PER_SLOT:-1}"
STEP_MEM="${STEP_MEM:-120G}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/rds/hpc-work/study3}"
SRUN="${SRUN:-srun}"
NODE_GPUS=4

if [ -z "${SLOTS}" ]; then
  echo "Set SLOTS, one launcher step per slot, for example SLOTS=\"STEP=gates,A8_ID=qwen3vl_32b\"." >&2
  exit 2
fi
if ! [[ "${BUDGET}" =~ ^[1-9][0-9]*m$ ]]; then
  echo "Set BUDGET in minutes, for example BUDGET=52m or BUDGET=700m." >&2
  exit 2
fi
if ! [[ "${GPUS_PER_SLOT}" =~ ^[1-4]$ ]]; then
  echo "GPUS_PER_SLOT is 1 to 4." >&2
  exit 2
fi
if ! [[ "${STEP_MEM}" =~ ^[1-9][0-9]*G$ ]]; then
  echo "STEP_MEM is a whole number of gigabytes, for example 120G." >&2
  exit 2
fi
# The job's memory in GB: Slurm gives SLURM_MEM_PER_NODE in MB; 500G is this file's request.
JOB_MEM_G=$(( ${SLURM_MEM_PER_NODE:-512000} / 1024 ))

read -r -a slots <<< "${SLOTS}"
if [ "${#slots[@]}" -eq 0 ] || [ $(( ${#slots[@]} * GPUS_PER_SLOT )) -gt "${NODE_GPUS}" ]; then
  echo "Need 1 to $(( NODE_GPUS / GPUS_PER_SLOT )) slots at ${GPUS_PER_SLOT} GPU(s) each, got ${#slots[@]}." >&2
  exit 2
fi

is_count() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_index() { [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]; }
resumable() { case "$1" in a3|a2_run|a4_run) return 0 ;; *) return 1 ;; esac; }

# Parses one slot into the variables below; refuses anything A8's launcher would not take from a8_job.sh.
parse_slot() {
  local slot="$1" pair key value
  step="" a8_id="" rotation="" condition="" n_shards="" shard="" mem="${STEP_MEM}"
  IFS=, read -r -a pairs <<< "${slot}"
  for pair in "${pairs[@]}"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    if [ "${key}" = "${pair}" ] || [ -z "${value}" ]; then
      echo "Slot ${slot}: ${pair} is not KEY=value." >&2
      return 2
    fi
    case "${key}" in
      STEP) step="${value}" ;;
      A8_ID) a8_id="${value}" ;;
      ROTATION) rotation="${value}" ;;
      CONDITION) condition="${value}" ;;
      N_SHARDS) n_shards="${value}" ;;
      SHARD) shard="${value}" ;;
      MEM) mem="${value}" ;;
      *) echo "Slot ${slot}: unknown key ${key}." >&2; return 2 ;;
    esac
  done
  if ! [[ "${mem}" =~ ^[1-9][0-9]*G$ ]]; then
    echo "Slot ${slot}: MEM is a whole number of gigabytes, for example 240G." >&2
    return 2
  fi
  case "${a8_id}" in
    qwen3vl_32b|gemma4_31b) ;;
    *) echo "Slot ${slot}: A8_ID must be qwen3vl_32b or gemma4_31b." >&2; return 2 ;;
  esac
  case "${step}" in
    gates|delivery|ladder|a3)
      if [ -n "${rotation}${condition}${n_shards}${shard}" ]; then
        echo "Slot ${slot}: ${step} takes no ROTATION, CONDITION, N_SHARDS or SHARD." >&2
        return 2
      fi ;;
    a1_shard)
      if ! [[ "${rotation}" =~ ^[0-4]$ ]] || ! [[ "${condition}" =~ ^(real|mismatch)$ ]]; then
        echo "Slot ${slot}: a1_shard needs ROTATION 0 to 4 and CONDITION real or mismatch." >&2
        return 2
      fi ;;
    a2_run|a4_run)
      if [ -n "${rotation}${condition}" ]; then
        echo "Slot ${slot}: ${step} takes no ROTATION or CONDITION." >&2
        return 2
      fi ;;
    *) echo "Slot ${slot}: STEP must be a GPU step: gates, delivery, ladder, a3, a1_shard, a2_run or a4_run." >&2
       return 2 ;;
  esac
  case "${step}" in
    a1_shard|a2_run|a4_run)
      if ! is_count "${n_shards}" || ! is_index "${shard}" || [ "${shard}" -ge "${n_shards}" ]; then
        echo "Slot ${slot}: ${step} needs N_SHARDS of 1 or more and SHARD from 0 to N_SHARDS - 1." >&2
        return 2
      fi ;;
  esac
  return 0
}

seen=" "
total_mem_g=0
for slot in "${slots[@]}"; do
  parse_slot "${slot}" || exit 2
  identity="${step}:${a8_id}:${rotation}:${condition}:${shard}"
  case "${seen}" in
    *" ${identity} "*) echo "Slot ${slot} repeats a step already in this job." >&2; exit 2 ;;
  esac
  seen="${seen}${identity} "
  total_mem_g=$(( total_mem_g + ${mem%G} ))
done
if [ "${total_mem_g}" -gt "${JOB_MEM_G}" ]; then
  echo "The slots ask for ${total_mem_g}G of host memory together; the job has ${JOB_MEM_G}G." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p logs

pids=()
for i in "${!slots[@]}"; do
  parse_slot "${slots[$i]}"
  vars=(STEP="${step}" A8_ID="${a8_id}")
  [ -n "${rotation}" ] && vars+=(ROTATION="${rotation}")
  [ -n "${condition}" ] && vars+=(CONDITION="${condition}")
  [ -n "${n_shards}" ] && vars+=(N_SHARDS="${n_shards}")
  [ -n "${shard}" ] && vars+=(SLURM_ARRAY_TASK_ID="${shard}")
  log="logs/a8_node_${SLURM_JOB_ID:-local}_${i}.out"
  echo "$(date -Is) start slot ${i} (${slots[$i]}), ${GPUS_PER_SLOT} GPU(s), ${mem}, budget ${BUDGET}, log ${log}"
  timeout --signal=TERM --kill-after=90s "${BUDGET}" \
    "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:"${GPUS_PER_SLOT}" \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --mem="${mem}" \
      env "${vars[@]}" bash slurm_jobs/a8_job.sh > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  rc=0
  wait "${pids[$i]}" || rc=$?
  parse_slot "${slots[$i]}"
  case "${rc}" in
    0) echo "$(date -Is) slot ${i} (${slots[$i]}): ended (exit 0)" ;;
    124|137)
      if resumable "${step}"; then
        echo "$(date -Is) slot ${i} (${slots[$i]}): stopped by the budget (exit ${rc}); a later job resumes it"
      else
        echo "$(date -Is) slot ${i} (${slots[$i]}): stopped by the budget (exit ${rc}); a failed attempt of a step that does not resume" >&2
        status=1
      fi ;;
    *) echo "$(date -Is) slot ${i} (${slots[$i]}): FAILED (exit ${rc}); see its log" >&2; status=1 ;;
  esac
done
exit "${status}"
