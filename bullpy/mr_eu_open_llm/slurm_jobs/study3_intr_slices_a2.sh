#!/usr/bin/env bash
#SBATCH -J s3_intr_a2
#SBATCH -A BARON-COHEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=01:00:00
#SBATCH -o logs/study3_intr_a2_%j.out
#SBATCH -e logs/study3_intr_a2_%j.err
#
# One-hour slices of A2's frozen, resumable full run under the `intr` QoS
# (research/DECISIONS.md, 2026-11-13, "A2's full-run shards may run in one-hour slices").
#
# Each slot runs one A2 shard through the unchanged job script (slurm_jobs/study3_stimulus_interventions.sh)
# with the full run's settings (STEP=run, N_SHARDS=3, CONFIRM_GATES_PASSED=1, the default inputs), as its
# own job step on one GPU, and `timeout` stops it before the job's limit. The runner skips cells already
# written (scripts/run_stimulus_interventions.py, done_cells), so the next slice resumes the shard.
#
# The runner skips an unparseable line but does not trim it, so before each slot starts, any line that does
# not parse is removed by A2's own repair rule (PREREG_A2, ruling A2-2): each removed line's number, byte
# offset, length and sha256 are logged to results/stimulus_interventions_torn/repairs.jsonl, the original
# file is kept there under a name that does not match cells_*_shard*.jsonl, and the cells are recomputed by
# the resumed run. No parseable row is edited or removed.
#
# Before a slot runs a shard, the gpu2 array task for that shard must be held, so that two runs never write
# one file at once. Whether a shard is complete is decided by its line count against the design (81 items,
# 29 cells each) and, finally, by A2's frozen pre-run check and analysis.
#
#   SLOTS="a2:gemma4:0 a2:gemma4:1 a2:gemma4:2 a2:qwen3vl:0" sbatch slurm_jobs/study3_intr_slices_a2.sh
#
# Exit 0 when every slot finished its shard or was stopped by the budget; 1 when any slot or repair failed.

set -euo pipefail

SLOTS="${SLOTS:-}"
if [ -z "${SLOTS}" ]; then
  echo "Set SLOTS, for example SLOTS=\"a2:gemma4:0 a2:qwen3vl:1\"." >&2
  exit 2
fi
BUDGET="${BUDGET:-52m}"
STEP_MEM="${STEP_MEM:-120G}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/rds/hpc-work/study3}"
SRUN="${SRUN:-srun}"
REPAIR_PY="${REPAIR_PY:-${HOME}/rds/hpc-work/study2/conda_envs/mr_eu_open_llm/bin/python}"

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
    a2) ;;
    *) echo "Slot ${slot}: this wrapper runs a2 slots only." >&2; exit 2 ;;
  esac
  case "${model}" in
    gemma4|qwen3vl) ;;
    *) echo "Slot ${slot}: model must be gemma4 or qwen3vl." >&2; exit 2 ;;
  esac
  if ! [[ "${shard}" =~ ^[0-2]$ ]]; then
    echo "Slot ${slot}: an A2 shard is 0 to 2." >&2
    exit 2
  fi
  case "${seen}" in
    *" ${model}:${shard} "*) echo "Slot ${slot} appears twice." >&2; exit 2 ;;
  esac
  seen="${seen}${model}:${shard} "
done

# A2-2 repair, before any slot starts: remove only unparseable lines, log them, keep the original.
for slot in "${slots[@]}"; do
  IFS=: read -r amendment model shard <<< "${slot}"
  cells="results/stimulus_interventions/cells_${model}_shard${shard}.jsonl"
  [ -f "${cells}" ] || continue
  if ! "${REPAIR_PY}" - "${cells}" "${SLURM_JOB_ID:-local}" <<'PY'
import hashlib, json, os, sys, time
path, job = sys.argv[1], sys.argv[2]
data = open(path, "rb").read()
keep, removed, offset = [], [], 0
for number, raw in enumerate(data.splitlines(keepends=True), start=1):
    ok = raw.endswith(b"\n")
    if ok:
        try:
            json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            ok = False
    if ok:
        keep.append(raw)
    else:
        removed.append({"line": number, "byte_offset": offset, "length": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    offset += len(raw)
if removed:
    folder = "results/stimulus_interventions_torn"
    os.makedirs(folder, exist_ok=True)
    original = os.path.join(folder, os.path.basename(path) + f".{job}.orig")
    with open(original, "wb") as handle:
        handle.write(data)
    with open(os.path.join(folder, "repairs.jsonl"), "a", encoding="utf-8") as log:
        log.write(json.dumps({"file": path, "job": job, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                              "original_kept_as": original, "original_sha256": hashlib.sha256(data).hexdigest(),
                              "removed": removed, "rule": "PREREG_A2 ruling A2-2"}) + "\n")
    tmp = path + f".repair.{job}"
    with open(tmp, "wb") as handle:
        handle.write(b"".join(keep))
    os.replace(tmp, path)
    print(f"A2-2 repair on {path}: removed {len(removed)} unparseable line(s); original kept as {original}")
PY
  then
    echo "$(date -Is) A2-2 repair FAILED on ${cells}; no slot started." >&2
    exit 1
  fi
done

pids=()
for slot in "${slots[@]}"; do
  IFS=: read -r amendment model shard <<< "${slot}"
  log="logs/study3_intr_a2_${SLURM_JOB_ID:-local}_${model}_${shard}.out"
  echo "$(date -Is) start ${slot}, budget ${BUDGET}, log ${log}"
  timeout --signal=TERM --kill-after=90s "${BUDGET}" \
    "${SRUN}" --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
      --mem="${STEP_MEM}" \
      env MODEL="${model}" STEP=run N_SHARDS=3 CONFIRM_GATES_PASSED=1 SLURM_ARRAY_TASK_ID="${shard}" \
      bash slurm_jobs/study3_stimulus_interventions.sh > "${log}" 2>&1 &
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
