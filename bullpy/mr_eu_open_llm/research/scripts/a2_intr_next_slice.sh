# A2 intr slices: plan, and unless DRY_RUN=1 (the default) submit, the next slice
# (research/DECISIONS.md, 2026-11-13, "A2's full-run shards may run in one-hour slices").
#
# Run from the Mac on the CSD3 login node:  ssh <host> 'DRY_RUN=0 bash -l -s' < research/scripts/a2_intr_next_slice.sh
#
# Mechanics only: line counts, unparseable lines, the count of cells by status ("ok", "failed" and any other),
# and the mean of the `seconds` field. No score field is read. Failed cells are expected: A2 writes a cell that
# cannot be edited as "failed" with its reason, and the analysis needs them present. The script refuses to plan
# while any intr job of ours exists, stops on an over-full shard or a status other than ok or failed, leaves any
# shard whose gpu2 task is running to that task, holds the gpu2 task of each shard it slots, and cancels the held
# gpu2 task of each shard whose file is complete. Each shard plans 81 items (243 split three ways by index) times
# 29 cells (scripts/run_stimulus_interventions.py INPUTS: 20 + 1 + 7 + 1) = 2,349 cells.
set -u
DRY_RUN="${DRY_RUN:-1}"
cd ~/rds/hpc-work/study3 || exit 3

if squeue -u eb2007 -q intr -h -o "%i" | grep -q .; then
  echo "An intr job of ours exists; not planning a new slice:"
  squeue -u eb2007 -q intr -h -o "%i %j %T %M"
  exit 0
fi

python3 - "$DRY_RUN" <<'PY'
import json, os, subprocess, sys

dry = sys.argv[1] == "1"
PLANNED = 81 * 29
ARRAY = {"gemma4": "40468625", "qwen3vl": "40468626"}
ORDER = [(m, k) for m in ("gemma4", "qwen3vl") for k in range(3)]

def sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True).stdout

def count(model, k):
    path = f"results/stimulus_interventions/cells_{model}_shard{k}.jsonl"
    n = failed = other = torn = 0
    secs = []
    if not os.path.exists(path):
        return 0, 0, 0, 0, float("nan")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                torn += 1
                continue
            try:
                r = json.loads(line)
            except ValueError:
                torn += 1
                continue
            n += 1
            status = r.get("status")
            if status == "failed":
                failed += 1
            elif status != "ok":
                other += 1
            if isinstance(r.get("seconds"), (int, float)):
                secs.append(r["seconds"])
    return n, failed, other, torn, (sum(secs) / len(secs) if secs else float("nan"))

states = {}
for model, job in ARRAY.items():
    for line in sh(f"squeue -r -h -j {job} -o '%i %T %r'").splitlines():
        tid, state, reason = (line.split(None, 2) + ["", ""])[:3]
        states[tid] = (state, reason)

complete, open_shards, problems = [], [], []
for model, k in ORDER:
    n, failed, other, torn, mean_s = count(model, k)
    tid = f"{ARRAY[model]}_{k}"
    st = states.get(tid, ("GONE", ""))
    print(f"{model}_{k}: {n}/{PLANNED} lines, partial {torn}, failed {failed}, mean seconds {mean_s:.3f}, gpu2 task {tid} {st[0]} {st[1]}")
    if other:
        problems.append(f"{model}_{k}: {other} cells with a status other than ok or failed")
    if n > PLANNED:
        problems.append(f"{model}_{k}: {n} lines, more than the {PLANNED} planned")
    if n == PLANNED and torn == 0:
        complete.append((model, k, tid, st))
    elif st[0] == "RUNNING":
        print(f"  {model}_{k}: its gpu2 task is running, so this shard is left to it")
    else:
        open_shards.append((n, model, k, tid, st))

if problems:
    print("STOP:", "; ".join(problems))
    sys.exit(4)

open_shards.sort(key=lambda s: (s[0], ORDER.index((s[1], s[2]))))
slots = [(m, k, tid, st) for _, m, k, tid, st in open_shards][:4]
print("complete shards:", [f"{m}_{k}" for m, k, _, _ in complete])
print("next slots:", " ".join(f"a2:{m}:{k}" for m, k, _, _ in slots) or "none")
cancel = [tid for _, _, tid, st in complete if st[0] == "PENDING"]
hold = [tid for _, _, tid, st in slots if st[0] == "PENDING" and "JobHeldUser" not in st[1]]
print("to cancel (held, shard complete):", cancel or "none")
print("to hold before the slice:", hold or "none")
if dry:
    print("DRY RUN: nothing changed.")
    sys.exit(0)
if cancel:
    print(sh("scancel " + " ".join(cancel) + " && echo cancelled"))
if hold:
    print(sh("scontrol hold " + " ".join(hold) + " && echo held"))
if not slots:
    print("No shard left to slice.")
    sys.exit(0)
env = "SLOTS='" + " ".join(f"a2:{m}:{k}" for m, k, _, _ in slots) + "'"
out = sh(f"{env} sbatch --parsable slurm_jobs/study3_intr_slices_a2.sh 2>&1")
print("submitted:", out.strip())
PY
