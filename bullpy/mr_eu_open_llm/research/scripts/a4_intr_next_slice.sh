# A4 intr slices: plan, and unless DRY_RUN=1 (the default) submit, the next slice
# (research/DECISIONS.md, 2026-11-13, "A4's scored shards may run in one-hour slices").
#
# Run from the Mac on the CSD3 login node:  ssh <host> 'DRY_RUN=0 bash -l -s' < research/scripts/a4_intr_next_slice.sh
#
# Mechanics only: line counts, the count of cells whose status is not "ok", partial lines and the mean of
# the `seconds` field. No score field is read. It refuses to plan while any intr job of ours exists, stops
# on a non-ok cell or an over-full shard, never slots a shard whose gpu2 task is running, holds the gpu2
# task of each shard it slots, and cancels the held gpu2 task of each shard whose file is complete.
# Planned cells per shard come from planned_cells(results/a4/manifest.csv, k, 12), the same for both models.
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
PLANNED = [6281, 6407, 6603, 6847, 6899, 6290, 6457, 6715, 6618, 6523, 6841, 6465]
ARRAY = {"gemma4": "40589226", "qwen3vl": "40593824"}
ORDER = [(m, k) for m in ("gemma4", "qwen3vl") for k in range(12)]

def sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True).stdout

def count(model, k):
    path = f"results/a4/cells_{model}_{k}.jsonl"
    n = bad = partial = 0
    secs = []
    if not os.path.exists(path):
        return 0, 0, 0, float("nan")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                partial += 1
                continue
            try:
                r = json.loads(line)
            except ValueError:
                partial += 1
                continue
            n += 1
            if r.get("status") != "ok":
                bad += 1
            if isinstance(r.get("seconds"), (int, float)):
                secs.append(r["seconds"])
    return n, bad, partial, (sum(secs) / len(secs) if secs else float("nan"))

# Task states of the gpu2 arrays, one line per task.
states = {}
for model, job in ARRAY.items():
    for line in sh(f"squeue -r -h -j {job} -o '%i %T %r'").splitlines():
        tid, state, reason = (line.split(None, 2) + ["", ""])[:3]
        states[tid] = (state, reason)

complete, partial_shards, fresh, problems = [], [], [], []
for model, k in ORDER:
    n, bad, part, mean_s = count(model, k)
    if bad:
        problems.append(f"{model}_{k}: {bad} cells not ok")
    if n > PLANNED[k]:
        problems.append(f"{model}_{k}: {n} lines, more than the {PLANNED[k]} planned")
    tid = f"{ARRAY[model]}_{k}"
    st = states.get(tid, ("GONE", ""))
    print(f"{model}_{k}: {n}/{PLANNED[k]} lines, partial {part}, mean seconds {mean_s:.3f}, gpu2 task {tid} {st[0]} {st[1]}")
    if n == PLANNED[k]:
        complete.append((model, k, tid, st))
    elif st[0] == "RUNNING":
        problems.append(f"{model}_{k}: its gpu2 task is RUNNING; leave this shard to it")
    elif n > 0:
        partial_shards.append((model, k, tid, st))
    else:
        fresh.append((model, k, tid, st))

if problems:
    print("STOP:", "; ".join(problems))
    sys.exit(4)

slots = [s for s in partial_shards + fresh if s[3][0] != "RUNNING"][:4]
print("complete shards:", [f"{m}_{k}" for m, k, _, _ in complete])
print("next slots:", " ".join(f"a4:{m}:{k}" for m, k, _, _ in slots) or "none")
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
env = "SLOTS='" + " ".join(f"a4:{m}:{k}" for m, k, _, _ in slots) + "'"
out = sh(f"{env} sbatch --parsable slurm_jobs/study3_intr_slices.sh 2>&1")
print("submitted:", out.strip())
PY
