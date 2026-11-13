# A2 and A9 intr slices: plan, and unless DRY_RUN=1 (the default) submit, the next slice of
# slurm_jobs/study3_intr_slices_mixed.sh (research/DECISIONS.md, 2026-11-13: "A2's full-run shards may run in
# one-hour slices", "A4's and A2's slices may take up to three nodes", "A9's smoke and scored shards may run
# in one-hour slices beside A2's").
#
# Run from the Mac on the CSD3 login node:  ssh <host> 'DRY_RUN=0 bash -l -s' < research/scripts/mixed_intr_next_slice.sh
#
# Mechanics only: line counts, unparseable or partial lines, the count of cells by status, and the mean of the
# `seconds` field. No score field is read.
#
# Slots, in this order, four per node on up to MAX_NODES nodes (default 3, the intr QoS's per-user limit),
# never on more nodes than the ampere partition has idle (at least one):
#   1. A2's open shards, least advanced first. Each plans 81 items times 29 cells, 2,349. Failed cells are
#      allowed: A2 writes a cell it cannot edit as "failed", with its reason.
#   2. A9's timing smokes (results/a9/cells_<model>_smoke_0.jsonl), until each holds its 40 cells.
#   3. A9's open scored shards, partial ones first, for a model only when its smoke holds 40 cells, all "ok",
#      and the model is listed in results/a9/scored_slices_models.txt. The list is written by hand, after A4's
#      analysis, for the models eligible under RQ0 and A4's gates (A9 checklist step 4). Planned cells per
#      shard come from planned_cells(results/a9/manifest.csv, k, 12) in scripts/a9_run.py.
# It refuses to plan while any intr job of ours exists. It stops on an over-full shard, an A2 status other than
# ok or failed, an A9 cell that is not ok, or when only A9 scored shards are left and none may run. It leaves
# any shard whose gpu2 task is running to that task, holds the gpu2 task of each shard it slots, and cancels
# the held gpu2 task of each shard whose file is complete.
set -u
DRY_RUN="${DRY_RUN:-1}"
MAX_NODES="${MAX_NODES:-3}"
cd "${STUDY3_ROOT:-${HOME}/rds/hpc-work/study3}" || exit 3

if squeue -u eb2007 -q intr -h -o "%i" | grep -q .; then
  echo "An intr job of ours exists; not planning a new slice:"
  squeue -u eb2007 -q intr -h -o "%i %j %T %M"
  exit 0
fi

python3 - "$DRY_RUN" "$MAX_NODES" <<'PY'
import json, os, subprocess, sys

dry = sys.argv[1] == "1"
max_nodes = min(3, max(1, int(sys.argv[2])))
MODELS = ("gemma4", "qwen3vl")
A2_PLANNED = 81 * 29
A9_PLANNED = [3164, 2889, 2904, 2951, 3202, 2728, 2774, 3086, 3111, 3106, 3019, 3068]
SMOKE_CELLS = 40
A2_ARRAY = {"gemma4": "40468625", "qwen3vl": "40468626"}
A9_ARRAY = {"gemma4": "40863756", "qwen3vl": "40863757"}
SMOKE_JOB = {"gemma4": "40843619", "qwen3vl": "40843620"}
ALLOW = "results/a9/scored_slices_models.txt"

def sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True).stdout

def count(path):
    n = torn = 0
    status, secs = {}, []
    if not os.path.exists(path):
        return 0, 0, {}, float("nan")
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
            s = str(r.get("status"))
            status[s] = status.get(s, 0) + 1
            if isinstance(r.get("seconds"), (int, float)):
                secs.append(r["seconds"])
    return n, torn, status, (sum(secs) / len(secs) if secs else float("nan"))

def idle_nodes():
    # Nodes whose state is exactly idle or idle~ (powered down). sinfo -t idle also lists idle nodes
    # in a maintenance reservation as "maint"; those are not counted.
    total = 0
    for line in sh("sinfo -p ampere -h -t idle -o '%T %D'").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("idle", "idle~") and parts[1].isdigit():
            total += int(parts[1])
    return total

states = {}
for job in list(A2_ARRAY.values()) + list(SMOKE_JOB.values()) + list(A9_ARRAY.values()):
    for line in sh(f"squeue -r -h -j {job} -o '%i %T %r'").splitlines():
        tid, state, reason = (line.split(None, 2) + ["", ""])[:3]
        states[tid] = (state, reason)

allowed = set()
if os.path.exists(ALLOW):
    with open(ALLOW, encoding="utf-8") as f:
        allowed = {w.strip() for w in f if w.strip() in MODELS}

units = []
for m in MODELS:
    for k in range(3):
        units.append({"kind": "a2", "model": m, "shard": k, "planned": A2_PLANNED, "tid": f"{A2_ARRAY[m]}_{k}",
                      "path": f"results/stimulus_interventions/cells_{m}_shard{k}.jsonl"})
for m in MODELS:
    units.append({"kind": "a9smoke", "model": m, "shard": 0, "planned": SMOKE_CELLS, "tid": f"{SMOKE_JOB[m]}_0",
                  "path": f"results/a9/cells_{m}_smoke_0.jsonl"})
for m in MODELS:
    for k in range(12):
        units.append({"kind": "a9", "model": m, "shard": k, "planned": A9_PLANNED[k], "tid": f"{A9_ARRAY[m]}_{k}",
                      "path": f"results/a9/cells_{m}_{k}.jsonl"})

problems = []
smoke_ok = {}
for u in units:
    n, torn, status, mean_s = count(u["path"])
    u["n"], u["torn"] = n, torn
    u["state"] = states.get(u["tid"], ("GONE", ""))
    name = f"{u['kind']}_{u['model']}" + ("" if u["kind"] == "a9smoke" else f"_{u['shard']}")
    u["name"] = name
    shown = ", ".join(f"{s} {c}" for s, c in sorted(status.items())) or "none"
    print(f"{name}: {n}/{u['planned']} lines, partial {torn}, status {shown}, mean seconds {mean_s:.3f}, "
          f"gpu2 task {u['tid']} {u['state'][0]} {u['state'][1]}")
    if u["kind"] == "a2":
        other = sum(c for s, c in status.items() if s not in ("ok", "failed"))
        if other:
            problems.append(f"{name}: {other} cells with a status other than ok or failed")
    else:
        bad = sum(c for s, c in status.items() if s != "ok")
        if bad:
            problems.append(f"{name}: {bad} cells not ok")
    if u["kind"] == "a9smoke":
        u["complete"] = n >= SMOKE_CELLS
        smoke_ok[u["model"]] = u["complete"] and torn == 0 and not any(s != "ok" for s in status)
    else:
        if n > u["planned"]:
            problems.append(f"{name}: {n} lines, more than the {u['planned']} planned")
        u["complete"] = n == u["planned"] and torn == 0
    if not u["complete"] and u["state"][0] == "RUNNING":
        print(f"  {name}: its gpu2 task is running, so this shard is left to it")

if problems:
    print("STOP:", "; ".join(problems))
    sys.exit(4)

def open_unit(u):
    return not u["complete"] and u["state"][0] != "RUNNING"

a2_open = sorted([u for u in units if u["kind"] == "a2" and open_unit(u)],
                 key=lambda u: (u["n"], MODELS.index(u["model"]), u["shard"]))
smoke_open = [u for u in units if u["kind"] == "a9smoke" and open_unit(u)]
a9_waiting = [u for u in units if u["kind"] == "a9" and open_unit(u)]
a9_ready = [u for u in a9_waiting if u["model"] in allowed and smoke_ok.get(u["model"])]
a9_ready = [u for u in a9_ready if u["n"] > 0] + [u for u in a9_ready if u["n"] == 0]
eligible = a2_open + smoke_open + a9_ready
idle = idle_nodes()
nodes = max(1, min(max_nodes, -(-len(eligible) // 4), idle)) if eligible else 0
slots = eligible[:4 * nodes]
complete = [u for u in units if u["complete"]]
print("complete:", [u["name"] for u in complete])
print("A9 scored models allowed:", sorted(allowed) or "none", "; smokes complete and ok:",
      sorted(m for m, ok in smoke_ok.items() if ok) or "none")
print(f"idle ampere nodes: {idle}; nodes for this slice: {nodes} (at most {max_nodes})")
print("next slots:", " ".join(f"{u['kind']}:{u['model']}:{u['shard']}" for u in slots) or "none")
cancel = [u["tid"] for u in complete if u["state"][0] == "PENDING"]
hold = [u["tid"] for u in slots if u["state"][0] == "PENDING" and "JobHeldUser" not in u["state"][1]]
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
    if a9_waiting:
        print(f"STOP: only A9 scored shards are left, and none may run yet (models allowed in {ALLOW}: "
              f"{sorted(allowed) or 'none'}; smokes complete and ok: "
              f"{sorted(m for m, ok in smoke_ok.items() if ok) or 'none'}).")
        sys.exit(5)
    print("No shard left to slice.")
    sys.exit(0)
env = "SLOTS='" + " ".join(f"{u['kind']}:{u['model']}:{u['shard']}" for u in slots) + "'"
out = sh(f"{env} sbatch --parsable --nodes={nodes} --ntasks={4 * nodes} --ntasks-per-node=4 "
         "slurm_jobs/study3_intr_slices_mixed.sh 2>&1")
print("submitted:", out.strip())
PY
