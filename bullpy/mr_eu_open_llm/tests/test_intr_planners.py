"""The `intr` slice planners, run with stand-ins for the Slurm commands and fake shard files.

research/scripts/a4_intr_next_slice.sh and a2_intr_next_slice.sh plan, and unless DRY_RUN=1 submit, the next
one-hour slice of A4's and A2's resumable runs (DECISIONS 2026-11-13: "A4's scored shards may run in one-hour
slices", "A2's full-run shards may run in one-hour slices", "A4's and A2's slices may take up to three nodes").
The planners read mechanics only (line counts, status counts, partial lines, mean seconds), never a score field.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
A4 = ROOT / "research/scripts/a4_intr_next_slice.sh"
A2 = ROOT / "research/scripts/a2_intr_next_slice.sh"
A4_PLANNED = [6281, 6407, 6603, 6847, 6899, 6290, 6457, 6715, 6618, 6523, 6841, 6465]
A2_PLANNED = 81 * 29

FAKES = {
    "squeue": """#!/usr/bin/env bash
case "$*" in
  *"-q intr"*) printf '%s' "${FAKE_INTR:-}" ;;
  *"-j 40589226"*|*"-j 40468625"*) printf '%s' "${FAKE_G:-}" ;;
  *"-j 40593824"*|*"-j 40468626"*) printf '%s' "${FAKE_Q:-}" ;;
esac
""",
    "sinfo": """#!/usr/bin/env bash
printf '%s' "${FAKE_SINFO:-}"
""",
    "sbatch": """#!/usr/bin/env bash
echo "sbatch SLOTS=[${SLOTS:-}] $*" >> "$CALLS"
echo 99999
""",
    "scancel": """#!/usr/bin/env bash
echo "scancel $*" >> "$CALLS"
""",
    "scontrol": """#!/usr/bin/env bash
echo "scontrol $*" >> "$CALLS"
""",
}


def tasks(array, ks, state, reason):
    return "".join(f"{array}_{k} {state} {reason}\n" for k in ks)


@pytest.fixture
def sim(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    root = tmp_path / "study3"
    root.mkdir()
    calls = tmp_path / "calls.log"

    class Sim:
        def __init__(self):
            self.root = root

        def run(self, script, **env):
            calls.write_text("")
            full = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path),
                    "STUDY3_ROOT": str(root), "CALLS": str(calls)}
            full.update({k: str(v) for k, v in env.items()})
            proc = subprocess.run(["bash", str(script)], env=full, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
            return proc.stdout, calls.read_text().splitlines()

    return Sim()


def write_a4(root, gemma, qwen):
    folder = root / "results/a4"
    folder.mkdir(parents=True, exist_ok=True)
    for model, counts in (("gemma4", gemma), ("qwen3vl", qwen)):
        for k, n in enumerate(counts):
            path = folder / f"cells_{model}_{k}.jsonl"
            if path.exists():
                path.unlink()
            if n:
                path.write_text('{"status": "ok", "seconds": 0.6}\n' * n)


def write_a2(root, counts):
    folder = root / "results/stimulus_interventions"
    folder.mkdir(parents=True, exist_ok=True)
    i = 0
    for model in ("gemma4", "qwen3vl"):
        for k in range(3):
            path = folder / f"cells_{model}_shard{k}.jsonl"
            if path.exists():
                path.unlink()
            n = counts[i]
            if n:
                rows = ['{"status": "failed", "seconds": 1}\n' if j % 10 == 0 else '{"status": "ok", "seconds": 1}\n'
                        for j in range(n)]
                path.write_text("".join(rows))
            i += 1


def sbatch_of(calls):
    lines = [c for c in calls if c.startswith("sbatch ")]
    assert len(lines) == 1, calls
    slots = re.search(r"SLOTS=\[(.*?)\]", lines[0]).group(1)
    nodes = re.search(r"--nodes=(\d+) --ntasks=(\d+) --ntasks-per-node=4", lines[0])
    assert nodes, lines[0]
    return slots.split(), int(nodes.group(1)), int(nodes.group(2)), lines[0]


def chain_total(out):
    # The Mac loop's awk: sum the counts before "/" on every "<shard>: N/P lines, partial" line.
    return sum(int(m.group(1)) for m in re.finditer(r"^\S+: (\d+)/\d+ lines, partial ", out, re.M))


GEMMA_8_11_PARTIAL = A4_PLANNED[:8] + [4500, 4400, 4600, 4300]
NO_QWEN = [0] * 12


def test_a4_fills_three_idle_nodes_partial_shards_first(sim):
    write_a4(sim.root, GEMMA_8_11_PARTIAL, NO_QWEN)
    out, calls = sim.run(A4, DRY_RUN=0, FAKE_G=tasks(40589226, range(8, 12), "PENDING", "JobHeldUser"),
                         FAKE_Q=tasks(40593824, range(12), "PENDING", "Priority"),
                         FAKE_SINFO="maint 17\nidle~ 7\nidle 2\n")
    slots, nodes, ntasks, line = sbatch_of(calls)
    assert (nodes, ntasks) == (3, 12)
    assert slots == [f"a4:gemma4:{k}" for k in range(8, 12)] + [f"a4:qwen3vl:{k}" for k in range(8)]
    assert line.endswith("slurm_jobs/study3_intr_slices.sh")
    assert [c for c in calls if c.startswith("scontrol")] == [
        "scontrol hold " + " ".join(f"40593824_{k}" for k in range(8))]
    assert not [c for c in calls if c.startswith("scancel")]
    assert "idle ampere nodes: 9; nodes for this slice: 3 (at most 3)" in out
    assert re.search(r"^submitted: 99999$", out, re.M)


@pytest.mark.parametrize("sinfo, max_nodes, want", [
    ("maint 17\nidle 1\n", None, 1),   # one idle node
    ("maint 17\n", None, 1),           # none idle: still one node, which waits as before
    ("idle 2\n", None, 2),
    ("idle 9\n", "1", 1),              # MAX_NODES=1 restores one-node slices
    ("idle 9\n", "5", 3),              # capped at the QoS's 3 nodes
    ("idle% 4\nidle# 2\nidle 1\n", None, 1),  # powering down or up is not idle
])
def test_a4_node_count(sim, sinfo, max_nodes, want):
    write_a4(sim.root, GEMMA_8_11_PARTIAL, NO_QWEN)
    env = dict(DRY_RUN=0, FAKE_G=tasks(40589226, range(8, 12), "PENDING", "JobHeldUser"),
               FAKE_Q=tasks(40593824, range(12), "PENDING", "Priority"), FAKE_SINFO=sinfo)
    if max_nodes is not None:
        env["MAX_NODES"] = max_nodes
    out, calls = sim.run(A4, **env)
    slots, nodes, ntasks, _ = sbatch_of(calls)
    assert (nodes, ntasks) == (want, 4 * want)
    assert len(slots) == 4 * want
    assert slots[:4] == [f"a4:gemma4:{k}" for k in range(8, 12)]


def test_a4_last_shards_take_only_the_nodes_they_need_and_complete_shards_are_cancelled(sim):
    write_a4(sim.root, A4_PLANNED, A4_PLANNED[:7] + [3000] * 5)
    out, calls = sim.run(A4, DRY_RUN=0, FAKE_G="", FAKE_Q=tasks(40593824, range(12), "PENDING", "JobHeldUser"),
                         FAKE_SINFO="idle 9\n")
    slots, nodes, ntasks, _ = sbatch_of(calls)
    assert (nodes, ntasks) == (2, 8)
    assert slots == [f"a4:qwen3vl:{k}" for k in range(7, 12)]
    assert [c for c in calls if c.startswith("scancel")] == [
        "scancel " + " ".join(f"40593824_{k}" for k in range(7))]
    assert not [c for c in calls if c.startswith("scontrol")]


def test_a4_done_when_every_shard_is_complete(sim):
    write_a4(sim.root, A4_PLANNED, A4_PLANNED)
    out, calls = sim.run(A4, DRY_RUN=0, FAKE_G="", FAKE_Q="", FAKE_SINFO="idle 9\n")
    assert re.search(r"^No shard left to slice\.$", out, re.M)
    assert not [c for c in calls if c.startswith("sbatch")]
    assert chain_total(out) == 2 * sum(A4_PLANNED)


def test_a4_refuses_while_an_intr_job_exists(sim):
    write_a4(sim.root, GEMMA_8_11_PARTIAL, NO_QWEN)
    out, calls = sim.run(A4, DRY_RUN=0, FAKE_INTR="12345\n", FAKE_SINFO="idle 9\n")
    assert "An intr job of ours exists" in out
    assert calls == []


def test_a4_dry_run_changes_nothing_and_the_chain_total_parses(sim):
    write_a4(sim.root, GEMMA_8_11_PARTIAL, NO_QWEN)
    out, calls = sim.run(A4, DRY_RUN=1, FAKE_G=tasks(40589226, range(8, 12), "PENDING", "JobHeldUser"),
                         FAKE_Q=tasks(40593824, range(12), "PENDING", "Priority"), FAKE_SINFO="idle 9\n")
    assert calls == []
    assert re.search(r"^DRY RUN: nothing changed\.$", out, re.M)
    assert chain_total(out) == sum(GEMMA_8_11_PARTIAL)


def test_a4_leaves_a_shard_to_its_running_gpu2_task(sim):
    write_a4(sim.root, GEMMA_8_11_PARTIAL, NO_QWEN)
    out, calls = sim.run(A4, DRY_RUN=0,
                         FAKE_G=tasks(40589226, [8], "RUNNING", "None") + tasks(40589226, range(9, 12), "PENDING", "JobHeldUser"),
                         FAKE_Q=tasks(40593824, range(12), "PENDING", "Priority"), FAKE_SINFO="idle 9\n")
    slots, nodes, _, _ = sbatch_of(calls)
    assert "a4:gemma4:8" not in slots
    assert nodes == 3 and len(slots) == 12


def test_a4_stops_on_a_cell_that_is_not_ok(sim):
    write_a4(sim.root, GEMMA_8_11_PARTIAL, NO_QWEN)
    with open(sim.root / "results/a4/cells_gemma4_8.jsonl", "a") as handle:
        handle.write('{"status": "error", "seconds": 1}\n')
    out, calls = sim.run(A4, DRY_RUN=0, FAKE_G="", FAKE_Q="", FAKE_SINFO="idle 9\n")
    assert re.search(r"^STOP:", out, re.M)
    assert not [c for c in calls if c.startswith("sbatch")]


def test_a2_runs_all_six_shards_on_two_nodes(sim):
    write_a2(sim.root, [0] * 6)
    out, calls = sim.run(A2, DRY_RUN=0, FAKE_G=tasks(40468625, range(3), "PENDING", "Priority"),
                         FAKE_Q=tasks(40468626, range(3), "PENDING", "Priority"),
                         FAKE_SINFO="maint 17\nidle~ 7\nidle 2\n")
    slots, nodes, ntasks, line = sbatch_of(calls)
    assert (nodes, ntasks) == (2, 8)
    assert slots == [f"a2:gemma4:{k}" for k in range(3)] + [f"a2:qwen3vl:{k}" for k in range(3)]
    assert line.endswith("slurm_jobs/study3_intr_slices_a2.sh")
    assert [c for c in calls if c.startswith("scontrol")] == [
        "scontrol hold " + " ".join([f"40468625_{k}" for k in range(3)] + [f"40468626_{k}" for k in range(3)])]


def test_a2_one_idle_node_takes_the_four_least_advanced(sim):
    write_a2(sim.root, [0] * 6)
    out, calls = sim.run(A2, DRY_RUN=0, FAKE_G=tasks(40468625, range(3), "PENDING", "Priority"),
                         FAKE_Q=tasks(40468626, range(3), "PENDING", "Priority"), FAKE_SINFO="idle 1\n")
    slots, nodes, ntasks, _ = sbatch_of(calls)
    assert (nodes, ntasks) == (1, 4)
    assert slots == ["a2:gemma4:0", "a2:gemma4:1", "a2:gemma4:2", "a2:qwen3vl:0"]


def test_a2_failed_cells_are_counted_not_stopped_and_complete_shards_cancelled(sim):
    write_a2(sim.root, [A2_PLANNED, A2_PLANNED, 1000, A2_PLANNED, 500, 0])
    out, calls = sim.run(A2, DRY_RUN=0, FAKE_G=tasks(40468625, range(3), "PENDING", "JobHeldUser"),
                         FAKE_Q=tasks(40468626, range(3), "PENDING", "JobHeldUser"), FAKE_SINFO="idle 9\n")
    assert not re.search(r"^STOP:", out, re.M)
    slots, nodes, _, _ = sbatch_of(calls)
    assert nodes == 1
    assert slots == ["a2:qwen3vl:2", "a2:qwen3vl:1", "a2:gemma4:2"]
    assert [c for c in calls if c.startswith("scancel")] == ["scancel 40468625_0 40468625_1 40468626_0"]
    assert chain_total(out) == 3 * A2_PLANNED + 1500


def test_a2_done_and_stop_lines(sim):
    write_a2(sim.root, [A2_PLANNED] * 6)
    out, calls = sim.run(A2, DRY_RUN=0, FAKE_G="", FAKE_Q="", FAKE_SINFO="idle 9\n")
    assert re.search(r"^No shard left to slice\.$", out, re.M)
    assert not [c for c in calls if c.startswith("sbatch")]

    write_a2(sim.root, [10, 0, 0, 0, 0, 0])
    with open(sim.root / "results/stimulus_interventions/cells_gemma4_shard0.jsonl", "a") as handle:
        handle.write('{"status": "weird", "seconds": 1}\n')
    out, calls = sim.run(A2, DRY_RUN=0, FAKE_G="", FAKE_Q="", FAKE_SINFO="idle 9\n")
    assert re.search(r"^STOP:", out, re.M)
    assert not [c for c in calls if c.startswith("sbatch")]
