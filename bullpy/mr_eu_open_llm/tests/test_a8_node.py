"""slurm_jobs/a8_node.sh, run with stand-ins for `srun` and `timeout`: which launcher steps it starts, with which
variables for the unchanged slurm_jobs/a8_job.sh, what it refuses before starting anything, and how it reads each
slot's exit status. No launcher, model or Slurm command runs."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "slurm_jobs/a8_node.sh"

FAKES = {
    # Drops timeout's options and duration, then runs the command, as GNU timeout does when the budget holds.
    "timeout": """#!/usr/bin/env bash
echo "timeout $*" >> "$CALLS"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --*) shift ;;
    *) shift; break ;;
  esac
done
exec "$@"
""",
    # Records the step and exits with FAKE_EXIT_<step> (default 0).
    "srun": """#!/usr/bin/env bash
echo "srun $*" >> "$CALLS"
step=""
for a in "$@"; do
  case "$a" in
    STEP=*) step="${a#STEP=}" ;;
  esac
done
var="FAKE_EXIT_${step}"
exit "${!var:-0}"
""",
}


@pytest.fixture
def node(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    root = tmp_path / "study3"
    root.mkdir()
    calls = tmp_path / "calls.log"

    def run(**env):
        calls.write_text("")
        full = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path),
                "PROJECT_ROOT": str(root), "CALLS": str(calls)}
        full.update({k: str(v) for k, v in env.items()})
        proc = subprocess.run(["bash", str(NODE)], env=full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=60)
        return proc.returncode, proc.stdout, calls.read_text().splitlines()

    return run


def srun_lines(calls):
    return sorted(line for line in calls if line.startswith("srun "))


def test_four_steps_start_on_one_gpu_each_through_the_unchanged_job_file(node):
    slots = ("STEP=gates,A8_ID=qwen3vl_32b STEP=gates,A8_ID=gemma4_31b "
             "STEP=a1_shard,A8_ID=qwen3vl_32b,ROTATION=0,CONDITION=real,N_SHARDS=4,SHARD=3 "
             "STEP=a2_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=0")
    rc, out, calls = node(SLOTS=slots, BUDGET="700m")
    assert rc == 0, out
    assert sorted(line.split(" srun ")[0] for line in calls if line.startswith("timeout ")) == \
        ["timeout --signal=TERM --kill-after=90s 700m"] * 4
    tail = "bash slurm_jobs/a8_job.sh"
    head = "srun --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G env"
    assert srun_lines(calls) == sorted([
        f"{head} STEP=gates A8_ID=qwen3vl_32b {tail}",
        f"{head} STEP=gates A8_ID=gemma4_31b {tail}",
        f"{head} STEP=a1_shard A8_ID=qwen3vl_32b ROTATION=0 CONDITION=real N_SHARDS=4 SLURM_ARRAY_TASK_ID=3 {tail}",
        f"{head} STEP=a2_run A8_ID=gemma4_31b N_SHARDS=12 SLURM_ARRAY_TASK_ID=0 {tail}",
    ])
    assert out.count("ended (exit 0)") == 4


def test_two_gpus_per_slot_allow_two_slots(node):
    rc, out, calls = node(SLOTS="STEP=a3,A8_ID=qwen3vl_32b STEP=ladder,A8_ID=qwen3vl_32b", BUDGET="52m",
                          GPUS_PER_SLOT=2)
    assert rc == 0, out
    assert all("--gres=gpu:2 " in line for line in srun_lines(calls)) and len(srun_lines(calls)) == 2
    rc, out, calls = node(SLOTS="STEP=a3,A8_ID=qwen3vl_32b STEP=ladder,A8_ID=qwen3vl_32b STEP=gates,A8_ID=gemma4_31b",
                          BUDGET="52m", GPUS_PER_SLOT=2)
    assert rc == 2 and calls == [] and "Need 1 to 2 slots" in out


@pytest.mark.parametrize("env, message", [
    ({"BUDGET": "52m"}, "Set SLOTS"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b"}, "Set BUDGET"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b", "BUDGET": "11h40m"}, "Set BUDGET"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b", "BUDGET": "52m", "GPUS_PER_SLOT": "5"}, "GPUS_PER_SLOT is 1 to 4"),
    ({"SLOTS": " ".join(["STEP=gates,A8_ID=qwen3vl_32b"] * 5), "BUDGET": "52m"}, "Need 1 to 4 slots"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b,MODEL=qwen3vl", "BUDGET": "52m"}, "unknown key MODEL"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_8b", "BUDGET": "52m"}, "A8_ID must be"),
    ({"SLOTS": "STEP=gates,A8_ID", "BUDGET": "52m"}, "is not KEY=value"),
    ({"SLOTS": "STEP=a4_analyse,A8_ID=qwen3vl_32b", "BUDGET": "52m"}, "STEP must be a GPU step"),
    ({"SLOTS": "STEP=a1_merge,A8_ID=qwen3vl_32b,ROTATION=0,CONDITION=real,N_SHARDS=4", "BUDGET": "52m"},
     "STEP must be a GPU step"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b,SHARD=0", "BUDGET": "52m"}, "gates takes no"),
    ({"SLOTS": "STEP=a1_shard,A8_ID=qwen3vl_32b,ROTATION=0,N_SHARDS=4,SHARD=0", "BUDGET": "52m"},
     "a1_shard needs ROTATION"),
    ({"SLOTS": "STEP=a1_shard,A8_ID=qwen3vl_32b,ROTATION=5,CONDITION=real,N_SHARDS=4,SHARD=0", "BUDGET": "52m"},
     "a1_shard needs ROTATION"),
    ({"SLOTS": "STEP=a2_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=12", "BUDGET": "52m"}, "SHARD from 0"),
    ({"SLOTS": "STEP=a4_run,A8_ID=gemma4_31b,SHARD=1", "BUDGET": "52m"}, "SHARD from 0"),
    ({"SLOTS": "STEP=a4_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=1,CONDITION=real", "BUDGET": "52m"},
     "a4_run takes no"),
    ({"SLOTS": "STEP=a2_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=1 STEP=a2_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=1",
      "BUDGET": "52m"}, "repeats a step"),
])
def test_refused_before_anything_starts(node, env, message):
    rc, out, calls = node(**env)
    assert rc == 2 and message in out and calls == []


def test_a_slot_may_ask_for_more_host_memory_within_the_job(node):
    slots = "STEP=gates,A8_ID=qwen3vl_32b,MEM=240G STEP=a3,A8_ID=qwen3vl_32b STEP=a3,A8_ID=gemma4_31b"
    rc, out, calls = node(SLOTS=slots, BUDGET="700m")
    assert rc == 0, out
    lines = srun_lines(calls)
    assert [line for line in lines if "STEP=gates" in line][0].count("--mem=240G ") == 1
    assert sum("--mem=120G " in line for line in lines) == 2
    assert all("MEM=" not in line.split(" env ", 1)[1] for line in lines)  # a run setting, not a job variable
    rc, out, calls = node(SLOTS=slots + " STEP=ladder,A8_ID=gemma4_31b", BUDGET="700m")
    assert rc == 2 and calls == [] and "600G of host memory together; the job has 500G" in out
    rc, out, calls = node(SLOTS=slots, BUDGET="700m", SLURM_MEM_PER_NODE=400 * 1024)
    assert rc == 2 and calls == [] and "the job has 400G" in out


@pytest.mark.parametrize("env, message", [
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b,MEM=240GB", "BUDGET": "52m"}, "MEM is a whole number"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b,MEM=", "BUDGET": "52m"}, "is not KEY=value"),
    ({"SLOTS": "STEP=gates,A8_ID=qwen3vl_32b", "BUDGET": "52m", "STEP_MEM": "120"}, "STEP_MEM is a whole number"),
])
def test_memory_settings_are_checked(node, env, message):
    rc, out, calls = node(**env)
    assert rc == 2 and message in out and calls == []


def test_one_bad_slot_stops_the_whole_job(node):
    rc, out, calls = node(SLOTS="STEP=gates,A8_ID=qwen3vl_32b STEP=a3,A8_ID=gemma4_31b,SHARD=0", BUDGET="52m")
    assert rc == 2 and calls == []


def test_a_resumable_step_stopped_by_the_budget_is_not_a_failure(node):
    rc, out, _ = node(SLOTS="STEP=a4_run,A8_ID=gemma4_31b,N_SHARDS=12,SHARD=0 STEP=a3,A8_ID=qwen3vl_32b",
                      BUDGET="52m", FAKE_EXIT_a4_run=124, FAKE_EXIT_a3=137)
    assert rc == 0, out
    assert out.count("a later job resumes it") == 2


@pytest.mark.parametrize("step_slot, code", [
    ("STEP=gates,A8_ID=qwen3vl_32b", 124),
    ("STEP=a1_shard,A8_ID=qwen3vl_32b,ROTATION=1,CONDITION=mismatch,N_SHARDS=4,SHARD=2", 137),
])
def test_a_step_that_does_not_resume_and_is_stopped_fails_the_job(node, step_slot, code):
    step = step_slot.split(",")[0].split("=")[1]
    rc, out, _ = node(SLOTS=f"{step_slot} STEP=ladder,A8_ID=gemma4_31b", BUDGET="700m", **{f"FAKE_EXIT_{step}": code})
    assert rc == 1
    assert "a failed attempt of a step that does not resume" in out and "ended (exit 0)" in out


def test_a_failed_step_fails_the_job_and_the_others_still_finish(node):
    rc, out, calls = node(SLOTS="STEP=delivery,A8_ID=qwen3vl_32b STEP=delivery,A8_ID=gemma4_31b", BUDGET="120m",
                          FAKE_EXIT_delivery=1)
    assert rc == 1 and out.count("FAILED (exit 1)") == 2 and len(srun_lines(calls)) == 2


def test_the_job_file_it_calls_reads_these_variables_and_takes_the_shard_from_the_array_index():
    job = (ROOT / "slurm_jobs/a8_job.sh").read_text()
    for name in ("STEP", "A8_ID", "ROTATION", "CONDITION", "N_SHARDS"):
        assert "${" + name in job
    assert 'ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}")' in job
