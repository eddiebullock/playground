"""One condition of A8's GPU equivalence test (Freeze checklist item 2), for one step of an intr job.

This is the cluster driver `~/rds/hpc-work/a8_check/a8_equiv_driver.py` (sha256 51a0887b..., written 2026-11-09, which
replicates `tests/test_a8.py::test_a1_equivalence_on_qwen3vl_8b_gpu` because the env has no pytest) restricted to the
condition named as the only argument. Its commands, their order and the output paths are the same, so the two conditions
can run at once as two one-GPU steps of one `intr` job (DECISIONS 2026-11-14, scheduling only). Within a condition, the
frozen command runs twice and the shard runner twice, one after another in this process, on one GPU, as before.

Run from `~/rds/hpc-work/a8_check` with A8_EQUIV_MODELS_DIR set (`slurm_jobs/a8_equiv_intr.sh` does this):

    python a8_equiv_driver_condition.py real
    python a8_equiv_driver_condition.py mismatch
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "src"), str(ROOT)]
import a8_rotation_shard as shard  # noqa: E402

if len(sys.argv) != 2 or sys.argv[1] not in ("real", "mismatch"):
    sys.exit("usage: a8_equiv_driver_condition.py real|mismatch")
condition = sys.argv[1]
models_dir = os.environ["A8_EQUIV_MODELS_DIR"]
out_root = ROOT / "results/a8/equivalence"
script = str(ROOT / "research/scripts/a8_rotation_shard.py")
base = ["--equivalence_label", "equivalence_qwen3vl_8b", "--equivalence_models_dir", models_dir,
        "--rotation", "0", "--condition", condition, "--max_trials", "6"]
frozen = []
for run in ("a", "b"):
    out = out_root / condition / f"frozen_{run}" / "eval.json"
    subprocess.run([sys.executable, script, "frozen", *base, "--output", str(out)], check=True, cwd=ROOT)
    frozen.append(out)
shards = []
for k in range(2):
    out = out_root / condition / f"shard{k}of2" / "eval.json"
    subprocess.run([sys.executable, script, "run", *base, "--shard", str(k), "--n_shards", "2", "--output", str(out)],
                   check=True, cwd=ROOT)
    shards.append(str(out))
merged = out_root / condition / "merged.json"
subprocess.run([sys.executable, script, "merge", "--shards", *shards, "--output", str(merged)], check=True, cwd=ROOT)
report = {"frozen_vs_frozen": shard.compare(frozen[0], frozen[1]), "frozen_vs_merged": shard.compare(frozen[0], merged)}
(out_root / condition / "equivalence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("EQUIVALENCE", condition, json.dumps({k: v.get("identical") for k, v in report.items()}))
