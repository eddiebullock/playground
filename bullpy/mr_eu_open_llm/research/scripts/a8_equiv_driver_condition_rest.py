"""The rest of one condition of A8's GPU equivalence test, after its job hit the one-hour limit (DECISIONS 2026-11-14).

Job 40986052 ran `a8_equiv_driver_condition.py` (sha256 e929e3d0...) for both conditions. In each condition both frozen
runs ended with exit 0: the driver starts the shard runner only after both frozen commands pass `check=True`. The job then
hit its one-hour limit during the first shard run. This driver issues that driver's remaining commands, unchanged and in
the same order, in one process on one GPU: the shard runner for shard 0, then for shard 1, then the merge, then the two
comparisons and the report, exactly as there.

It stops before any command unless both frozen outputs exist and no shard, merged or report output exists. The partial
first shard of job 40986052 is moved out of the equivalence tree, unread, before this runs.

Run from `~/rds/hpc-work/a8_check` with A8_EQUIV_MODELS_DIR set (`slurm_jobs/a8_equiv_rest_intr.sh` does this):

    python a8_equiv_driver_condition_rest.py real
    python a8_equiv_driver_condition_rest.py mismatch
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
    sys.exit("usage: a8_equiv_driver_condition_rest.py real|mismatch")
condition = sys.argv[1]
models_dir = os.environ["A8_EQUIV_MODELS_DIR"]
out_root = ROOT / "results/a8/equivalence"
script = str(ROOT / "research/scripts/a8_rotation_shard.py")
base = ["--equivalence_label", "equivalence_qwen3vl_8b", "--equivalence_models_dir", models_dir,
        "--rotation", "0", "--condition", condition, "--max_trials", "6"]
frozen = [out_root / condition / f"frozen_{run}" / "eval.json" for run in ("a", "b")]
missing = [str(p) for p in frozen if not (p.is_file() and p.stat().st_size > 0)]
if missing:
    sys.exit(f"frozen outputs missing: {missing}")
present = [str(p) for p in [out_root / condition / f"shard{k}of2" for k in range(2)]
           if p.exists() and any(p.iterdir())]
present += [str(p) for p in (out_root / condition / "merged.json", out_root / condition / "equivalence_report.json")
            if p.exists()]
if present:
    sys.exit(f"outputs already present, nothing run: {present}")
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
