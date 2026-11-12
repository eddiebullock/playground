#!/usr/bin/env python3
"""Pre-run check before A9's single frozen analysis (PREREG_A9 deviation of 2026-11-12, ruling 5).

Run on the cluster from the project root, after every scored A9 shard has finished and after A4's
single analysis has written `results/a4/summary.json`, before `python -m scripts.a9_analyse`. It reads
no response value. It applies A4's pre-run check (`research/scripts/a4_prerun_check.py`) to A9's own
cell plan (`scripts/a9_run.planned_cells`) and design hash, and adds A9's frozen pin checks.

Checks, each of which must pass:
1. A9's code pins, input pins and manifest pin, by A9's own frozen checks (`scripts/a9_design.py`).
2. Both models have exactly 12 scored shards, each holding exactly its planned cells, once each, with
   status ok and A9's design hash; every line parses.
3. Every joint log probability is a finite number (ruling 1). With --clean_dir, filtered copies without
   the rows that are not are written there with the list, and the analysis runs on that folder.
4. The gates file is the one built from the frozen outputs (`research/scripts/a4_gates.py`): A9's gates
   are exactly A4's, per model.
5. A4's summary exists and carries A4's frozen design hash (A9 compares with A4's decisions).
6. BLAS runs single-threaded (ruling 4).

Usage (project root):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=.:src python research/scripts/a9_prerun_check.py --record results/a9/prerun_check.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells_dir", type=Path, default=Path("results/a9"))
    ap.add_argument("--manifest", type=Path, default=Path("results/a9/manifest.csv"))
    ap.add_argument("--gates", type=Path, default=Path("research/reproduction/a4_gates.json"))
    ap.add_argument("--a4_summary", type=Path, default=Path("results/a4/summary.json"))
    ap.add_argument("--clean_dir", type=Path, default=None)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy, pandas, scipy  # noqa: E401
    from scripts import a4_design
    from scripts import a9_design as a9
    from scripts import a9_run
    pre = _load("a4_prerun_check")
    gates_mod = _load("a4_gates")

    failures: List[str] = []
    record: Dict[str, Any] = {"command": sys.argv, "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                                                                 "pandas": pandas.__version__, "python": sys.version.split()[0]},
                              "threads": {v: os.environ.get(v) for v in pre.THREAD_VARS}}
    if any(os.environ.get(v) != "1" for v in pre.THREAD_VARS):
        failures.append("BLAS is not single-threaded: set " + ", ".join(f"{v}=1" for v in pre.THREAD_VARS))
    for name, check in (("code pins", lambda: a9.check_code_pins(require=True)), ("input pins", a9.check_pins),
                        ("manifest", lambda: a9.check_manifest(args.manifest))):
        try:
            check()
        except (SystemExit, FileNotFoundError, OSError) as exc:
            failures.append(f"A9 {name}: {exc}")
    if not args.gates.exists() or json.loads(args.gates.read_text(encoding="utf-8")) != gates_mod.build(ROOT):
        failures.append("gates file missing or not what research/scripts/a4_gates.py builds from the frozen outputs")
    if not args.a4_summary.exists():
        failures.append("A4's summary does not exist yet; A9 runs after A4's single analysis")
    elif json.loads(args.a4_summary.read_text(encoding="utf-8")).get("design_sha256") != a4_design.design_hash():
        failures.append("A4's summary does not carry A4's frozen design hash")
    design = a9.design_hash()
    record["design_sha256"] = design
    manifest = pandas.read_csv(args.manifest).to_dict("records") if args.manifest.exists() else []
    drop: List[Dict[str, Any]] = []
    record["models"] = {}
    for model in a9.MODELS:
        result = pre.check_model(model, args.cells_dir, manifest, design, a9_run)
        failures += result["failures"]
        drop += result["non_finite_rows"]
        record["models"][model] = {"shards": result["shards"], "n_non_finite_rows": len(result["non_finite_rows"])}
    record["non_finite_rows"] = drop
    if drop and args.clean_dir is None:
        failures.append(f"{len(drop)} rows with non-finite log probabilities: rerun with --clean_dir (ruling 1)")
    elif drop:
        pre.write_clean(args.cells_dir, args.clean_dir, a9.MODELS, drop)
        record["clean_dir"] = str(args.clean_dir)
    inputs = [args.manifest, args.gates, args.a4_summary] + sorted(args.cells_dir.glob("cells_*_*.jsonl"))
    record["sha256"] = {str(p): pre.sha256(p) for p in inputs if p.exists()}
    record["failures"] = failures
    record["script_sha256"] = pre.sha256(Path(__file__))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for failure in failures:
        print("FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
