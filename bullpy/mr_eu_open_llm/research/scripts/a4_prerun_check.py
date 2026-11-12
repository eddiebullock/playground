#!/usr/bin/env python3
"""Pre-run check before A4's single frozen analysis (DECISIONS 2026-11-12, A4 rulings 1 to 7).

Run on the cluster from the project root, after every scored A4 shard has finished and before
`python -m scripts.a4_analyse`. It reads no response value: it parses the cells files, compares their
cell keys with the frozen plan, checks that log probabilities are finite numbers, and hashes inputs.

Checks, each of which must pass:
1. Both Holm-family models (gemma4, qwen3vl) have exactly 12 scored shard files,
   `cells_<model>_0.jsonl` to `cells_<model>_11.jsonl` (rulings 1 and 3).
2. Every line parses; each shard holds exactly the cells `planned_cells(manifest, k, 12)` lists, each
   once, all with status ok and the frozen design hash (rulings 3 to 5).
3. Every joint log probability (option readouts) and family log probability (blind readouts) is a finite
   number (ruling 2). Rows that are not are listed. With --clean_dir, filtered copies of the shards
   without those rows are written there, with the list, and the analysis then runs on that folder.
4. The manifest matches its pin; human target, ceiling, V1 pilot and recovery files match the sha256
   on line 9 of `PREREG_A4_DRAFT.md`; the V1 file passed is the pilot named on line 268 (ruling 7).
5. The gates file equals a fresh build by `research/scripts/a4_gates.py` from the frozen outputs (ruling 6).
6. BLAS runs single-threaded (ruling 7, audit A4-14): OMP, OpenBLAS, MKL and vecLib thread counts are 1.

Writes the record (command line, versions, every input's sha256, failures) to --record; exits 1 on any
failure.

Usage (project root):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=.:src python research/scripts/a4_prerun_check.py --record results/a4/prerun_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
N_SHARDS = 12
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PIN_LINE = re.compile(r"`(research/reproduction/[^`]+)` `([0-9a-f]{64})`")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_inputs(prereg: Path) -> Dict[str, str]:
    for line in prereg.read_text(encoding="utf-8").splitlines():
        if line.startswith("Frozen inputs and outputs, sha256:"):
            return dict(PIN_LINE.findall(line))
    raise SystemExit("PREREG_A4_DRAFT.md states no frozen input hashes.")


def non_finite(row: Dict[str, Any]) -> bool:
    values: List[Any] = list(row.get("joint_logprob") or []) + list((row.get("family_logprob") or {}).values())
    if row.get("readout") in ("mcq", "image_free") and not row.get("joint_logprob"):
        return True
    return any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)


def check_model(model: str, cells_dir: Path, manifest: List[Dict[str, Any]], design: str, run) -> Dict[str, Any]:
    out: Dict[str, Any] = {"failures": [], "non_finite_rows": []}
    files = sorted(p for p in cells_dir.glob(f"cells_{model}_*.jsonl") if p.stem[len(f"cells_{model}_"):].isdigit())
    numbers = sorted(int(p.stem[len(f"cells_{model}_"):]) for p in files)
    out["shards"] = numbers
    if numbers != list(range(N_SHARDS)):
        out["failures"].append(f"{model}: scored shards are {numbers}, not 0 to {N_SHARDS - 1}")
    for path in files:
        k = int(path.stem[len(f"cells_{model}_"):])
        planned = {run.cell_key(c) for c in run.planned_cells(manifest, k, N_SHARDS)}
        keys, bad_lines = [], 0
        for number, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                bad_lines += 1
                continue
            keys.append(run.cell_key(row))
            if row.get("status") != "ok":
                out["failures"].append(f"{path.name}: line {number} has status {row.get('status')!r}")
            if row.get("design_sha256") != design:
                out["failures"].append(f"{path.name}: line {number} carries another design hash")
            if non_finite(row):
                out["non_finite_rows"].append({"file": path.name, "line": number, "key": list(run.cell_key(row))})
        if bad_lines:
            out["failures"].append(f"{path.name}: {bad_lines} unparseable lines; resume the shard to completion")
        if len(keys) != len(set(keys)):
            out["failures"].append(f"{path.name}: {len(keys) - len(set(keys))} duplicate cells")
        if set(keys) != planned:
            out["failures"].append(f"{path.name}: {len(planned - set(keys))} planned cells missing, "
                                   f"{len(set(keys) - planned)} cells not planned for this shard")
    return out


def write_clean(cells_dir: Path, clean_dir: Path, models: Sequence[str], drop: List[Dict[str, Any]]) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    dropped = {(r["file"], r["line"]) for r in drop}
    for model in models:
        for path in sorted(cells_dir.glob(f"cells_{model}_*.jsonl")):
            if not path.stem[len(f"cells_{model}_"):].isdigit():
                continue
            lines = path.read_bytes().split(b"\n")
            kept = [raw for n, raw in enumerate(lines, start=1) if raw.strip() and (path.name, n) not in dropped]
            (clean_dir / path.name).write_bytes(b"\n".join(kept) + b"\n")
    (clean_dir / "removed_non_finite_rows.json").write_text(json.dumps(drop, indent=1) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells_dir", type=Path, default=Path("results/a4"))
    ap.add_argument("--manifest", type=Path, default=Path("results/a4/manifest.csv"))
    ap.add_argument("--human", type=Path, default=Path("research/reproduction/human_au_effects.json"))
    ap.add_argument("--ceiling", type=Path, default=Path("research/reproduction/a4_human_ceiling.json"))
    ap.add_argument("--recovery", type=Path, default=Path("research/reproduction/a4_recovery.json"))
    ap.add_argument("--detector", type=Path, default=Path("research/reproduction/a4_v1_pilot.csv"))
    ap.add_argument("--gates", type=Path, default=Path("research/reproduction/a4_gates.json"))
    ap.add_argument("--zip", type=Path, default=Path("data/external/snoek_au/au_emotion_data.zip"))
    ap.add_argument("--clean_dir", type=Path, default=None)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy, pandas, scipy  # noqa: E401
    from scripts import a4_design as d
    from scripts import a4_run as run
    spec = importlib.util.spec_from_file_location("a4_gates", Path(__file__).with_name("a4_gates.py"))
    gates_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gates_mod)

    failures: List[str] = []
    record: Dict[str, Any] = {"command": sys.argv, "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                                                                 "pandas": pandas.__version__, "python": sys.version.split()[0]},
                              "threads": {v: os.environ.get(v) for v in THREAD_VARS}}
    if any(os.environ.get(v) != "1" for v in THREAD_VARS):
        failures.append("BLAS is not single-threaded: set " + ", ".join(f"{v}=1" for v in THREAD_VARS))
    if not args.manifest.exists() or sha256(args.manifest) != d.MANIFEST_SHA256:
        failures.append("manifest missing or not the pinned one")
    pins = pinned_inputs(ROOT / "research" / "PREREG_A4_DRAFT.md")
    for arg in (args.human, args.ceiling, args.recovery, args.detector):
        want = pins.get(str(arg))
        if want is None:
            failures.append(f"{arg} is not one of the inputs pinned on PREREG_A4 line 9")
        elif not arg.exists() or sha256(arg) != want:
            failures.append(f"{arg} missing or not its pinned bytes")
    if not args.gates.exists() or json.loads(args.gates.read_text(encoding="utf-8")) != gates_mod.build(ROOT):
        failures.append("gates file missing or not what research/scripts/a4_gates.py builds from the frozen outputs")
    design = d.design_hash()
    record["design_sha256"] = design
    manifest = pandas.read_csv(args.manifest).to_dict("records") if args.manifest.exists() else []
    drop: List[Dict[str, Any]] = []
    record["models"] = {}
    for model in d.MODELS:
        result = check_model(model, args.cells_dir, manifest, design, run)
        failures += result["failures"]
        drop += result["non_finite_rows"]
        record["models"][model] = {"shards": result["shards"], "n_non_finite_rows": len(result["non_finite_rows"])}
    record["non_finite_rows"] = drop
    if drop and args.clean_dir is None:
        failures.append(f"{len(drop)} rows with non-finite log probabilities: rerun with --clean_dir (ruling 2)")
    elif drop:
        write_clean(args.cells_dir, args.clean_dir, d.MODELS, drop)
        record["clean_dir"] = str(args.clean_dir)
    inputs = [args.manifest, args.human, args.ceiling, args.recovery, args.detector, args.gates, args.zip]
    inputs += sorted(args.cells_dir.glob("cells_*_*.jsonl"))
    record["sha256"] = {str(p): sha256(p) for p in inputs if p.exists()}
    record["failures"] = failures
    record["script_sha256"] = sha256(Path(__file__))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for failure in failures:
        print("FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
