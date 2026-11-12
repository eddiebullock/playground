#!/usr/bin/env python3
"""Pre-run check before A2's single frozen analysis (DECISIONS 2026-11-12, A2 rulings 2, 3, 6 and 7).

Run on the cluster from the project root, after every A2 shard has finished and before
`scripts/analyse_stimulus_interventions.py --a2_frozen`. It computes no outcome: it parses the cells
files and reads each ok row's pixel tensor shapes, and nothing else in a row.

Checks, each of which must pass:
1. The working directory is the project root.
2. Cells and report files follow the full-run names, `cells_{model}_shard{k}.jsonl` and
   `run_{model}_shard{k}.json`, for this model, with the same shard numbers on both.
3. Every non-empty line of every cells file parses as JSON. Unparseable lines are listed with line
   number, byte offset, length and sha256, so the repair rule (ruling 2) can be applied and logged.
4. Model identity (ruling 3): the last dimension of every pixel tensor of every ok row is one the model's
   committed delivery record shows (`research/reproduction/input_delivery_{model}.json`).
5. Gate inputs (ruling 6): the A1 summary, the A3 summary (output folder or committed copy), the vision
   gate file and the delivery file exist, and the pinned geometry matches its hash.
6. Pins (ruling 7): the sha256 of the A2 code, its analysis dependencies, the gate inputs, the geometry
   and every cells and report file is written to the record.

Writes the record to --record and exits 0 if every check passes; exits 1 and writes the record with the
failures otherwise.

Usage (project root):
    PYTHONPATH=.:src python research/scripts/a2_prerun_check.py --model gemma4 \
        --cells results/stimulus_interventions/cells_gemma4_shard*.jsonl \
        --run_report results/stimulus_interventions/run_gemma4_shard*.json \
        --record results/stimulus_interventions/prerun_check_gemma4.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

ROOT_MARKER = Path("scripts/analyse_stimulus_interventions.py")
REPRO = Path("research/reproduction")
A3_NAME = "face_input_validity_summary.json"
A3_OUTPUT_DIR = Path("results/face_input_validity")
GEOMETRY = Path("results/stimulus_interventions/geometry.json")
PINNED = (
    "scripts/stimulus_design.py", "scripts/stimulus_interventions.py", "scripts/run_stimulus_interventions.py",
    "scripts/analyse_stimulus_interventions.py", "scripts/recovery_stimulus_interventions.py",
    "slurm_jobs/study3_stimulus_interventions.sh", "scripts/human_response_predictor.py",
    "research/scripts/analyse_rotation.py", "src/paper_v2/option_score.py", "scripts/prompts.py",
    "scripts/vision_gates.py", "research/scripts/a2_declared_extras.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shard_numbers(paths: Sequence[Path], pattern: str) -> Optional[List[int]]:
    numbers = []
    for p in paths:
        m = re.fullmatch(pattern, p.name)
        if not m:
            return None
        numbers.append(int(m.group(1)))
    return sorted(numbers)


def unparseable_lines(path: Path) -> List[Dict[str, Any]]:
    bad, offset = [], 0
    for number, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
        if raw.strip():
            try:
                json.loads(raw)
            except ValueError:
                bad.append({"line": number, "byte_offset": offset, "length": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()})
        offset += len(raw) + 1
    return bad


def model_dims(delivery: Dict[str, Any]) -> Set[int]:
    dims: Set[int] = set()

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "pixel_shapes" and isinstance(v, dict):
                    dims.update(int(s[-1]) for s in v.values() if isinstance(s, list) and s)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(delivery)
    return dims


def foreign_rows(paths: Iterable[Path], allowed: Set[int]) -> List[Dict[str, Any]]:
    out = []
    for path in paths:
        for number, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get("status") != "ok":
                continue
            shapes = (row.get("input") or {}).get("pixel_shapes") or {}
            dims = {int(s[-1]) for s in shapes.values() if isinstance(s, list) and s}
            if not dims or not dims <= allowed:
                out.append({"file": path.name, "line": number, "pixel_last_dims": sorted(dims)})
    return out


def check(model: str, cells: Sequence[Path], reports: Sequence[Path], root: Path) -> Dict[str, Any]:
    failures: List[str] = []
    record: Dict[str, Any] = {"model": model}
    if not (root / ROOT_MARKER).exists():
        failures.append("not run from the project root")
        return {**record, "failures": failures}
    cell_shards = shard_numbers(cells, rf"cells_{re.escape(model)}_shard(\d+)\.jsonl")
    report_shards = shard_numbers(reports, rf"run_{re.escape(model)}_shard(\d+)\.json")
    record["shards"] = cell_shards
    if cell_shards is None or report_shards is None:
        failures.append("a cells or report file is not named for this model's full run")
    elif cell_shards != report_shards or cell_shards != list(range(len(cell_shards))):
        failures.append(f"shard numbers differ or are not one complete set: cells {cell_shards}, reports {report_shards}")
    bad = {p.name: unparseable_lines(p) for p in cells}
    bad = {k: v for k, v in bad.items() if v}
    record["unparseable_lines"] = bad
    if bad:
        failures.append("unparseable cells lines: apply the repair rule (DECISIONS 2026-11-12, A2 ruling 2)")
    delivery_path = root / REPRO / f"input_delivery_{model}.json"
    gate_inputs = {"a1": root / REPRO / "rotation_summary.json", "vision_gates": root / REPRO / f"vision_gates_{model}.json",
                   "delivery": delivery_path}
    a3 = next((p for p in (root / A3_OUTPUT_DIR / A3_NAME, root / REPRO / A3_NAME) if p.exists()), None)
    gate_inputs["a3"] = a3 if a3 is not None else root / REPRO / A3_NAME
    missing = [k for k, p in gate_inputs.items() if not p.exists()]
    if missing:
        failures.append(f"gate inputs missing: {missing}")
    allowed = model_dims(json.loads(delivery_path.read_text(encoding="utf-8"))) if delivery_path.exists() else set()
    record["allowed_pixel_last_dims"] = sorted(allowed)
    foreign = foreign_rows(cells, allowed)
    record["rows_not_matching_model"] = foreign[:50]
    record["n_rows_not_matching_model"] = len(foreign)
    if foreign:
        failures.append(f"{len(foreign)} ok rows have pixel shapes not seen for {model}")
    geometry = root / GEOMETRY
    if geometry.exists():
        import sys
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))
        from scripts import run_stimulus_interventions as run  # frozen; only its pinned hash is used here
        if sha256(geometry) != run.GEOMETRY_SHA256:
            failures.append("geometry.json does not match its pin")
    else:
        failures.append("geometry.json is missing")
    files = [root / p for p in PINNED] + list(gate_inputs.values()) + [geometry] + list(cells) + list(reports)
    record["sha256"] = {str(p.relative_to(root)) if p.is_absolute() and root in p.parents else str(p): sha256(p)
                        for p in files if p.exists()}
    return {**record, "failures": failures}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--run_report", type=Path, nargs="+", required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)
    result = check(args.model, args.cells, args.run_report, args.root.resolve())
    result["script_sha256"] = sha256(Path(__file__))
    args.record.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for failure in result["failures"]:
        print("FAIL:", failure)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
