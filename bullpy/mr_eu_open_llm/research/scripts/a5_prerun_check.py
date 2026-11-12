#!/usr/bin/env python3
"""Pre-run check before A5's single frozen analysis (DECISIONS 2026-11-12, A5 rulings 1 to 3 and 9).

Run on the cluster from the project root, after both models' A2 runs have finished and before
`python -m scripts.a5_analyse`. It computes no outcome: it parses the A2 cells files, counts ok
`original` cells per model and route from their `status` fields, and hashes inputs.

Checks, each of which must pass:
1. Both A5 models (gemma4, qwen3vl) have a cells file and a report file for every shard below
   `n_shards`, named as A2's full run names them (ruling 1: a model missing here would otherwise be
   read as having no A2 cells, and the other model's step-down families would shrink).
2. Every cells line parses (as in `a2_prerun_check.py`, ruling 2 of A2).
3. Every model has at least one ok `original` cell on every route A5 reads (ruling 3: A2's `gate_b`
   stops with StopIteration on a route with none).
4. The A1 and A3 summaries are present, the A3 summary has the pinned bytes, and both record
   `vision_dependent: true` for both models (ruling 2: anything else is an input error, since both are
   locked as Pass).
5. The human norms file has the frozen sha256 (ruling 9).

Writes the record, with the sha256 of every input, to --record; exits 1 on any failure.

Usage (project root):
    PYTHONPATH=.:src python research/scripts/a5_prerun_check.py \
        --cells_dir results/stimulus_interventions --record results/a5/prerun_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MODELS = ("gemma4", "qwen3vl")
ROUTES = ("single_face", "video_face")
REPRO = Path("research/reproduction")
A3_NAME = "face_input_validity_summary.json"
A3_SHA256 = "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4"
HUMAN = Path("data/eu_emotions_human_entropy.json")
HUMAN_SHA256 = "f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744"


def _a2_check():
    spec = importlib.util.spec_from_file_location("a2_prerun_check", Path(__file__).with_name("a2_prerun_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_counts(paths: Sequence[Path]) -> Dict[str, int]:
    counts = {route: 0 for route in ROUTES}
    for path in paths:
        for raw in path.read_bytes().split(b"\n"):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get("condition") == "original" and row.get("status") == "ok" and row.get("input_name") in counts:
                counts[row["input_name"]] += 1
    return counts


def check(cells_dir: Path, root: Path) -> Dict[str, Any]:
    a2 = _a2_check()
    failures: List[str] = []
    record: Dict[str, Any] = {"models": {}}
    hashes: Dict[str, str] = {}
    for model in MODELS:
        cells = sorted(cells_dir.glob(f"cells_{model}_shard*.jsonl"))
        reports = sorted(cells_dir.glob(f"run_{model}_shard*.json"))
        cs = a2.shard_numbers(cells, rf"cells_{model}_shard(\d+)\.jsonl") or []
        rs = a2.shard_numbers(reports, rf"run_{model}_shard(\d+)\.json") or []
        n_shards = {json.loads(p.read_text(encoding="utf-8")).get("n_shards") for p in reports}
        entry: Dict[str, Any] = {"cell_shards": cs, "report_shards": rs, "n_shards": sorted(map(str, n_shards))}
        complete = bool(cs) and cs == rs and len(n_shards) == 1
        if complete:
            total = next(iter(n_shards))
            complete = isinstance(total, int) and cs == list(range(total))
        if not complete:
            failures.append(f"{model}: cells and reports are not one complete shard set")
        bad = {p.name: a2.unparseable_lines(p) for p in cells}
        bad = {k: v for k, v in bad.items() if v}
        entry["unparseable_lines"] = bad
        if bad:
            failures.append(f"{model}: unparseable cells lines (A2 ruling 2 applies)")
        counts = original_counts(cells)
        entry["ok_original_cells"] = counts
        for route, n in counts.items():
            if n == 0:
                failures.append(f"{model} {route}: no ok original cell; log a dated deviation before running")
        record["models"][model] = entry
        hashes.update({str(p): sha256(p) for p in cells + reports})
    a1 = root / REPRO / "rotation_summary.json"
    a3 = next((p for p in (root / "results/face_input_validity" / A3_NAME, root / REPRO / A3_NAME)
               if p.exists() and sha256(p) == A3_SHA256), None)
    if not a1.exists():
        failures.append("A1 summary missing")
    if a3 is None:
        failures.append("A3 summary missing or not the pinned bytes")
    for model in MODELS:
        a1_ok = a1.exists() and (json.loads(a1.read_text()).get(model) or {}).get("vision_dependent") is True
        a3_ok = a3 is not None and (json.loads(a3.read_text()).get(model) or {}).get("vision_dependent") is True
        if not (a1_ok and a3_ok):
            failures.append(f"{model}: A1 or A3 does not read vision_dependent true; input error, not a result")
    human = root / HUMAN
    if not human.exists() or sha256(human) != HUMAN_SHA256:
        failures.append("human norms file missing or not the frozen bytes")
    for p in [a1, human] + ([a3] if a3 else []) + [root / REPRO / f"{k}_{m}.json" for k in ("vision_gates", "input_delivery") for m in MODELS]:
        if p.exists():
            hashes[str(p.relative_to(root)) if root in p.parents else str(p)] = sha256(p)
    record["sha256"] = hashes
    record["failures"] = failures
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells_dir", type=Path, default=Path("results/stimulus_interventions"))
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)
    result = check(args.cells_dir, args.root.resolve())
    result["script_sha256"] = sha256(Path(__file__))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for failure in result["failures"]:
        print("FAIL:", failure)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
