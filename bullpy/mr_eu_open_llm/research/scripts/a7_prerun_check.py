#!/usr/bin/env python3
"""Pre-run check before A7's single frozen Stage 1 analysis; writes the --inputs file it reads.

Written 2026-11-12, before any Stage 1 cell or A2 analysis output exists. Run on the Mac from the project
root, after A2's analysis has run (its run note exists) and A7's Stage 1 shards have been pulled
(`post_phase3.py pull a7`). It computes no outcome: from Stage 1 cells it reads only `trial_id`,
`input_name`, `condition`, `status`, `a7_design_sha256` and `decision_date`; from A2's cells only
`trial_id`; from A2's run reports only the delivery rows' `tokens_across_view`, as A2's analysis does.

Checks, each of which must pass:
1. The working directory is the project root.
2. A2's analysis has run: `research/reproduction/a2_run_note.json` reads complete, and both A2 outputs
   exist with the sha256 the run note recorded. (PREREG_A7_DRAFT "Preconditions": A7's gates come from
   A2's analysis output; a missing gate has not passed, so the run would be void, not a result.)
3. The A2 cells and run reports A7 reads are exactly the ones A2's analysis read, taken from the run
   note's command, with the sha256 A2's pre-run record holds for each.
4. Stage 1 cells: `results/a7/a7_cells_{model}_shard{0,1}.jsonl` for both models (the scope entry of
   DECISIONS 2026-11-10 ran N_SHARDS=2). Every non-empty line parses; every row carries this design's
   hash, `decision_date` 2026-11-10 and A7's route; no (trial, condition) appears twice.
5. Stage 0 tables: `research/reproduction/a7_stage0_{model}.json` exist and are this design's.

Then it writes the inputs file: per model, the Stage 0 table, the Stage 1 cells, A2's output and cells,
and the recorded decisions (DECISIONS 2026-11-10: route arm run, conditional on preconditions applied by
code; token arm not run, not admitted by Stage 0); plus `below_token_floor` per model, the clips under
8 tokens across the face, computed with A2's frozen functions on A2's own inputs, exactly as A2's
analysis computes them (`analyse_stimulus_interventions.main`), for A7's declared sensitivity.

The record holds every input's sha256. Exit 0 if every check passes; exit 1, with the record and no
inputs file, otherwise.

Usage (project root):
    PYTHONPATH=.:src python research/scripts/a7_prerun_check.py \
        --record research/reproduction/a7_prerun.json --inputs research/reproduction/a7_inputs.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT_MARKER = Path("scripts/a7_analyse.py")
REPRO = Path("research/reproduction")
MODELS = ("gemma4", "qwen3vl")
N_SHARDS = 2
DECISION_DATE = "2026-11-10"
DECISIONS_HEADING = "## 2026-11-10: A7 Stage 1 scope"
GEOMETRY = Path("results/stimulus_interventions/geometry.json")
DECISIONS = {
    "route": {"run": True, "date": DECISION_DATE,
              "reason": "DECISIONS 2026-11-10, A7 Stage 1 scope: route arm run on Gemma 4 and Qwen3-VL, "
                        "conditional on the preconditions, applied by code (DECISIONS 2026-11-12)"},
    "token": {"run": False, "date": DECISION_DATE,
              "reason": "DECISIONS 2026-11-10, A7 Stage 1 scope: token arm not run; Stage 0 did not admit it"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flag_values(command: Sequence[str], flag: str) -> List[str]:
    """The values after `flag` in a recorded command, up to the next flag."""
    if flag not in command:
        return []
    out = []
    for token in command[command.index(flag) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def a2_inputs(note: Dict[str, Any], model: str) -> Tuple[List[str], List[str], Optional[str]]:
    """Cells, reports and output path of this model's frozen A2 analysis step, as the run note recorded it."""
    for step in note.get("steps", []):
        command = step.get("command") or []
        if "scripts.analyse_stimulus_interventions" in command and flag_values(command, "--model") == [model]:
            out = flag_values(command, "--output")
            return flag_values(command, "--cells"), flag_values(command, "--run_report"), (out[0] if out else None)
    return [], [], None


def parse_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows, bad = [], []
    for number, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
        if raw.strip():
            try:
                rows.append(json.loads(raw))
            except ValueError:
                bad.append(number)
    return rows, bad


def check(root: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    failures: List[str] = []
    record: Dict[str, Any] = {"decision_date": DECISION_DATE, "n_shards": N_SHARDS}
    if not (root / ROOT_MARKER).exists():
        return {**record, "failures": ["not run from the project root"]}, None
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from scripts import a7_design  # frozen; only its design hash is used here
    from scripts import a7_stage0 as s0
    from scripts.a7_analyse import ROUTE
    from scripts import analyse_stimulus_interventions as an  # frozen A2 functions, used as A2's analysis uses them
    from scripts import run_stimulus_interventions as run

    hashed: List[Path] = []
    decisions_text = (root / "research" / "DECISIONS.md").read_text(encoding="utf-8")
    if DECISIONS_HEADING not in decisions_text:
        failures.append("the DECISIONS 2026-11-10 entry that scopes Stage 1 is missing")

    note_path = root / REPRO / "a2_run_note.json"
    note = json.loads(note_path.read_text(encoding="utf-8")) if note_path.exists() else {}
    if note.get("status") != "complete":
        failures.append("A2's analysis has not run to completion (no complete a2_run_note.json)")
    inputs: Dict[str, Any] = {}
    floor: Dict[str, List[str]] = {}
    record["models"] = {}
    for model in MODELS:
        m: Dict[str, Any] = {}
        cells, reports, a2_out = a2_inputs(note, model)
        m["a2_cells"], m["a2_reports"], m["a2_output"] = cells, reports, a2_out
        if not cells or not reports or not a2_out:
            failures.append(f"{model}: A2's run note does not name its cells, reports and output")
        elif not (root / a2_out).exists() or sha256(root / a2_out) != (note.get("outputs") or {}).get(a2_out):
            failures.append(f"{model}: A2 output {a2_out} is missing or differs from the run note")
        a2_pre = root / REPRO / f"a2_prerun_{model}.json"
        pre = json.loads(a2_pre.read_text(encoding="utf-8")).get("sha256", {}) if a2_pre.exists() else {}
        for rel in cells + reports:
            p = root / rel
            if not p.exists():
                failures.append(f"{model}: A2 input {rel} is missing")
            elif pre.get(rel) != sha256(p):
                failures.append(f"{model}: A2 input {rel} differs from A2's pre-run record")
            else:
                hashed.append(p)

        stage0 = root / REPRO / f"a7_stage0_{model}.json"
        if s0.load_table(stage0) is None:
            failures.append(f"{model}: Stage 0 table missing or not this design's")
        else:
            hashed.append(stage0)

        shards = [Path("results/a7") / f"a7_cells_{model}_shard{k}.jsonl" for k in range(N_SHARDS)]
        seen: Dict[Tuple[str, str], int] = {}
        n_rows, n_failed, problems = 0, 0, []
        for rel in shards:
            p = root / rel
            if not p.exists():
                failures.append(f"{model}: Stage 1 shard {rel} is missing")
                continue
            hashed.append(p)
            rows, bad = parse_rows(p)
            if bad:
                problems.append(f"{rel.name}: unparseable lines {bad[:20]}")
            for r in rows:
                n_rows += 1
                n_failed += r.get("status") != "ok"
                if r.get("a7_design_sha256") != a7_design.DESIGN_SHA256:
                    problems.append(f"{rel.name}: a row not from this design")
                if r.get("decision_date") != DECISION_DATE:
                    problems.append(f"{rel.name}: decision_date {r.get('decision_date')!r}")
                if r.get("input_name") != ROUTE:
                    problems.append(f"{rel.name}: input_name {r.get('input_name')!r}")
                key = (str(r.get("trial_id")), str(r.get("condition")))
                seen[key] = seen.get(key, 0) + 1
        dupes = sorted(k for k, n in seen.items() if n > 1)
        if dupes:
            problems.append(f"{len(dupes)} duplicated (trial, condition) keys, e.g. {dupes[:3]}")
        problems = sorted(set(problems))
        if problems:
            failures.append(f"{model}: Stage 1 cells: {problems[:10]}")
        m.update({"stage1_cells": [str(s) for s in shards], "n_rows": n_rows, "n_failed_rows": n_failed,
                  "n_items_seen": len({k[0] for k in seen})})

        if cells and reports and all((root / x).exists() for x in cells + reports) and (root / GEOMETRY).exists():
            run.check_geometry(root / GEOMETRY)
            geometry = json.loads((root / GEOMETRY).read_text(encoding="utf-8"))["items"]
            trials = {json.loads(l)["trial_id"] for c in cells
                      for l in (root / c).read_text(encoding="utf-8").splitlines() if l.strip()}
            rep = [json.loads((root / r).read_text(encoding="utf-8")) for r in sorted(reports)]
            across_values = [d["tokens_across_view"] for d in an.delivery_rows(rep, ROUTE) if "failed" not in d]
            across = float(np.median(across_values)) if across_values else None  # as A2's main
            floor[model] = sorted(an.below_token_floor({t: {} for t in sorted(trials)}, across, geometry))
            m["tokens_across_median"] = across
            m["n_below_token_floor"] = len(floor[model])
            hashed.append(root / GEOMETRY)
        else:
            failures.append(f"{model}: cannot compute the token-floor list (A2 inputs or geometry missing)")

        inputs[model] = {"stage0": str(REPRO / f"a7_stage0_{model}.json"), "stage1_cells": [str(s) for s in shards],
                         "a2_output": a2_out, "a2_cells": cells, "decisions": DECISIONS}
        record["models"][model] = m
        if a2_out and (root / a2_out).exists():
            hashed.append(root / a2_out)
    inputs["below_token_floor"] = floor
    hashed += [root / f for f in ("scripts/a7_analyse.py", "scripts/a7_design.py", "scripts/a7_stage0.py",
                                  "scripts/analyse_stimulus_interventions.py", "research/PREREG_A7_DRAFT.md")]
    record["sha256"] = {str(p.relative_to(root)): sha256(p) for p in dict.fromkeys(hashed) if p.exists()}
    record["failures"] = failures
    return record, (None if failures else inputs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)
    root = args.root.resolve()
    record, inputs = check(root)
    record["script_sha256"] = sha256(Path(__file__))
    if inputs is not None:
        text = json.dumps(inputs, indent=1) + "\n"
        if args.inputs.exists() and args.inputs.read_text(encoding="utf-8") != text:
            # A rerun after a crashed analysis step may rewrite the same file; a different one is a fault.
            record["failures"].append(f"{args.inputs} exists with different contents")
        else:
            args.inputs.write_text(text, encoding="utf-8")
            record["inputs_sha256"] = sha256(args.inputs)
    args.record.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for failure in record["failures"]:
        print("FAIL:", failure)
    return 1 if record["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
