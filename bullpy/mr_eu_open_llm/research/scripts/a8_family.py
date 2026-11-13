"""A8 families, computed by rule from the A8 output files. No hand-picked model list.

Paths are relative to a base folder (the project root in production, a temporary folder in
tests). Every rule here is the design block's `families` text in code:

- A3: active models whose single_face delivery passed. Each must have a complete A3 cell
  file. A failed delivery means "not tested". A missing delivery file stops the analysis.
- A1: active models with G2 to G4 passed and A3 vision_dependent true, the models the stop
  rule lets run A1. Each must have all 10 merged files, and a model outside the family must
  have none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import a8_design

ROTATIONS = range(5)
CONDITIONS = ("real", "mismatch")
N_ITEMS = 243


def repro(base: Path, a8_id: str) -> Path:
    return base / "research/reproduction/a8" / a8_id


def a3_cells_path(base: Path, a8_id: str, key: str) -> Path:
    return base / "results/a8" / a8_id / "face_input_validity" / f"cells_{key}.jsonl"


def merged_path(base: Path, a8_id: str, rotation: int, condition: str) -> Path:
    return base / "results/a8" / a8_id / "rotation" / f"eval_v2_eu_emotions_{a8_id}_video_only_seed42_rot{rotation}_{condition}.json"


A3_SUMMARY = "research/reproduction/a8/face_input_validity_summary_a8.json"
A1_SUMMARY = "research/reproduction/a8/rotation_summary_a8.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def gates_passed(base: Path, a8_id: str, key: str) -> bool:
    """As the frozen outside_gates reads them: not a smoke, and G2, G3, G4 all pass."""
    vision = _json(repro(base, a8_id) / f"vision_gates_{key}.json") or {}
    verdicts = vision.get("verdict") or {}
    return not vision.get("smoke", True) and all((verdicts.get(g) or {}).get("pass") is True for g in ("G2", "G3", "G4"))


def single_face_delivery(base: Path, a8_id: str, key: str):
    """True or False from the delivery file; None when the file does not exist."""
    data = _json(repro(base, a8_id) / f"input_delivery_{key}.json")
    if data is None:
        return None
    return (data.get("verdicts", {}).get("single_face") or {}).get("pass") is True


def check_a3_cells_complete(path: Path) -> None:
    """243 items by 10 cells, unique keys, failed cells present, no partial line."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist; A3 has not run for this model.")
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise SystemExit(f"{path} ends in a partial line; the job did not finish.")
    rows = [json.loads(line) for line in data.splitlines() if line]
    keys = [(r["trial_id"], r["rotation"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{path} holds duplicate cells.")
    items = {k[0] for k in keys}
    if len(items) != N_ITEMS or len(keys) != N_ITEMS * len(ROTATIONS) * len(CONDITIONS):
        raise SystemExit(f"{path} holds {len(items)} items and {len(keys)} cells, not {N_ITEMS} and {N_ITEMS * 10}.")


def launch_records(base: Path, a8_id: str, step: str) -> List[Dict[str, Any]]:
    """Launcher records of one step for one model, oldest first."""
    folder = base / "research/reproduction/a8" / a8_id / "launch"
    if not folder.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob(f"{step}_*.json"))]


def ok_record(base: Path, a8_id: str, step: str, weights_dir: str, context: Mapping[str, Any] = None) -> Dict[str, Any]:
    """The launcher record of a finished job of this step, with the design's weights path."""
    want = str(Path(weights_dir).resolve())
    for record in reversed(launch_records(base, a8_id, step)):
        if record.get("status") != "ok":
            continue
        if context and any(record.get("context", {}).get(k) != v for k, v in context.items()):
            continue
        if (record.get("loader") or {}).get("resolve_model_path") != want:
            raise SystemExit(f"{a8_id} {step}: the job loaded {(record.get('loader') or {}).get('resolve_model_path')}, not {want}.")
        return record
    raise SystemExit(f"{a8_id} {step}{' ' + str(dict(context)) if context else ''}: no launcher record with status ok.")


def not_tested(base: Path, a8_id: str, step: str) -> bool:
    """Design rule not_tested_after_retry: two launched attempts, none ok."""
    records = launch_records(base, a8_id, step)
    return len(records) >= 2 and not any(r.get("status") == "ok" for r in records)


def oom_retries_spent(base: Path, a8_id: str, step: str, context: Mapping[str, Any] = None) -> bool:
    """Dated note 2026-11-09 (`PREREG_A8_DRAFT.md`, "Dated notes"): no attempt ended ok, and two
    attempts at the largest GPU count used for the step ended out of memory. The launcher records
    the GPU count of every attempt and marks an out-of-memory crash as out_of_memory. With a context
    (for example one shard), only that job's attempts count."""
    records = [r for r in launch_records(base, a8_id, step)
               if not context or all((r.get("context") or {}).get(k) == v for k, v in context.items())]
    if any(r.get("status") == "ok" for r in records):
        return False
    counts = [r["visible_gpus"] for r in records if isinstance(r.get("visible_gpus"), int)]
    if not counts:
        return False
    top = max(counts)
    return sum(r.get("visible_gpus") == top and r.get("status") == "out_of_memory" for r in records) >= 2


def a3_family(design: Mapping[str, Any], base: Path, check_launch: bool = True) -> Tuple[List[Tuple[str, Path]], Dict[str, str]]:
    """(name, cells file) for the A3 family, in block order, and the excluded with reasons."""
    family, excluded = [], {}
    for a8_id in a8_design.active_models(design):
        entry = design["models"][a8_id]
        key = entry["code_key"]
        delivered = single_face_delivery(base, a8_id, key)
        if delivered is None:
            if not_tested(base, a8_id, "delivery"):
                excluded[a8_id] = "delivery job did not finish after one retry: single_face delivery failed, not tested"
                continue
            raise SystemExit(f"{a8_id} has no delivery file; the A3 family cannot be set yet.")
        if not delivered:
            excluded[a8_id] = "single_face delivery failed: not tested (PREREG_A3_DRAFT.md, Preconditions)"
            continue
        path = a3_cells_path(base, a8_id, key)
        if not path.exists() and not_tested(base, a8_id, "a3"):
            excluded[a8_id] = "A3 job did not finish after one retry: no A3 result"
            continue
        if path.exists() and oom_retries_spent(base, a8_id, "a3"):
            excluded[a8_id] = ("A3 cell file incomplete after two out-of-memory attempts at the largest GPU count used: "
                               "no A3 result (PREREG_A8_DRAFT.md, dated note 2026-11-09)")
            continue
        check_a3_cells_complete(path)
        if check_launch:
            ok_record(base, a8_id, "a3", entry["weights_dir"])
        family.append((a8_id, path))
    return family, excluded


def a1_family(design: Mapping[str, Any], base: Path) -> Tuple[List[str], Dict[str, str]]:
    """The A1 family by the stop rule, from the gate files and the A8 A3 summary."""
    a3 = _json(base / A3_SUMMARY)
    if a3 is None:
        raise SystemExit(f"{A3_SUMMARY} does not exist; the A1 family needs the A3 result.")
    family, excluded = [], {}
    for a8_id in a8_design.active_models(design):
        key = design["models"][a8_id]["code_key"]
        files = [merged_path(base, a8_id, r, c) for r in ROTATIONS for c in CONDITIONS]
        present = [p for p in files if p.exists()]
        runs = gates_passed(base, a8_id, key) and (a3.get(a8_id) or {}).get("vision_dependent") is True
        if runs:
            if len(present) != len(files):
                raise SystemExit(f"{a8_id} is in the A1 family but has {len(present)} of {len(files)} merged files.")
            family.append(a8_id)
        else:
            if present:
                raise SystemExit(f"{a8_id} is outside the A1 family by the stop rule, yet has A1 files.")
            excluded[a8_id] = "A1 not run, by the A8 stop rule (G2 to G4 not passed, or A3 not Pass)"
    return family, excluded


def check_a1_run_settings(base: Path, family: List[str], design: Mapping[str, Any] = None) -> Dict[str, int]:
    """All 10 merged files of a model share one GPU count, came from production shards, and
    (given the design) were produced from the design's weights."""
    out = {}
    for a8_id in family:
        counts, test = set(), False
        want = str(Path(design["models"][a8_id]["weights_dir"]).resolve()) if design else None
        for r in ROTATIONS:
            for c in CONDITIONS:
                merged = json.loads(merged_path(base, a8_id, r, c).read_text(encoding="utf-8"))["a8_merged"]
                for side in merged["shard_sidecars"]:
                    counts.add(side.get("visible_gpus"))
                    test = test or bool(side.get("test_mode"))
                    if want is not None and (side.get("resolved_model_path") != want or side.get("mode") != "production"):
                        raise SystemExit(f"{a8_id} rot{r} {c}: a shard was not a production run on {want}.")
        if test:
            raise SystemExit(f"{a8_id} has test-mode shards in its A1 files.")
        if len(counts) != 1:
            raise SystemExit(f"{a8_id} A1 files were run with different GPU counts {sorted(map(str, counts))}.")
        out[a8_id] = counts.pop()
    return out


def provenance_path(output: Path) -> Path:
    return Path(str(output) + ".provenance.json")


def portable(path: Path) -> str:
    """A path under the project root is stored relative to it, so a provenance record written on the
    cluster can be checked after a pull to the Mac, and the other way round. Other paths (test folders)
    are stored as given."""
    try:
        return Path(path).resolve().relative_to(a8_design.ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def located(name: str) -> Path:
    """The inverse of `portable`: a relative name is read under the project root."""
    p = Path(name)
    return p if p.is_absolute() else a8_design.ROOT / p


def write_provenance(output: Path, inputs: List[Path], extra: Dict[str, Any]) -> Path:
    import hashlib

    record = {
        "output": portable(output),
        "output_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        "inputs_sha256": {portable(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
        **extra,
    }
    path = provenance_path(output)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_provenance(output: Path) -> Dict[str, Any]:
    """The output is the one its producer recorded, and its inputs have not changed since."""
    import hashlib

    path = provenance_path(output)
    if not Path(output).exists() or not path.exists():
        raise SystemExit(f"{output} or its provenance is missing; run its A8 producer first.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if hashlib.sha256(Path(output).read_bytes()).hexdigest() != record["output_sha256"]:
        raise SystemExit(f"{output} differs from the file its producer wrote.")
    for name, digest in record["inputs_sha256"].items():
        path = located(name)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Input {name} of {output} changed after it was produced.")
    return record


def eligibility(design: Mapping[str, Any], base: Path, a8_id: str, step: str) -> Dict[str, Any]:
    """Design rule eligibility_checked_by_launcher, by the frozen rules, from A8 files only."""
    key = design["models"][a8_id]["code_key"]
    a3_path = base / A3_SUMMARY
    check_provenance(a3_path)
    a3 = (json.loads(a3_path.read_text(encoding="utf-8")).get(a8_id) or {}).get("vision_dependent")
    decision: Dict[str, Any] = {"step": step, "g2_g4": gates_passed(base, a8_id, key), "a3": a3}
    if step == "a1_shard":
        decision["eligible"] = decision["g2_g4"] and a3 is True
    elif step in ("a2_run", "a4_run"):
        a1_path = base / A1_SUMMARY
        check_provenance(a1_path)
        a1 = (json.loads(a1_path.read_text(encoding="utf-8")).get(a8_id) or {}).get("vision_dependent")
        from scripts.analyse_stimulus_interventions import rq0_status

        decision["a1"] = a1
        decision["rq0_single_face"] = rq0_status(a1, a3, "single_face")
        decision["eligible"] = decision["g2_g4"] and decision["rq0_single_face"]
        if step == "a2_run":
            decision["single_face_delivery"] = single_face_delivery(base, a8_id, key) is True
            decision["eligible"] = decision["eligible"] and decision["single_face_delivery"]
    else:
        raise SystemExit(f"No eligibility rule for {step}.")
    if not decision["eligible"]:
        raise SystemExit(f"{a8_id} is not eligible for {step} by the frozen rules: {decision}")
    return decision
