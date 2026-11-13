#!/usr/bin/env python3
"""Pre-run checks for A8's runs of A2's and A4's frozen analyses.

Source: `PREREG_A8_DRAFT.md`, dated note of 2026-11-13, "A2's and A4's rulings of 2026-11-12 carried
over to A8", with its addendum. Run on the cluster from the project root, after the step's A8 jobs have
finished. Like A2's and A4's own checks, it reads cell keys, status, design hash, pixel shapes and whether
log probabilities are finite, and no other value.

The record goes to a fixed path. The single analyses refuse to run unless that record shows no failure
and the files they read are the bytes it hashed:
- A2: `research/reproduction/a8/<a8_id>/a2_prerun_check.json`, read by `a8_analyse_a2.py`;
- A4: `research/reproduction/a8/a4_prerun_check.json`, read by the launcher's `a4_analyse` step.

  a2 --a8_id <id>
      A2 rulings 2, 3, 6 and 7, on A8's own inputs:
      - each shard is backed by an ok launcher record, and so are the gate and delivery jobs;
      - the cells lines parse;
      - the pixel shapes match the model's own A8 delivery record;
      - A8's gate files exist, and its A1 and A3 summaries pass their provenance check;
      - the geometry matches its pin.
      The small siblings' gate files are never read.

  a4 [--apply_filter] [--no_result <id> ...] [--deferred <id> ... --deferral_note <text>]
      A4 rulings 1 to 7, as the dated note reads them. A model must be accounted for if it is eligible
      for A4 by the frozen rules, has an a4_run launcher record, or has shard files. Each such model is
      complete for its launched shard count, with A4's own row check passing; or it is set aside with
      --no_result, allowed only when a shard has spent its out-of-memory retries (dated note 2026-11-09);
      or it is deferred with --deferred, allowed only when no A4 job or cell exists for it, naming the
      dated note that defers it. Inputs must match their pins, the gates file must be what
      a8_a4_gates.py builds (with its provenance), and BLAS must be single-threaded.
      Files move only after every other check has passed, never onto an existing file:
      - --no_result moves the model's shard files, unchanged, to results/a8/a4/no_result/;
      - --apply_filter keeps each original that holds non-finite rows, unchanged, in
        results/a8/a4/unfiltered/, then puts the filtered copy under the original name in one rename.
      Every record hashes the files in both subfolders.

Usage (project root, on the cluster, once A8's code is pinned):
    PYTHONPATH=.:src python research/scripts/a8_prerun_check.py a2 --a8_id qwen3vl_32b
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=.:src python research/scripts/a8_prerun_check.py a4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402
import a8_family  # noqa: E402

ROOT = a8_design.ROOT
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CELLS_A4 = a8_family.A4_CELLS
MANIFEST_A4 = "results/a4/manifest.csv"
A4_PREREG = "research/PREREG_A4_DRAFT.md"
GATES_A4 = "research/reproduction/a8/a4_gates.json"
SNOEK_ZIP = "data/external/snoek_au/au_emotion_data.zip"  # read by the frozen a4_analyse by default; recorded
UNFILTERED = "unfiltered"
NO_RESULT = "no_result"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_a8", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a2pre = _load("a2_prerun_check")
a4pre = _load("a4_prerun_check")


def _rel(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _hashes(paths: Sequence[Path], base: Path) -> Dict[str, str]:
    return {_rel(p, base): a2pre.sha256(p) for p in paths if Path(p).is_file()}


# A2 -------------------------------------------------------------------------


def check_a2(a8_id: str, base: Path = ROOT) -> Dict[str, Any]:
    """A2 rulings 2, 3, 6 and 7 on one A8 model's own inputs."""
    import a8_analyse_a2 as a2w

    design = a8_design.load_design()
    record: Dict[str, Any] = {"step": "a2", "a8_id": a8_id}
    try:
        entry = a8_design.model(design, a8_id)
    except SystemExit as exc:
        return {**record, "failures": [str(exc)]}
    key = record["key"] = entry["code_key"]
    if not (base / a2pre.ROOT_MARKER).exists():
        return {**record, "failures": ["not run from the project root"]}
    failures: List[str] = []
    try:
        names_cells, names_reports, used = a2w.shard_inputs(design, base, a8_id)
        record["launch_records_used"] = used
    except SystemExit as exc:
        failures.append(f"launcher records: {exc}")
        names_cells, names_reports = [], []
    cells, reports = [Path(c) for c in names_cells], [Path(r) for r in names_reports]
    record["analysed_files"] = [_rel(p, base) for p in cells + reports]
    missing = [p.name for p in cells + reports if not p.exists()]
    if missing:
        failures.append(f"shard files missing: {missing}")
    present = [p for p in cells if p.exists()]
    bad = {p.name: a2pre.unparseable_lines(p) for p in present}
    record["unparseable_lines"] = {k: v for k, v in bad.items() if v}
    if record["unparseable_lines"]:
        failures.append("unparseable cells lines: apply A2's repair rule (A2 ruling 2)")
    repro = a8_family.repro(base, a8_id)
    delivery, gate_file = repro / f"input_delivery_{key}.json", repro / f"vision_gates_{key}.json"
    summaries = {"A3": base / a8_family.A3_SUMMARY, "A1": base / a8_family.A1_SUMMARY}
    for name, path in summaries.items():
        try:
            a8_family.check_provenance(path)
        except SystemExit as exc:
            failures.append(f"A8 {name} summary: {exc}")
    for path in (delivery, gate_file):
        if not path.exists():
            failures.append(f"gate input missing: {_rel(path, base)}")
    allowed = a2pre.model_dims(json.loads(delivery.read_text(encoding="utf-8"))) if delivery.exists() else set()
    record["allowed_pixel_last_dims"] = sorted(allowed)
    foreign = a2pre.foreign_rows(present, allowed)
    record["n_rows_not_matching_model"] = len(foreign)
    record["rows_not_matching_model"] = foreign[:50]
    if foreign:
        failures.append(f"{len(foreign)} ok rows have pixel shapes not in {key}'s A8 delivery record (A2 ruling 3)")
    geometry = base / a2pre.GEOMETRY
    if not geometry.exists():
        failures.append("geometry.json is missing")
    else:
        from scripts import run_stimulus_interventions as run  # frozen; only its pinned hash is used here

        if a2pre.sha256(geometry) != run.GEOMETRY_SHA256:
            failures.append("geometry.json does not match its pin")
    files = [base / p for p in a2pre.PINNED] + list(summaries.values()) + [delivery, gate_file, geometry] + cells + reports
    record["sha256"] = _hashes(files, base)
    return {**record, "failures": failures}


# A4 -------------------------------------------------------------------------


def numbered(folder: Path, key: str) -> List[Path]:
    """The shard files the frozen analysis reads: cells_<key>_<n>.jsonl at the top of the folder."""
    prefix = f"cells_{key}_"
    return sorted(p for p in folder.glob(f"{prefix}*.jsonl") if p.stem[len(prefix):].isdigit())


def launched_shards(base: Path, a8_id: str, weights: str) -> Tuple[int, Dict[int, str]]:
    """The run setting: the shard count of the model's ok a4_run records, and an ok record per shard."""
    records = [r for r in a8_family.launch_records(base, a8_id, "a4_run") if r.get("status") == "ok"]
    if not records:
        raise SystemExit(f"{a8_id}: no a4_run job finished ok.")
    counts = {(r.get("context") or {}).get("n_shards") for r in records}
    if len(counts) != 1 or not isinstance(next(iter(counts)), int):
        raise SystemExit(f"{a8_id}: a4_run records use shard counts {sorted(map(str, counts))}, not one.")
    n = counts.pop()
    used = {k: a8_family.ok_record(base, a8_id, "a4_run", weights, {"shard": k, "n_shards": n})["started_at_utc"]
            for k in range(n)}
    return n, used


def filtered_bytes(original: bytes, name: str, drop: Sequence[Mapping[str, Any]]) -> bytes:
    """As A4's write_clean filters one file: the lines listed for it are left out, the rest kept as written."""
    dropped = {r["line"] for r in drop if r["file"] == name}
    kept = [raw for n, raw in enumerate(original.split(b"\n"), start=1) if raw.strip() and n not in dropped]
    return b"\n".join(kept) + b"\n"


def check_a4_model(base: Path, a8_id: str, key: str, weights: str, manifest: List[Dict[str, Any]],
                   design_hash: str, run) -> Dict[str, Any]:
    """A4's own check_model at the launched shard count, on the original of every shard (the copy kept in
    unfiltered/ where one exists). A filtered copy must equal its original filtered; an original whose
    filtered copy is not written yet, or whose filter was interrupted, is listed as pending."""
    folder = base / CELLS_A4
    staging = folder / UNFILTERED
    out: Dict[str, Any] = {"failures": [], "non_finite_rows": [], "pending_filter": []}
    try:
        n, out["launch_records_used"] = launched_shards(base, a8_id, weights)
    except SystemExit as exc:
        return {**out, "failures": [f"launcher records: {exc}"]}
    out["n_shards"] = n
    top = {p.name: p for p in numbered(folder, key)}
    kept = {p.name: p for p in numbered(staging, key)}
    with tempfile.TemporaryDirectory(prefix="a8_a4_originals_") as tmp:
        view = Path(tmp)  # links to the original of every shard, outside the results folder
        for name in sorted(set(top) | set(kept)):
            (view / name).symlink_to((kept.get(name) or top[name]).resolve())
        saved = a4pre.N_SHARDS
        a4pre.N_SHARDS = n  # the run setting replaces A4's 12; the check is otherwise A4's own, unchanged
        try:
            result = a4pre.check_model(key, view, manifest, design_hash, run)
        finally:
            a4pre.N_SHARDS = saved
    out["shards"] = result["shards"]
    out["failures"] += result["failures"]
    out["non_finite_rows"] = result["non_finite_rows"]
    with_rows = {r["file"] for r in result["non_finite_rows"]}
    for name in sorted(set(top) | set(kept)):
        if name not in kept:
            if name in with_rows:
                out["pending_filter"].append(name)
            continue
        original = kept[name].read_bytes()
        current = top[name].read_bytes() if name in top else None
        if current == filtered_bytes(original, name, result["non_finite_rows"]):
            continue
        if current is None or current == original:
            out["pending_filter"].append(name)  # interrupted before the rename: --apply_filter completes it
        else:
            out["failures"].append(f"{name}: differs from its original in {UNFILTERED}/ with the non-finite rows removed")
    out["filtered_earlier"] = sorted(name for name in kept if name not in out["pending_filter"])
    return out


def filter_in_place(folder: Path, drop: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Ruling 2 under A8's fixed --cells_dir. The original is kept, unchanged, in unfiltered/ first; the
    filtered copy then takes its name in one rename, so an interruption never leaves a shard name empty,
    and a rerun completes it."""
    staging = folder / UNFILTERED
    staging.mkdir(exist_ok=True)
    done = {}
    for name in names:
        original, kept = folder / name, staging / name
        if not kept.exists():
            tmp_kept = staging / f".{name}.copy"
            shutil.copy2(original, tmp_kept)
            os.replace(tmp_kept, kept)
        tmp = folder / f".{name}.filtered"
        tmp.write_bytes(filtered_bytes(kept.read_bytes(), name, drop))
        os.replace(tmp, original)
        done[name] = {"original_sha256": a2pre.sha256(kept), "filtered_sha256": a2pre.sha256(original)}
    (staging / "removed_non_finite_rows.json").write_text(json.dumps(list(drop), indent=1) + "\n", encoding="utf-8")
    return done


def set_aside_plan(base: Path, design: Mapping[str, Any], a8_id: str) -> Dict[str, Any]:
    """Dated note 2026-11-09 for A4, checked without moving anything: a shard of the model has spent its
    out-of-memory retries, and no file would overwrite one already in no_result/."""
    key = a8_design.model(design, a8_id)["code_key"]
    shards = sorted({(r.get("context") or {}).get("shard") for r in a8_family.launch_records(base, a8_id, "a4_run")}
                    - {None})
    spent = [k for k in shards if a8_family.oom_retries_spent(base, a8_id, "a4_run", {"shard": k})]
    if not spent:
        raise SystemExit(f"--no_result {a8_id}: no shard of {a8_id} has spent its out-of-memory retries "
                         "(dated note 2026-11-09); its A4 cannot be set aside.")
    folder = base / CELLS_A4
    files = [p.name for p in numbered(folder, key)]
    clash = [name for name in files if (folder / NO_RESULT / name).exists()]
    if clash:
        raise SystemExit(f"--no_result {a8_id}: {clash} already in {NO_RESULT}/; nothing is overwritten.")
    return {"a8_id": a8_id, "shards_out_of_retries": spent, "files": files}


def apply_set_aside(base: Path, plan: Mapping[str, Any]) -> Dict[str, Any]:
    folder = base / CELLS_A4
    target = folder / NO_RESULT
    target.mkdir(exist_ok=True)
    moved = {}
    for name in plan["files"]:
        moved[name] = a2pre.sha256(folder / name)
        os.replace(folder / name, target / name)
    return {**plan, "moved_sha256": moved}


def a4_members(design: Mapping[str, Any], base: Path, failures: List[str]) -> Dict[str, List[str]]:
    """Every A8 model A8's A4 run must account for, with why: eligible for A4 by the frozen rules (block
    rule a2_a4_only_if_eligible), launched for it, or holding shard files."""
    folder = base / CELLS_A4
    members: Dict[str, List[str]] = {}
    for a8_id in a8_design.active_models(design):
        key = design["models"][a8_id]["code_key"]
        why = []
        try:
            if a8_family.eligibility_decision(design, base, a8_id, "a4_run")["eligible"]:
                why.append("eligible")
        except SystemExit as exc:
            failures.append(f"{a8_id}: its A4 eligibility cannot be read yet: {exc}")
        if a8_family.launch_records(base, a8_id, "a4_run"):
            why.append("launched")
        if numbered(folder, key):
            why.append("cells")
        if why:
            members[a8_id] = why
    return members


def check_a4(base: Path = ROOT, apply_filter: bool = False, no_result: Sequence[str] = (),
             deferred: Sequence[str] = (), deferral_note: str = "") -> Dict[str, Any]:
    import pandas
    import a8_a4_gates
    from scripts import a4_design
    from scripts import a4_run as run

    design = a8_design.load_design()
    failures: List[str] = []
    record: Dict[str, Any] = {"step": "a4", "threads": {v: os.environ.get(v) for v in a4pre.THREAD_VARS},
                              "a4_design_sha256": a4_design.design_hash(), "a8_design_sha256": a8_design.block_hash()}
    if any(os.environ.get(v) != "1" for v in a4pre.THREAD_VARS):
        failures.append("BLAS is not single-threaded: set " + ", ".join(f"{v}=1" for v in a4pre.THREAD_VARS))
    for a8_id in (*no_result, *deferred):
        try:
            a8_design.model(design, a8_id)
        except SystemExit as exc:
            failures.append(str(exc))
    if set(no_result) & set(deferred):
        failures.append(f"{sorted(set(no_result) & set(deferred))} given as both --no_result and --deferred")
    if deferred and not deferral_note.strip():
        failures.append("--deferred needs --deferral_note naming the dated note that defers A4")
    try:
        a8_design.check_data(design, [MANIFEST_A4], root=base)
    except (SystemExit, OSError) as exc:
        failures.append(f"manifest: {exc}")
    try:
        pins = a4pre.pinned_inputs(base / A4_PREREG)
    except (SystemExit, OSError) as exc:
        failures.append(f"A4's input pins: {exc}")
        pins = {}
    for name, digest in pins.items():
        if not (base / name).is_file() or a2pre.sha256(base / name) != digest:
            failures.append(f"{name} does not match its pin in {A4_PREREG}")
    for name, path in (("A3", base / a8_family.A3_SUMMARY), ("A1", base / a8_family.A1_SUMMARY)):
        try:
            a8_family.check_provenance(path)
        except SystemExit as exc:
            failures.append(f"A8 {name} summary: {exc}")
    gates = base / GATES_A4
    try:
        if not gates.exists() or json.loads(gates.read_text(encoding="utf-8")) != a8_a4_gates.gates(base):
            failures.append(f"{GATES_A4} missing or not what a8_a4_gates.py builds from A8's outputs")
        else:
            a8_family.check_provenance(gates)
    except (OSError, ValueError, SystemExit) as exc:
        failures.append(f"{GATES_A4}: {exc}")
    manifest_path = base / MANIFEST_A4
    manifest = pandas.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
    folder = base / CELLS_A4
    members = a4_members(design, base, failures)
    record["members"] = members
    record["models"], record["set_aside"], record["deferred"] = {}, [], {}
    plans, drop, pending = [], [], []
    for a8_id in (*no_result, *deferred):
        if a8_id in design["models"] and a8_id not in members:
            failures.append(f"{a8_id}: not eligible, launched or holding cells for A4; nothing to set aside or defer")
    for a8_id, why in members.items():
        key = design["models"][a8_id]["code_key"]
        if a8_id in no_result:
            try:
                plans.append(set_aside_plan(base, design, a8_id))
            except SystemExit as exc:
                failures.append(str(exc))
            continue
        if a8_id in deferred:
            if "launched" in why or "cells" in why:
                failures.append(f"{a8_id}: --deferred, but A4 jobs or cells exist for it")
            else:
                record["deferred"][a8_id] = deferral_note
            continue
        if "launched" not in why:
            failures.append(f"{a8_id}: eligible for A4 but no a4_run job has started; wait for it, or pass --deferred "
                            "with the dated note that defers it")
            continue
        result = check_a4_model(base, a8_id, key, design["models"][a8_id]["weights_dir"], manifest,
                                a4_design.design_hash(), run)
        failures += [f"{a8_id}: {f}" for f in result["failures"]]
        drop += result["non_finite_rows"]
        pending += result["pending_filter"]
        record["models"][a8_id] = {k: v for k, v in result.items() if k not in ("failures", "non_finite_rows")}
        record["models"][a8_id]["n_non_finite_rows"] = len(result["non_finite_rows"])
    if not record["models"]:
        failures.append("no A8 model is left to analyse for A4")
    record["non_finite_rows"] = drop
    if pending and not apply_filter:
        failures.append(f"{len(drop)} rows with non-finite log probabilities; {pending} not yet filtered: rerun with "
                        "--apply_filter (A4 ruling 2, as the A8 dated note of 2026-11-13 applies it)")
    if failures:
        if plans or (pending and apply_filter):
            failures.append("no file was moved or filtered, because other checks fail")
    else:
        # Files move only now, when every other check has passed.
        record["set_aside"] = [apply_set_aside(base, plan) for plan in plans]
        if pending:
            record["filtered"] = filter_in_place(folder, drop, pending)
    analysed = a8_family.a4_cells_files(base)
    record["analysed_files"] = [_rel(p, base) for p in analysed]
    files = [manifest_path, gates, base / a8_family.A3_SUMMARY, base / a8_family.A1_SUMMARY, base / SNOEK_ZIP]
    files += [base / name for name in pins]
    files += analysed + sorted((folder / UNFILTERED).glob("*")) + sorted((folder / NO_RESULT).glob("*"))
    record["sha256"] = _hashes(files, base)
    return {**record, "failures": failures}


# Entry point -----------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    import numpy
    import pandas
    import scipy

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    a2 = sub.add_parser("a2")
    a2.add_argument("--a8_id", required=True)
    a4 = sub.add_parser("a4")
    a4.add_argument("--apply_filter", action="store_true")
    a4.add_argument("--no_result", nargs="*", default=[])
    a4.add_argument("--deferred", nargs="*", default=[])
    a4.add_argument("--deferral_note", default="")
    args = ap.parse_args(argv)
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    pins = a8_design.check_own_code(require=True)  # like every A8 entry point: this check also moves inputs
    if args.step == "a2":
        result = check_a2(args.a8_id, ROOT)
        path = ROOT / a8_family.A2_PRERUN.format(a8_id=args.a8_id)
    else:
        result = check_a4(ROOT, args.apply_filter, args.no_result, args.deferred, args.deferral_note)
        path = ROOT / a8_family.A4_PRERUN
    result.update(command=sys.argv, a8_code_pins=pins, script_sha256=a2pre.sha256(Path(__file__)),
                  versions={"numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
                            "python": sys.version.split()[0]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    for failure in result["failures"]:
        print("FAIL:", failure)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
