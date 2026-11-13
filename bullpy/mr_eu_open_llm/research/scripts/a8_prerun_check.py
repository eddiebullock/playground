#!/usr/bin/env python3
"""Pre-run checks for A8's runs of A2's and A4's frozen analyses.

Source: `PREREG_A8_DRAFT.md`, dated note of 2026-11-13, "A2's and A4's rulings of 2026-11-12 carried
over to A8". Run on the cluster from the project root, after the step's A8 jobs have finished. Like A2's
and A4's own checks, it reads cell keys, status, design hash, pixel shapes and whether log probabilities
are finite, and no other value. It writes the record to --record and exits 1 on any failure.

  a2 --a8_id <id>
      Runs before `research/scripts/a8_analyse_a2.py --a8_id <id>`, for A2 rulings 2, 3, 6 and 7.
      It checks A8's own inputs only:
      - every shard is backed by an ok launcher record, as are the gate and delivery jobs;
      - the cells lines parse;
      - the pixel shapes match the model's own A8 delivery record;
      - A8's gate files exist, and its A1 and A3 summaries pass their provenance check;
      - the geometry matches its pin;
      - the sha256 of each of these is recorded.
      The small siblings' gate files are never read.

  a4 [--apply_filter] [--no_result <a8_id> ...]
      Runs before the launcher's `a4_analyse` step, for A4 rulings 1 to 7 as the dated note reads them:
      - every model with cells in results/a8/a4 is complete for its launched shard count;
      - each shard is backed by an ok a4_run record;
      - rows are checked by A4's own `check_model`;
      - inputs match their pins;
      - the gates file is what a8_a4_gates.py builds;
      - BLAS is single-threaded.
      --apply_filter moves each shard file that holds non-finite rows, unchanged, to
      results/a8/a4/unfiltered/, and writes the filtered copy under the original name. A later check
      verifies the copy against its original.
      --no_result moves a model's shard files, unchanged, to results/a8/a4/no_result/. It is allowed only
      when a shard of that model has spent its out-of-memory retries (dated note 2026-11-09).

Usage (project root, on the cluster):
    PYTHONPATH=.:src python research/scripts/a8_prerun_check.py a2 --a8_id qwen3vl_32b \
        --record research/reproduction/a8/qwen3vl_32b/a2_prerun_check.json
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=.:src python research/scripts/a8_prerun_check.py a4 --record research/reproduction/a8/a4_prerun_check.json
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

CELLS_A4 = "results/a8/a4"
MANIFEST_A4 = "results/a4/manifest.csv"
A4_PREREG = "research/PREREG_A4_DRAFT.md"
GATES_A4 = "research/reproduction/a8/a4_gates.json"
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
    entry = a8_design.model(design, a8_id)
    key = entry["code_key"]
    record: Dict[str, Any] = {"step": "a2", "a8_id": a8_id, "key": key}
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


def a4_models(design: Mapping[str, Any], base: Path) -> List[Tuple[str, str]]:
    """(a8_id, key) for each active model with any shard file in results/a8/a4 (block `families.a4`)."""
    folder = base / CELLS_A4
    return [(m, design["models"][m]["code_key"]) for m in a8_design.active_models(design)
            if numbered(folder, design["models"][m]["code_key"])]


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
    """A4's check_model at the launched shard count, on the original files. A file already filtered by an
    earlier --apply_filter is checked on its original in unfiltered/, and must equal that original filtered."""
    folder = base / CELLS_A4
    out: Dict[str, Any] = {"failures": [], "non_finite_rows": []}
    try:
        n, out["launch_records_used"] = launched_shards(base, a8_id, weights)
    except SystemExit as exc:
        return {**out, "failures": [f"launcher records: {exc}"]}
    out["n_shards"] = n
    staging = folder / UNFILTERED
    originals = {p.name: p for p in numbered(folder, key)}
    filtered_now = {name for name in originals if (staging / name).exists()}
    with tempfile.TemporaryDirectory(prefix="a8_a4_originals_") as tmp:
        view = Path(tmp)  # links to the original of every shard, outside the results folder
        for name, path in originals.items():
            source = staging / name if name in filtered_now else path
            (view / name).symlink_to(source.resolve())
        saved = a4pre.N_SHARDS
        a4pre.N_SHARDS = n  # the run setting replaces A4's 12; the check is otherwise A4's own, unchanged
        try:
            result = a4pre.check_model(key, view, manifest, design_hash, run)
        finally:
            a4pre.N_SHARDS = saved
    out["shards"] = result["shards"]
    out["failures"] += result["failures"]
    out["non_finite_rows"] = result["non_finite_rows"]
    out["filtered_earlier"] = sorted(filtered_now)
    for name in filtered_now:
        want = filtered_bytes((staging / name).read_bytes(), name, result["non_finite_rows"])
        if (folder / name).read_bytes() != want:
            out["failures"].append(f"{name}: differs from its original in {UNFILTERED}/ with the non-finite rows removed")
    return out


def filter_in_place(folder: Path, drop: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Ruling 2 under A8's fixed --cells_dir: originals kept unchanged in unfiltered/, filtered copies in place."""
    staging = folder / UNFILTERED
    staging.mkdir(exist_ok=True)
    moved = {}
    for name in sorted({r["file"] for r in drop}):
        original, kept = folder / name, staging / name
        if kept.exists():
            continue  # filtered by an earlier run of this check, and verified above
        data = original.read_bytes()
        shutil.move(str(original), str(kept))
        original.write_bytes(filtered_bytes(data, name, drop))
        moved[name] = {"original_sha256": a2pre.sha256(kept), "filtered_sha256": a2pre.sha256(original)}
    log = staging / "removed_non_finite_rows.json"
    log.write_text(json.dumps(list(drop), indent=1) + "\n", encoding="utf-8")
    return moved


def set_aside(base: Path, design: Mapping[str, Any], a8_id: str) -> Dict[str, Any]:
    """Dated note 2026-11-09 for A4: a model with a shard out of out-of-memory retries has no A4 result. Its
    shard files are moved, unchanged, to no_result/, so the other models' analysis goes on."""
    key = design["models"][a8_id]["code_key"]
    shards = sorted({(r.get("context") or {}).get("shard") for r in a8_family.launch_records(base, a8_id, "a4_run")}
                    - {None})
    spent = [k for k in shards if a8_family.oom_retries_spent(base, a8_id, "a4_run", {"shard": k})]
    if not spent:
        raise SystemExit(f"--no_result {a8_id}: no shard of {a8_id} has spent its out-of-memory retries "
                         "(dated note 2026-11-09); its A4 cannot be set aside.")
    folder = base / CELLS_A4
    target = folder / NO_RESULT
    target.mkdir(exist_ok=True)
    moved = {}
    for path in numbered(folder, key):
        moved[path.name] = a2pre.sha256(path)
        shutil.move(str(path), str(target / path.name))
    return {"a8_id": a8_id, "shards_out_of_retries": spent, "moved_sha256": moved}


def check_a4(base: Path = ROOT, apply_filter: bool = False, no_result: Sequence[str] = ()) -> Dict[str, Any]:
    import pandas
    import a8_a4_gates
    from scripts import a4_design
    from scripts import a4_run as run

    design = a8_design.load_design()
    failures: List[str] = []
    record: Dict[str, Any] = {"step": "a4", "threads": {v: os.environ.get(v) for v in a4pre.THREAD_VARS}}
    if any(os.environ.get(v) != "1" for v in a4pre.THREAD_VARS):
        failures.append("BLAS is not single-threaded: set " + ", ".join(f"{v}=1" for v in a4pre.THREAD_VARS))
    record["set_aside"] = []
    for a8_id in no_result:
        try:
            record["set_aside"].append(set_aside(base, design, a8_id))
        except SystemExit as exc:
            failures.append(str(exc))
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
    models = a4_models(design, base)
    record["models"] = {}
    drop: List[Dict[str, Any]] = []
    if not models:
        failures.append(f"no model has A4 cells in {CELLS_A4}")
    for a8_id, key in models:
        result = check_a4_model(base, a8_id, key, design["models"][a8_id]["weights_dir"], manifest,
                                a4_design.design_hash(), run)
        failures += [f"{a8_id}: {f}" for f in result["failures"]]
        drop += result["non_finite_rows"]
        record["models"][a8_id] = {k: v for k, v in result.items() if k not in ("failures", "non_finite_rows")}
        record["models"][a8_id]["n_non_finite_rows"] = len(result["non_finite_rows"])
    record["non_finite_rows"] = drop
    folder = base / CELLS_A4
    pending = sorted({r["file"] for r in drop} - {p.name for p in (folder / UNFILTERED).glob("*.jsonl")})
    if pending and not apply_filter:
        failures.append(f"{len(drop)} rows with non-finite log probabilities in {pending}: rerun with --apply_filter "
                        "(A4 ruling 2, as the A8 dated note of 2026-11-13 applies it)")
    elif pending and failures:
        failures.append("--apply_filter not applied while other checks fail")
    elif drop:
        record["filtered"] = filter_in_place(folder, drop)
    files = [manifest_path, gates, base / a8_family.A3_SUMMARY, base / a8_family.A1_SUMMARY]
    files += [base / name for name in pins]
    files += sorted(folder.glob("cells_*_*.jsonl")) + sorted((folder / UNFILTERED).glob("*.jsonl"))
    record["sha256"] = _hashes(files, base)
    return {**record, "failures": failures}


# Entry point -----------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    import numpy
    import scipy

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    a2 = sub.add_parser("a2")
    a2.add_argument("--a8_id", required=True)
    a4 = sub.add_parser("a4")
    a4.add_argument("--apply_filter", action="store_true")
    a4.add_argument("--no_result", nargs="*", default=[])
    for p in (a2, a4):
        p.add_argument("--record", type=Path, required=True)
        p.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    base = args.root.resolve()
    if args.step == "a2":
        result = check_a2(args.a8_id, base)
    else:
        result = check_a4(base, args.apply_filter, args.no_result)
    result.update(command=sys.argv, script_sha256=a2pre.sha256(Path(__file__)),
                  versions={"numpy": numpy.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]})
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    for failure in result["failures"]:
        print("FAIL:", failure)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
