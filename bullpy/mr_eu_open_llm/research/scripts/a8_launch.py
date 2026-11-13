#!/usr/bin/env python
"""A8 launcher: the one entry point for every A8 job.

For one step and one model it does, in this order, in one process:
1. Loads the design (hash-checked), checks every frozen file and A8's own code pins.
2. Builds the frozen command line from the design's `commands` entry, so the arguments the
   frozen script receives are the ones the checks below were run on.
3. For a step that loads a model: sets MR_EU_HPC_MODELS_DIR from the design (refusing a
   different preset value), runs the weights preflight, then imports config and
   scripts.evaluate and asserts that `config.MODELS[key]["hpc_path"]` and
   `evaluate.resolve_model_path(key)` both resolve to the design's weights directory.
4. Checks the pinned data inputs of the step (block `data_by_step`), the storage stop rule,
   the software versions (enforced for the steps the design names), and runs the frozen
   decoder preflight where the frozen job runs it. Before a1_shard, a2_run and a4_run it
   checks eligibility by the frozen rules from the A8 files (block rule
   `eligibility_checked_by_launcher`) and refuses an ineligible model.
5. Requires every output to sit under the A8 roots, and a non-resumable output not to exist.
6. Hashes every frozen output of the sibling key before the job and again after, and fails
   the job if any changed.
7. Writes a "started" record, runs the frozen module (runpy) or the A8 shard runner, then
   fails the job if a trial, cell or delivery-row error field records an out-of-memory error
   (design rule `oom`; the gate file is not scanned).
8. Writes a provenance JSON under research/reproduction/a8/<a8_id>/launch/.

    python research/scripts/a8_launch.py --a8_id qwen3vl_32b --step gates --confirm_phase2_completed
    python research/scripts/a8_launch.py --a8_id qwen3vl_32b --step a1_shard --rotation 0 \
        --condition mismatch --shard 2 --n_shards 4 --confirm_phase2_completed
    python research/scripts/a8_launch.py --step a4_analyse --confirm_phase2_completed
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402
import a8_preflight as pre  # noqa: E402
import a8_family  # noqa: E402
import a8_rotation_shard as shard  # noqa: E402

ROOT = a8_design.ROOT
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
# The frozen jobs run `python -m ...` from the project root, so the root and src are importable.
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
BIG = 100 * 1024 * 1024


def fill(value: str, ctx: Mapping[str, Any]) -> str:
    return value.format(**ctx)


def command(design: Mapping[str, Any], step: str, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    spec = dict(design["commands"][step])
    extra = (spec.get("argv_extra_by_key") or {}).get(ctx.get("key"), [])
    spec["argv"] = [fill(a, ctx) for a in list(spec.get("argv", [])) + list(extra)]
    spec["outputs"] = [fill(o, ctx) for o in spec["outputs"]]
    if "declared_output_outside_roots" in spec:
        spec["declared_output_outside_roots"] = fill(spec["declared_output_outside_roots"], ctx)
    return spec


def frozen_output_files(patterns: Sequence[str], keys: Sequence[str], root: Path = ROOT) -> List[Path]:
    files = set()
    for key in keys:
        for pattern in patterns:
            files.update(Path(p) for p in glob.glob(str(root / pattern.format(key=key))) if Path(p).is_file())
    return sorted(files)


def snapshot(files: Sequence[Path]) -> Dict[str, str]:
    """sha256 of each frozen output; size, mtime and inode for files over 100 MB."""
    out = {}
    for path in files:
        stat = path.stat()
        if stat.st_size > BIG:
            out[str(path)] = f"stat:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}"
        else:
            out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def changed(before: Mapping[str, str], after_files: Sequence[Path]) -> List[str]:
    after = snapshot(after_files)
    return sorted(set(before) ^ set(after) | {k for k in before if k in after and before[k] != after[k]})


SCANNED_STEPS = ("a1_shard", "a3", "a2_run", "a4_run", "delivery")
SKIP_KEYS = ("float32_failed",)


def _rows(path: Path):
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    elif path.suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _failed_values(obj: Any):
    """Values of `failed` keys at any depth, never `float32_failed` or raw model text."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            if k == "failed" and isinstance(v, str):
                yield v
            else:
                yield from _failed_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _failed_values(v)


def error_texts(path: Path, step: str) -> List[str]:
    """Design rule oom: trial, cell and delivery-row error fields only."""
    texts: List[str] = []
    for obj in _rows(path):
        if not isinstance(obj, dict):
            continue
        if step == "a1_shard" and "trials" in obj:
            texts += [str(t.get("error")) for t in obj["trials"] if t.get("error")]
        elif path.suffix == ".jsonl":
            texts += [str(obj[k]) for k in ("reason", "error", "detail") if obj.get(k)]
        elif step == "a2_run" and path.name.startswith("run_"):
            texts += [str(f.get("reason")) for f in obj.get("failures", []) if isinstance(f, dict)]
            texts += list(_failed_values(obj.get("delivery", [])))
        elif step == "delivery":
            texts += list(_failed_values(obj))
    return texts


def oom_in_outputs(paths: Sequence[Path], step: str) -> List[str]:
    """Files under the outputs whose structured error fields record an out-of-memory error.

    Steps outside SCANNED_STEPS (gates, ladder, merges, analyses) are not scanned: the frozen
    gate script records its expected float32 out-of-memory error in `float32_failed`.
    """
    if step not in SCANNED_STEPS:
        return []
    hits = []
    for path in paths:
        files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else ([path] if path.exists() else [])
        for f in files:
            if f.suffix in (".json", ".jsonl") and any(shard.is_oom(x) for x in error_texts(f, step)):
                hits.append(str(f))
    return hits


def check_loader_path(key: str, weights_dir: Path) -> Dict[str, str]:
    """After MR_EU_HPC_MODELS_DIR is set: what config and the frozen loader resolve to."""
    for p in (str(ROOT / "src"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import config  # noqa: E402

    hpc = Path(config.MODELS[key]["hpc_path"]).resolve()
    shard.check_evaluate()
    from scripts.evaluate import resolve_model_path  # noqa: E402

    resolved = Path(resolve_model_path(key)).resolve()
    want = Path(weights_dir).resolve()
    if hpc != want or resolved != want:
        raise SystemExit(f"config resolves {hpc} and the frozen loader {resolved}; the design's weights are {want}.")
    return {"config_hpc_path": str(hpc), "resolve_model_path": str(resolved)}


def run_frozen(spec: Mapping[str, Any]) -> None:
    """Run a frozen script in this process, as `python -m <module>` or `python <path>` would.

    A frozen script may end with `sys.exit(main())`, as `same_clip_family_analysis.py` does. Run in
    this process, that exit would also end the launcher before the rest of the job (the move of a
    declared output, the final record), so a clean exit, code None or 0, counts as the script
    finishing. Any other code, or a refusal message, still fails the job.
    """
    saved = sys.argv
    try:
        if "module" in spec:
            sys.argv = [spec["module"], *spec["argv"]]
            runpy.run_module(spec["module"], run_name="__main__", alter_sys=True)
        else:
            sys.argv = [spec["path"], *spec["argv"]]
            runpy.run_path(str(ROOT / spec["path"]), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = saved


def run_frozen_step(spec: Mapping[str, Any], declared: Optional[str] = None) -> None:
    """A frozen script, then the move of an output it writes outside the A8 roots to its A8 path."""
    run_frozen(spec)
    if declared:
        target = ROOT / spec["outputs"][0]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(ROOT / declared, target)


def final_status(finished: bool, moved: Sequence[str], oom_files: Sequence[str], error: Optional[str]) -> str:
    """A job's status. A changed frozen output overrides everything. Design rule `oom` reads the
    structured error fields of the outputs; a job that crashed with an out-of-memory error (for
    example a CUDA error raised out of a frozen script that does not catch it) is recorded as
    out_of_memory too, so the dated note of 2026-11-09 (a step incomplete after two out-of-memory
    attempts at the largest GPU count) can be applied from the records alone."""
    if moved:
        return "frozen_output_changed"
    if oom_files or (not finished and shard.is_oom(error)):
        return "out_of_memory"
    return "ok" if finished else "failed"


def check_threads(step: str) -> Dict[str, Optional[str]]:
    """A4's analysis runs with single-threaded BLAS (A4 pre-data ruling 7, DECISIONS 2026-11-12);
    A8's a4_analyse step runs the same frozen analysis, so it is held to the same rule."""
    threads = {v: os.environ.get(v) for v in THREAD_VARS}
    if step == "a4_analyse" and any(value != "1" for value in threads.values()):
        raise SystemExit("a4_analyse runs with single-threaded BLAS: set " + ", ".join(f"{v}=1" for v in THREAD_VARS))
    return threads


def a1_shard_files(design: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Path]:
    template = design["commands"]["a1_shard"]["outputs"][0]
    return [ROOT / template.format(**{**ctx, "shard": k}) for k in range(int(ctx["n_shards"]))]


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a8_id")
    ap.add_argument("--step", required=True)
    ap.add_argument("--rotation", type=int, choices=range(5))
    ap.add_argument("--condition", choices=("real", "mismatch"))
    ap.add_argument("--shard", type=int)
    ap.add_argument("--n_shards", type=int)
    ap.add_argument("--confirm_phase2_completed", action="store_true",
                    help="Required: both Qwen A1 arrays and the frozen Phase 2 runs are COMPLETED.")
    args = ap.parse_args(argv)
    if not args.confirm_phase2_completed:
        raise SystemExit("A8 jobs queue behind the frozen Phase 2 runs; confirm they are COMPLETED.")

    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    pins = a8_design.check_own_code(require=True)
    if args.step not in design["commands"]:
        raise SystemExit(f"Unknown step {args.step}; steps are {sorted(design['commands'])}.")
    base = design["commands"][args.step]
    if base["uses_model"] or args.step in ("a1_merge", "natural_rq1"):
        if not args.a8_id:
            raise SystemExit(f"{args.step} needs --a8_id.")
        entry = a8_design.model(design, args.a8_id)
        keys = [entry["code_key"]]
    else:
        entry = None
        keys = [design["models"][m]["code_key"] for m in a8_design.active_models(design)]
    ctx = {"a8_id": args.a8_id, "key": entry["code_key"] if entry else None, "rotation": args.rotation,
           "condition": args.condition, "shard": args.shard, "n_shards": args.n_shards}
    spec = command(design, args.step, ctx)
    if args.step == "a1_merge" and args.n_shards is None:
        raise SystemExit("a1_merge needs --n_shards.")
    if any(v is None for k, v in ctx.items() if "{" + k + "}" in json.dumps(design["commands"][args.step])):
        raise SystemExit(f"{args.step} needs {[k for k in ctx if '{' + k + '}' in json.dumps(design['commands'][args.step])]}.")

    record: Dict[str, Any] = {"step": args.step, "a8_id": args.a8_id, "context": ctx, "command": spec,
                              "design_sha256": a8_design.block_hash(), "a8_code_pins": pins,
                              "launcher_sha256": a8_design.file_sha256(Path(__file__).resolve()),
                              "started_at_utc": datetime.now(timezone.utc).isoformat(), "test_mode": False}
    if entry is not None and base["uses_model"]:
        record["models_dir_env"] = shard.set_models_dir(Path(os.path.expanduser(entry["models_dir"])))
        weights = pre.check_weights_dir(Path(record["models_dir_env"]), entry)
        record["config"] = pre.check_config(weights, entry["expected_config"])
        record["weight_files"] = pre.check_files(weights, entry["weight_files"])
        record["loader"] = check_loader_path(entry["code_key"], weights)
    record["data_sha256"] = a8_design.check_data(design, [fill(n, ctx) for n in design["data_by_step"].get(args.step, [])])
    if args.step in ("a1_shard", "a2_run", "a4_run"):
        record["eligibility"] = a8_family.eligibility(design, ROOT, args.a8_id, args.step)
    if args.step == "a4_analyse":
        # The single A4 analysis runs only on the shard files A8's pre-run check passed (dated note 2026-11-13).
        record["prerun"] = a8_family.check_prerun_record(ROOT / a8_family.A4_PRERUN, a8_family.a4_cells_files(ROOT), ROOT)
    record["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    usage = pre.parse_quota(pre.subprocess.run(["quota"], capture_output=True, text=True, check=False).stdout)
    pre.check_storage(usage, design["storage_stop"])
    record["rds_usage"] = usage
    record["versions"] = shard.versions()
    record["visible_gpus"] = shard.visible_gpus()
    record["threads"] = check_threads(args.step)
    if args.step in design["software_versions"]["enforced_steps"]:
        shard.check_versions(record["versions"], design["software_versions"]["pinned"])
    if base["decoder_preflight"]:
        shard.run_decoder_preflight()

    outputs = [Path(o) for o in spec["outputs"]]
    pre.check_outputs(outputs, design["output_roots"], allow_existing=base["resumable"])
    declared = spec.get("declared_output_outside_roots")
    if declared and (ROOT / declared).exists():
        raise SystemExit(f"{declared} exists; the frozen script would overwrite it.")
    guard = frozen_output_files(design["frozen_outputs_guard"], keys)
    before = snapshot(guard)
    record["frozen_outputs_hashed"] = len(before)

    folder = ROOT / "research/reproduction/a8" / (args.a8_id or "all") / "launch"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    tag = "_".join(str(ctx[k]) for k in ("rotation", "condition", "shard") if ctx[k] is not None)
    record_path = folder / f"{args.step}{'_' + tag if tag else ''}_{stamp}.json"
    # Written now, so a job killed by Slurm still leaves an attempt on record (rule not_tested_after_retry).
    record["status"] = "started"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    finished = False
    try:
        if args.step == "a1_shard":
            ns = argparse.Namespace(cmd="run", a8_id=args.a8_id, equivalence_label=None, equivalence_models_dir=None,
                                    rotation=args.rotation, condition=args.condition, max_trials=None,
                                    human_options=None, output=ROOT / spec["outputs"][0], test_mode=False,
                                    shard=args.shard, n_shards=args.n_shards)
            record["shard_report"] = {k: v for k, v in shard.cmd_run(ns).items() if k != "shard_trial_ids"}
        elif args.step == "a1_merge":
            shard.merge(a1_shard_files(design, ctx), ROOT / spec["outputs"][0])
        else:
            run_frozen_step(spec, declared)
        finished = True
    except BaseException as exc:
        # Kept in the record, so a crash's cause (an out-of-memory error above all) is read from the record.
        record["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        raise
    finally:
        moved = changed(before, frozen_output_files(design["frozen_outputs_guard"], keys))
        record["frozen_outputs_changed"] = moved
        record["oom_files"] = oom_in_outputs([ROOT / o for o in spec["outputs"]], args.step)
        status = final_status(finished, moved, record["oom_files"], record.get("error"))
        record["status"] = status
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if status != "ok":
        raise SystemExit(f"A8 job ended with status {status}.")
    return record


if __name__ == "__main__":
    main()
