#!/usr/bin/env python
"""A8: run one A1 rotation task in shards through the frozen scripts/evaluate.py, then merge.

Why: evaluate.py cannot resume, and a 32B task projects past the 12 h wall limit. Splitting
the manifest would change the design, because evaluate.py computes mismatch partners over
the trials it loads and reads a partner's video from that same set.

How: every shard runs the frozen `evaluate.main()` on the full manifest, with the frozen
arguments of slurm_jobs/study3_rotation_eval.sh. So the frozen `vision_mismatch_partners`
builds the partner map over all items, exactly as in an unsharded run. Only the scoring loop
is restricted: `scripts.evaluate` gets a module-level `enumerate` that, applied to the loaded
trial list (recognised by its exact trial-id sequence), yields the shard's (index, trial)
pairs with their original indices, and is the built-in everywhere else. Draws are seeded per
trial (`seed_generation(seed, trial_id, k)`) and the option set uses the original index, so a
trial's record does not depend on its shard. No frozen file is edited.

The runner also computes the full-set partner map with the frozen function and checks each
mismatch record's `vision_source_trial_id` against it. The merge refuses a gap, a duplicate
or shards from different runs.

    run      one shard; writes the frozen eval JSON and a sidecar next to it
    merge    all shards of one rotation and condition into one file, in manifest order
    compare  two eval files, trial for trial (the A1 equivalence test)
    frozen   the unsharded frozen evaluate.main() with the same arguments (equivalence test)

The model weights come from MR_EU_HPC_MODELS_DIR, set from --models_dir before evaluate
is imported, as the frozen slurm script sets it.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402

ROOT = a8_design.ROOT
EVALUATE = ROOT / "scripts" / "evaluate.py"
SIDECAR_NAME = "a8_shard.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_pin(name: str) -> str:
    """The design block's pin for a frozen file, read without the hash gate for tests."""
    return json.loads(a8_design.block_bytes())["frozen_code_sha256"][name]


def check_evaluate(path: Path = EVALUATE) -> str:
    digest = sha256_file(path)
    pinned = frozen_pin("scripts/evaluate.py")
    if digest != pinned:
        raise SystemExit(f"{path} sha256 {digest} is not the frozen {pinned}.")
    return digest


def import_evaluate(models_dir: Optional[Path]):
    """Import the frozen module after its hash check. MR_EU_HPC_MODELS_DIR must be set first."""
    if models_dir is not None:
        os.environ["MR_EU_HPC_MODELS_DIR"] = str(Path(os.path.expanduser(str(models_dir))))
    check_evaluate()
    for p in (str(ROOT / "src"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import scripts.evaluate as ev  # noqa: E402

    return ev


def evaluate_argv(
    key: str,
    sampling: str,
    rotation: int,
    condition: str,
    output: Path,
    a1: Mapping[str, Any],
    max_trials: Optional[int] = None,
    human_options: Optional[str] = None,
) -> List[str]:
    """The argument list of slurm_jobs/study3_rotation_eval.sh, in its order."""
    e = a1["evaluate_args"]
    argv = [
        "--model", key,
        "--dataset", e["dataset"],
        "--condition", e["condition"],
        "--stage", e["stage"],
        "--max_frames", str(e["max_frames"]),
        "--fps", str(e["fps"]),
        "--seed", str(e["seed"]),
        "--n_options", str(e["n_options"]),
        "--data_root", e["data_root"],
        "--manifest", e["manifest"],
        "--human_options", human_options or e["human_options"].format(r=rotation),
        "--fc_samples", str(e["fc_samples"]),
    ]
    if sampling == "sequential":
        argv.append("--fc_sequential")
    elif sampling != "batched":
        raise SystemExit(f"Unknown sampling mode {sampling}.")
    if condition == "mismatch":
        argv += ["--vision_mismatch_seed", str(e["vision_mismatch_seed"])]
    elif condition != "real":
        raise SystemExit(f"Unknown condition {condition}.")
    if max_trials is not None:
        argv += ["--max_trials", str(max_trials)]
    argv += ["--output", str(output)]
    return argv


def shard_indices(n_items: int, shard: int, n_shards: int) -> List[int]:
    if n_shards < 1 or not 0 <= shard < n_shards:
        raise SystemExit(f"Shard {shard} of {n_shards} is not valid.")
    return [i for i in range(n_items) if i % n_shards == shard]


class ShardEnumerate:
    """`enumerate` for scripts.evaluate: restricts only the loop over the loaded trials."""

    def __init__(self, ids: Sequence[str], keep: Iterable[int]):
        self.ids = [str(i) for i in ids]
        self.keep = set(keep)
        self.matches = 0

    def _is_trial_list(self, iterable: Any) -> bool:
        return (
            isinstance(iterable, list)
            and len(iterable) == len(self.ids)
            and all(isinstance(t, dict) for t in iterable)
            and [str(t.get("trial_id")) for t in iterable] == self.ids
        )

    def __call__(self, iterable: Any, start: int = 0):
        if start == 0 and self._is_trial_list(iterable):
            self.matches += 1
            keep = self.keep
            return ((i, t) for i, t in builtins.enumerate(iterable) if i in keep)
        return builtins.enumerate(iterable, start)


def loaded_trials(ev, a1: Mapping[str, Any], max_trials: Optional[int]) -> List[Dict[str, Any]]:
    """The trial list evaluate.run_evaluation iterates, built by the frozen loader."""
    e = a1["evaluate_args"]
    root = ev.resolve_dataset_root(e["dataset"], override_root=Path(e["data_root"]))
    trials, _ = ev.load_trials_from_manifest(Path(e["manifest"]), root)
    return trials[: int(max_trials)] if max_trials is not None else trials


def partner_map(trials: Sequence[Mapping[str, Any]], human_options_path: Path, seed: int) -> Dict[str, str]:
    """The frozen map, built as run_evaluation builds it, over the given trials."""
    from paper_v2.option_score import vision_mismatch_partners
    from paper_v2.schema import actor_id_from_eu_filename

    human = json.loads(Path(human_options_path).read_text(encoding="utf-8")).get("trials", {}) if Path(human_options_path).exists() else {}
    by_id = {str(t.get("trial_id")): t for t in trials}
    ids = list(by_id)
    labels = [(human.get(i) or {}).get("human_target_label") or by_id[i]["label"] for i in ids]
    return vision_mismatch_partners(ids, labels, [actor_id_from_eu_filename(i) for i in ids], seed=int(seed))


def map_sha256(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest()


def run_frozen_main(ev, argv: Sequence[str], enumerate_fn: Optional[ShardEnumerate] = None) -> None:
    """evaluate.main() with this argv, with the shard enumerate installed only for the call."""
    saved = sys.argv
    sys.argv = [str(EVALUATE), *argv]
    try:
        if enumerate_fn is not None:
            ev.enumerate = enumerate_fn
        ev.main()
    finally:
        sys.argv = saved
        if enumerate_fn is not None and "enumerate" in vars(ev):
            del ev.enumerate


def check_shard_output(obj: Mapping[str, Any], expected_ids: Sequence[str], condition: str, partners: Mapping[str, str]) -> Dict[str, Any]:
    trials = obj.get("trials") or []
    got = [str(t.get("trial_id")) for t in trials]
    if got != list(expected_ids):
        raise SystemExit(f"Shard output holds {len(got)} trials that are not this shard's {len(expected_ids)}, in order.")
    if condition == "mismatch":
        wrong = [t["trial_id"] for t in trials if t.get("vision_source_trial_id") != partners[str(t["trial_id"])]]
        if wrong:
            raise SystemExit(f"{len(wrong)} mismatch trials did not use the full-set partner: {wrong[:3]}")
    errored = sorted(str(t["trial_id"]) for t in trials if t.get("error"))
    return {"n_trials": len(trials), "errored_trial_ids": errored}


def cmd_run(args: argparse.Namespace, ev=None) -> Dict[str, Any]:
    design = json.loads(a8_design.block_bytes()) if args.no_design_gate else a8_design.load_design()
    if not args.no_design_gate:
        a8_design.check_frozen_code(design)
    a1 = design["a1"]
    ev = ev or import_evaluate(args.models_dir)
    trials = loaded_trials(ev, a1, args.max_trials)
    ids = [str(t.get("trial_id")) for t in trials]
    keep = shard_indices(len(ids), args.shard, args.n_shards)
    human_path = Path(args.human_options or a1["evaluate_args"]["human_options"].format(r=args.rotation))
    partners = partner_map(trials, human_path, a1["evaluate_args"]["vision_mismatch_seed"]) if args.condition == "mismatch" else {}
    output = Path(args.output)
    sidecar = output.parent / SIDECAR_NAME
    for path in (output, sidecar):
        if path.exists():
            raise SystemExit(f"{path} exists; a shard is never overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = evaluate_argv(args.key, args.sampling, args.rotation, args.condition, output, a1, args.max_trials, args.human_options)
    filt = ShardEnumerate(ids, keep)
    started = datetime.now(timezone.utc).isoformat()
    run_frozen_main(ev, argv, filt)
    if filt.matches != 1:
        raise SystemExit(f"The trial loop was matched {filt.matches} times, not once.")
    obj = json.loads(output.read_text(encoding="utf-8"))
    checked = check_shard_output(obj, [ids[i] for i in keep], args.condition, partners)
    report = {
        "label": args.label,
        "key": args.key,
        "sampling": args.sampling,
        "rotation": args.rotation,
        "condition": args.condition,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "max_trials": args.max_trials,
        "n_loaded_trials": len(ids),
        "loaded_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "shard_trial_ids": [ids[i] for i in keep],
        "partner_map_sha256": map_sha256(partners) if partners else None,
        "evaluate_argv": argv,
        "evaluate_sha256": sha256_file(EVALUATE),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "design_sha256": a8_design.block_hash(),
        "eval_output_sha256": sha256_file(output),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **checked,
    }
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def cmd_frozen(args: argparse.Namespace, ev=None) -> None:
    design = json.loads(a8_design.block_bytes()) if args.no_design_gate else a8_design.load_design()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists.")
    ev = ev or import_evaluate(args.models_dir)
    argv = evaluate_argv(args.key, args.sampling, args.rotation, args.condition, output, design["a1"], args.max_trials, args.human_options)
    run_frozen_main(ev, argv)


SAME_ACROSS_SHARDS = ("label", "key", "sampling", "rotation", "condition", "n_shards", "max_trials",
                      "n_loaded_trials", "loaded_ids_sha256", "partner_map_sha256", "evaluate_sha256",
                      "runner_sha256", "design_sha256")


def merge(shard_files: Sequence[Path], output: Path) -> Dict[str, Any]:
    """One file from every shard of one run: trials in manifest order, exact partition."""
    if output.exists():
        raise SystemExit(f"{output} exists.")
    loaded = []
    for path in shard_files:
        path = Path(path)
        side = json.loads((path.parent / SIDECAR_NAME).read_text(encoding="utf-8"))
        if sha256_file(path) != side["eval_output_sha256"]:
            raise SystemExit(f"{path} changed after its shard finished.")
        loaded.append((side, json.loads(path.read_text(encoding="utf-8"))))
    if not loaded:
        raise SystemExit("No shards.")
    first = loaded[0][0]
    for side, _ in loaded:
        diff = [k for k in SAME_ACROSS_SHARDS if side.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"Shards come from different runs: {diff}")
    shards = sorted(side["shard"] for side, _ in loaded)
    if shards != list(range(first["n_shards"])):
        raise SystemExit(f"Shards {shards} are not one complete set of {first['n_shards']}.")
    by_id: Dict[str, Dict[str, Any]] = {}
    order: Dict[str, int] = {}
    for side, obj in loaded:
        for trial in obj["trials"]:
            tid = str(trial["trial_id"])
            if tid in by_id:
                raise SystemExit(f"Trial {tid} is in two shards.")
            by_id[tid] = trial
    for side, _ in loaded:
        for offset, tid in enumerate(side["shard_trial_ids"]):
            order[tid] = side["shard"] + offset * first["n_shards"]
    if len(by_id) != first["n_loaded_trials"] or set(by_id) != set(order):
        raise SystemExit(f"Merged {len(by_id)} trials, expected {first['n_loaded_trials']}.")
    trials = [by_id[t] for t in sorted(order, key=order.get)]
    merged = {
        "a8_merged": {
            "n_trials": len(trials),
            "n_shards": first["n_shards"],
            "shard_sidecars": [side for side, _ in sorted(loaded, key=lambda x: x[0]["shard"])],
            "merger_sha256": sha256_file(Path(__file__).resolve()),
            "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "shard_headers": [
            {k: v for k, v in obj.items() if k != "trials"} for _, obj in sorted(loaded, key=lambda x: x[0]["shard"])
        ],
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return merged


def compare(frozen_path: Path, merged_path: Path) -> Dict[str, Any]:
    """Trial for trial: same ids in the same order, and every record equal."""
    a = json.loads(Path(frozen_path).read_text(encoding="utf-8"))["trials"]
    b = json.loads(Path(merged_path).read_text(encoding="utf-8"))["trials"]
    ids_a, ids_b = [str(t["trial_id"]) for t in a], [str(t["trial_id"]) for t in b]
    differing = [ta["trial_id"] for ta, tb in zip(a, b) if ta != tb]
    report = {
        "frozen": str(frozen_path),
        "merged": str(merged_path),
        "n_frozen": len(a),
        "n_merged": len(b),
        "same_ids_in_order": ids_a == ids_b,
        "n_differing_records": len(differing),
        "differing_trial_ids": differing,
    }
    report["identical"] = report["same_ids_in_order"] and not differing
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "frozen"):
        p = sub.add_parser(name)
        p.add_argument("--key", required=True, choices=("qwen3vl", "gemma4"))
        p.add_argument("--sampling", required=True, choices=("sequential", "batched"))
        p.add_argument("--rotation", type=int, required=True, choices=range(5))
        p.add_argument("--condition", required=True, choices=("real", "mismatch"))
        p.add_argument("--models_dir", type=Path, help="Sets MR_EU_HPC_MODELS_DIR before evaluate is imported.")
        p.add_argument("--max_trials", type=int)
        p.add_argument("--human_options", help="Override the rotation lookup path (tests only).")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--no_design_gate", action="store_true", help="Before the freeze, and in tests.")
        if name == "run":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--n_shards", type=int, required=True)
            p.add_argument("--label", required=True, help="A8 id, or a name for the equivalence test.")
    m = sub.add_parser("merge")
    m.add_argument("--shards", type=Path, nargs="+", required=True)
    m.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--frozen", type=Path, required=True)
    c.add_argument("--merged", type=Path, required=True)
    c.add_argument("--report", type=Path)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "frozen":
        return cmd_frozen(args)
    if args.cmd == "merge":
        merged = merge(args.shards, args.output)
        print(json.dumps({"n_trials": merged["a8_merged"]["n_trials"], "output": str(args.output)}))
        return merged
    report = compare(args.frozen, args.merged)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["identical"]:
        raise SystemExit("A1 equivalence test failed.")
    return report


if __name__ == "__main__":
    main()
