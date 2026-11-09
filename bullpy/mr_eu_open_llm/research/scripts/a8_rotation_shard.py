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
trial and the option set uses the original index, so a trial's record does not depend on its
shard. No frozen file is edited.

Bound to the design: `--a8_id` fixes the code key, sampling mode, models directory, weights
directory and output path. MR_EU_HPC_MODELS_DIR is set from the design before evaluate is
imported (a different preset value is refused), and the path the frozen loader will use is
asserted to be the design's weights directory and recorded. Production runs need the A8
code pins, the pinned data files, cv2 4.11 and the frozen decoder preflight, and the pinned
torch, transformers and cv2 versions.

`--equivalence_label` runs the 8B equivalence test instead (outputs under
results/a8/equivalence only). `--test_mode` is for local tests: it skips the environment
checks, is recorded, and a test-mode shard can never be merged into results/a8/<a8_id>.

    run      one shard; writes the frozen eval JSON and `<stem>.a8_shard.json` beside it
    merge    all shards of one rotation and condition into one file, in manifest order
    compare  two eval files, trial for trial (the A1 equivalence test)
    frozen   the unsharded frozen evaluate.main() with the same arguments (equivalence only)
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402

ROOT = a8_design.ROOT
EVALUATE = ROOT / "scripts" / "evaluate.py"
SIDECAR_SUFFIX = ".a8_shard.json"
EQUIVALENCE_ROOT = ROOT / "results" / "a8" / "equivalence"
OOM_MARKERS = ("out of memory", "outofmemoryerror")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design() -> Dict[str, Any]:
    return a8_design.load_design()


def sidecar_path(output: Path) -> Path:
    return Path(output).with_suffix(SIDECAR_SUFFIX)


def check_evaluate(path: Path = EVALUATE) -> str:
    digest = sha256_file(path)
    pinned = design()["frozen_code_sha256"]["scripts/evaluate.py"]
    if digest != pinned:
        raise SystemExit(f"{path} sha256 {digest} is not the frozen {pinned}.")
    return digest


def set_models_dir(models_dir: Path) -> str:
    """Set MR_EU_HPC_MODELS_DIR before config is imported. A different preset value is refused."""
    want = str(Path(os.path.expanduser(str(models_dir))).resolve())
    have = os.environ.get("MR_EU_HPC_MODELS_DIR")
    if have is not None and str(Path(os.path.expanduser(have)).resolve()) != want:
        raise SystemExit(f"MR_EU_HPC_MODELS_DIR is {have}, not the design's {want}.")
    if "config" in sys.modules and str(Path(str(sys.modules["config"].HPC_MODELS_DIR)).resolve()) != want:
        raise SystemExit("config was imported with another models directory.")
    os.environ["MR_EU_HPC_MODELS_DIR"] = want
    return want


def import_evaluate():
    check_evaluate()
    for p in (str(ROOT / "src"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import scripts.evaluate as ev  # noqa: E402

    return ev


def versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "cv2", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def check_versions(seen: Mapping[str, Optional[str]], pinned: Mapping[str, str]) -> None:
    wrong = {k: seen.get(k) for k, v in pinned.items() if seen.get(k) != v}
    if wrong:
        raise SystemExit(f"Software versions differ from the pinned {dict(pinned)}: {wrong}")


def run_decoder_preflight() -> None:
    """The frozen decoder check, as every frozen video job runs it first."""
    done = subprocess.run([sys.executable, str(ROOT / "research/scripts/decoder_preflight.py")], cwd=ROOT)
    if done.returncode != 0:
        raise SystemExit("decoder_preflight.py failed; A1.1 requires cv2 4.11 and a stable decode.")


def visible_gpus() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def evaluate_argv(key: str, sampling: str, rotation: int, condition: str, output: Path,
                  a1: Mapping[str, Any], max_trials: Optional[int] = None,
                  human_options: Optional[str] = None) -> List[str]:
    """The argument list of slurm_jobs/study3_rotation_eval.sh, in its order."""
    e = a1["evaluate_args"]
    argv = [
        "--model", key, "--dataset", e["dataset"], "--condition", e["condition"], "--stage", e["stage"],
        "--max_frames", str(e["max_frames"]), "--fps", str(e["fps"]), "--seed", str(e["seed"]),
        "--n_options", str(e["n_options"]), "--data_root", e["data_root"], "--manifest", e["manifest"],
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


def argv_without_output(argv: Sequence[str]) -> List[str]:
    argv = list(argv)
    i = argv.index("--output")
    return argv[:i] + argv[i + 2:]


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
    """The frozen map, built as run_evaluation builds it. A missing lookup is refused."""
    from paper_v2.option_score import vision_mismatch_partners
    from paper_v2.schema import actor_id_from_eu_filename

    path = Path(human_options_path)
    if not path.exists():
        raise SystemExit(f"{path} is missing; A8 never falls back to sampled foils or manifest labels.")
    human = json.loads(path.read_text(encoding="utf-8")).get("trials", {})
    by_id = {str(t.get("trial_id")): t for t in trials}
    ids = list(by_id)
    labels = [(human.get(i) or {}).get("human_target_label") or by_id[i]["label"] for i in ids]
    return vision_mismatch_partners(ids, labels, [actor_id_from_eu_filename(i) for i in ids], seed=int(seed))


def map_sha256(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest()


def run_frozen_main(ev, argv: Sequence[str], enumerate_fn: Optional[ShardEnumerate] = None) -> None:
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


def is_oom(message: Optional[str]) -> bool:
    return bool(message) and any(m in str(message).lower() for m in OOM_MARKERS)


def check_shard_output(obj: Mapping[str, Any], expected_ids: Sequence[str], condition: str, partners: Mapping[str, str]) -> Dict[str, Any]:
    """Shard records in order; scored mismatch records used the full-set partner.

    An errored record is kept, as the frozen pipeline keeps it and scores it wrong. It must
    carry no partner field, because evaluate.py writes none on that branch.
    """
    trials = obj.get("trials") or []
    got = [str(t.get("trial_id")) for t in trials]
    if got != list(expected_ids):
        raise SystemExit(f"Shard output holds {len(got)} trials that are not this shard's {len(expected_ids)}, in order.")
    errored = [t for t in trials if t.get("error")]
    if condition == "mismatch":
        bad = [t["trial_id"] for t in errored if t.get("vision_source_trial_id") is not None]
        if bad:
            raise SystemExit(f"Errored records carry a partner field: {bad[:3]}")
        wrong = [t["trial_id"] for t in trials if not t.get("error") and t.get("vision_source_trial_id") != partners[str(t["trial_id"])]]
        if wrong:
            raise SystemExit(f"{len(wrong)} mismatch trials did not use the full-set partner: {wrong[:3]}")
    return {
        "n_trials": len(trials),
        "errored": {str(t["trial_id"]): str(t.get("error")) for t in errored},
        "n_errored": len(errored),
        "n_oom": sum(is_oom(t.get("error")) for t in errored),
    }


def resolve_run(args: argparse.Namespace, d: Mapping[str, Any]) -> Dict[str, Any]:
    """Key, sampling, models and weights directories, label and allowed output, from the design."""
    if args.a8_id:
        entry = a8_design.model(d, args.a8_id)
        template = d["commands"]["a1_shard"]["outputs"][0]
        expected = ROOT / template.format(a8_id=args.a8_id, rotation=args.rotation, condition=args.condition,
                                          shard=args.shard, n_shards=args.n_shards) if args.cmd == "run" else None
        return {"label": args.a8_id, "key": entry["code_key"], "sampling": entry["a1_sampling"],
                "models_dir": Path(os.path.expanduser(entry["models_dir"])),
                "weights_dir": Path(entry["weights_dir"]), "expected_output": expected, "mode": "production"}
    if not args.equivalence_models_dir:
        raise SystemExit("--equivalence_label needs --equivalence_models_dir.")
    models_dir = Path(os.path.expanduser(str(args.equivalence_models_dir)))
    return {"label": args.equivalence_label, "key": "qwen3vl", "sampling": "sequential", "models_dir": models_dir,
            "weights_dir": (models_dir / "qwen3vl"), "expected_output": None, "mode": "equivalence"}


def check_output_place(output: Path, run: Mapping[str, Any], test_mode: bool) -> None:
    out = Path(output).resolve()
    if run["mode"] == "production" and not test_mode:
        if out != Path(run["expected_output"]).resolve():
            raise SystemExit(f"Output must be {run['expected_output']}.")
    if run["mode"] == "equivalence" and not test_mode and EQUIVALENCE_ROOT.resolve() not in out.parents:
        raise SystemExit(f"Equivalence outputs go under {EQUIVALENCE_ROOT}.")
    if out.parent.exists() and any(out.parent.iterdir()):
        raise SystemExit(f"{out.parent} is not empty; each shard needs a fresh directory.")


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    """Every check before evaluate is imported. Returns the run description and environment."""
    os.chdir(ROOT)
    d = design()
    a8_design.check_frozen_code(d)
    run = resolve_run(args, d)
    check_output_place(args.output, run, args.test_mode)
    if args.human_options and not args.test_mode:
        raise SystemExit("--human_options is for test mode only.")
    env: Dict[str, Any] = {"test_mode": bool(args.test_mode), "versions": versions(), "visible_gpus": visible_gpus()}
    if not args.test_mode:
        if run["mode"] == "production":
            a8_design.check_own_code(require=True)
        lookup = d["a1"]["evaluate_args"]["human_options"].format(r=args.rotation)
        env["data_sha256"] = a8_design.check_data(d, [d["a1"]["evaluate_args"]["manifest"], "data/eu_emotions_human_entropy.json", lookup])
        check_versions(env["versions"], d["software_versions"]["pinned"])
        run_decoder_preflight()
    # In test mode config is already imported by the test; the resolved-path check still runs.
    env["models_dir_env"] = None if args.test_mode else set_models_dir(run["models_dir"])
    return {"design": d, "run": run, "env": env}


def check_resolved_model(ev, run: Mapping[str, Any]) -> str:
    """The path the frozen loader will use, which must be the design's weights directory."""
    resolved = Path(ev.resolve_model_path(run["key"])).resolve()
    if resolved != Path(run["weights_dir"]).resolve():
        raise SystemExit(f"The frozen loader would read {resolved}, not {run['weights_dir']}.")
    return str(resolved)


def cmd_run(args: argparse.Namespace, ev=None) -> Dict[str, Any]:
    ctx = prepare(args)
    d, run, env = ctx["design"], ctx["run"], ctx["env"]
    a1 = d["a1"]
    ev = ev or import_evaluate()
    resolved = check_resolved_model(ev, run)
    trials = loaded_trials(ev, a1, args.max_trials)
    ids = [str(t.get("trial_id")) for t in trials]
    keep = shard_indices(len(ids), args.shard, args.n_shards)
    human_path = Path(args.human_options or a1["evaluate_args"]["human_options"].format(r=args.rotation))
    if not human_path.exists():
        raise SystemExit(f"{human_path} is missing; A8 never falls back to sampled foils.")
    partners = partner_map(trials, human_path, a1["evaluate_args"]["vision_mismatch_seed"]) if args.condition == "mismatch" else {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = evaluate_argv(run["key"], run["sampling"], args.rotation, args.condition, output, a1, args.max_trials, args.human_options)
    filt = ShardEnumerate(ids, keep)
    started = datetime.now(timezone.utc).isoformat()
    run_frozen_main(ev, argv, filt)
    if filt.matches != 1:
        raise SystemExit(f"The trial loop was matched {filt.matches} times, not once.")
    obj = json.loads(output.read_text(encoding="utf-8"))
    checked = check_shard_output(obj, [ids[i] for i in keep], args.condition, partners)
    report = {
        "label": run["label"], "mode": run["mode"], "key": run["key"], "sampling": run["sampling"],
        "rotation": args.rotation, "condition": args.condition, "shard": args.shard, "n_shards": args.n_shards,
        "max_trials": args.max_trials, "n_loaded_trials": len(ids),
        "loaded_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "shard_trial_ids": [ids[i] for i in keep],
        "partner_map_sha256": map_sha256(partners) if partners else None,
        "evaluate_argv": argv, "evaluate_argv_without_output": argv_without_output(argv),
        "resolved_model_path": resolved, "weights_dir": str(run["weights_dir"]),
        "evaluate_sha256": sha256_file(EVALUATE), "runner_sha256": sha256_file(Path(__file__).resolve()),
        "design_sha256": a8_design.block_hash(), "eval_output_sha256": sha256_file(output),
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **env, **checked,
    }
    sidecar_path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if checked["n_oom"]:
        raise SystemExit(f"{checked['n_oom']} trials ran out of memory; by the A8 rule this shard fails and is rerun with more GPUs.")
    return report


def cmd_frozen(args: argparse.Namespace, ev=None) -> None:
    """Unsharded frozen run, for the equivalence test only."""
    if args.a8_id:
        raise SystemExit("frozen is for the equivalence test, not for A8 models.")
    args.shard, args.n_shards = 0, 1
    ctx = prepare(args)
    ev = ev or import_evaluate()
    check_resolved_model(ev, ctx["run"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = evaluate_argv("qwen3vl", "sequential", args.rotation, args.condition, output, ctx["design"]["a1"], args.max_trials, args.human_options)
    run_frozen_main(ev, argv)


SAME_ACROSS_SHARDS = ("label", "mode", "key", "sampling", "rotation", "condition", "n_shards", "max_trials",
                      "n_loaded_trials", "loaded_ids_sha256", "partner_map_sha256", "evaluate_sha256",
                      "runner_sha256", "design_sha256", "evaluate_argv_without_output_minus_shard",
                      "resolved_model_path", "visible_gpus", "test_mode", "versions")


def _comparable(side: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(side)
    out["evaluate_argv_without_output_minus_shard"] = side.get("evaluate_argv_without_output")
    return out


def merge(shard_files: Sequence[Path], output: Path) -> Dict[str, Any]:
    """One file from every shard of one run: trials in manifest order, exact partition."""
    output = Path(output)
    if output.exists():
        raise SystemExit(f"{output} exists.")
    loaded = []
    for path in shard_files:
        path = Path(path)
        side = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
        if sha256_file(path) != side["eval_output_sha256"]:
            raise SystemExit(f"{path} changed after its shard finished.")
        loaded.append((_comparable(side), json.loads(path.read_text(encoding="utf-8"))))
    if not loaded:
        raise SystemExit("No shards.")
    first = loaded[0][0]
    for side, _ in loaded:
        diff = [k for k in SAME_ACROSS_SHARDS if side.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"Shards come from different runs: {diff}")
    if any(side.get("n_oom") for side, _ in loaded):
        raise SystemExit("A shard ran out of memory; by the A8 rule it is rerun before any merge.")
    production = (ROOT / "results" / "a8").resolve()
    out = output.resolve()
    if first.get("test_mode") and production in out.parents and EQUIVALENCE_ROOT.resolve() not in out.parents:
        raise SystemExit("Test-mode shards are never merged into results/a8.")
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
    sidecars = [json.loads(sidecar_path(Path(p)).read_text(encoding="utf-8")) for p in shard_files]
    merged = {
        "a8_merged": {
            "n_trials": len(trials), "n_shards": first["n_shards"],
            "n_errored": sum(s["n_errored"] for s in sidecars),
            "shard_sidecars": sorted(sidecars, key=lambda s: s["shard"]),
            "merger_sha256": sha256_file(Path(__file__).resolve()),
            "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "shard_headers": [{k: v for k, v in obj.items() if k != "trials"} for _, obj in sorted(loaded, key=lambda x: x[0]["shard"])],
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
    report = {"frozen": str(frozen_path), "merged": str(merged_path), "n_frozen": len(a), "n_merged": len(b),
              "same_ids_in_order": ids_a == ids_b, "n_differing_records": len(differing),
              "differing_trial_ids": differing}
    report["identical"] = report["same_ids_in_order"] and not differing
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "frozen"):
        p = sub.add_parser(name)
        who = p.add_mutually_exclusive_group(required=True)
        who.add_argument("--a8_id")
        who.add_argument("--equivalence_label")
        p.add_argument("--equivalence_models_dir", type=Path)
        p.add_argument("--rotation", type=int, required=True, choices=range(5))
        p.add_argument("--condition", required=True, choices=("real", "mismatch"))
        p.add_argument("--max_trials", type=int)
        p.add_argument("--human_options", help="Test mode only.")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--test_mode", action="store_true", help="Local tests: skips environment checks; recorded.")
        if name == "run":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--n_shards", type=int, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--shards", type=Path, nargs="+", required=True)
    m.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--frozen", type=Path, required=True)
    c.add_argument("--merged", type=Path, required=True)
    c.add_argument("--report", type=Path)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Any:
    os.chdir(ROOT)
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
