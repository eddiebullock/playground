#!/usr/bin/env python
"""A8: the frozen A2 analysis on one A8 model, with one input loader replaced.

`scripts/analyse_stimulus_interventions.py` reads RQ0, gates and delivery by model key from
`--reproduction`, which A8 fixes to `research/reproduction/a8/<a8_id>/`. It reads A3 only
from a file whose sha256 is the frozen three-model summary's, so an A8 model would always
read as "no A3 result", and every verdict would be Inconclusive by construction. This
wrapper replaces `a3_summary`, and nothing else, with a loader of the A8 A3 summary,
presented under the model's code key. The A8 A3 and A1 summaries are accepted only as their
A8 producers wrote them (provenance files: output and input hashes). Before the run it
writes, in the A8 reproduction folder, `rotation_summary.json` as this model's A8 A1 entry
under its code key, the form the frozen code reads. Then the frozen `main` runs.

Equivalence: through `main`, a frozen A3 summary with one entry renamed to an A8 id gives
output byte-identical to the frozen analysis (tests/test_a8.py).

    python research/scripts/a8_analyse_a2.py --a8_id qwen3vl_32b -- \
        --output results/a8/qwen3vl_32b/a2_summary.json --a2_frozen

`--model`, `--reproduction`, `--cells` and `--run_report` are set by the wrapper: the cells
and run reports come only from results/a8/<a8_id>/stimulus_interventions/, each shard backed
by a launcher record with status ok and the design's weights path, as are the gate and
delivery files it reads. Any pass-through argument that names or abbreviates one of these is
refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402
import a8_family  # noqa: E402

ROOT = a8_design.ROOT
FROZEN = ("scripts/analyse_stimulus_interventions.py", "scripts/stimulus_design.py")


def load_frozen():
    pins = json.loads(a8_design.block_bytes())["frozen_code_sha256"]
    for name in FROZEN:
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != pins[name]:
            raise SystemExit(f"{name} sha256 {digest} is not the frozen {pins[name]}.")
    for p in (str(ROOT / "src"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from scripts import analyse_stimulus_interventions as an

    return an


def a3_loader(path: Path, sha256: str, view: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Replacement for `a3_summary`: the given file, only at the given bytes, through `view`."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path} sha256 {digest} is not the given {sha256}.")
    parsed = json.loads(data)

    def a3_summary(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return view(parsed)

    return a3_summary


def run_frozen_main(an, argv: Sequence[str], a3_summary: Optional[Callable[..., Dict[str, Any]]]) -> None:
    original = an.a3_summary
    saved = sys.argv
    sys.argv = ["analyse_stimulus_interventions.py", *argv]
    try:
        if a3_summary is not None:
            an.a3_summary = a3_summary
        an.main()
    finally:
        sys.argv = saved
        an.a3_summary = original


def write_a1_view(a1_summary: Path, a8_id: str, key: str, folder: Path) -> Path:
    """rotation_summary.json in the A8 folder: this model's A8 A1 entry under its key.

    A model outside the A1 family has no entry; the view then holds nothing for the key,
    which the frozen code reads as no A1 result, as the frozen rule requires.
    """
    entry = json.loads(Path(a1_summary).read_text(encoding="utf-8")).get(a8_id)
    data = (json.dumps({key: entry} if entry is not None else {}, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = folder / "rotation_summary.json"
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f"{path} exists with other content.")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


OWN = ("--model", "--reproduction", "--cells", "--run_report")


def check_pass_through(rest: Sequence[str]) -> None:
    """Refuse any argument the wrapper sets, or an abbreviation the frozen parser would expand."""
    for token in rest:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if any(opt.startswith(name) for opt in OWN):
            raise SystemExit(f"{name} is set by the wrapper from the design; it cannot be passed through.")
        if "--output".startswith(name) and name != "--output":
            raise SystemExit(f"{name} is an abbreviation; write --output in full.")
    if sum(1 for tok in rest if tok.split("=", 1)[0] == "--output") != 1:
        raise SystemExit("Give --output exactly once.")


def shard_inputs(design: Mapping[str, Any], base: Path, a8_id: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Cells and run reports from the fixed A8 folder, each shard backed by an ok launcher record."""
    entry = design["models"][a8_id]
    key, weights = entry["code_key"], entry["weights_dir"]
    records = [r for r in a8_family.launch_records(base, a8_id, "a2_run") if r.get("status") == "ok"]
    if not records:
        raise SystemExit(f"{a8_id}: no a2_run job finished ok.")
    counts = {r["context"]["n_shards"] for r in records}
    if len(counts) != 1:
        raise SystemExit(f"{a8_id}: a2_run records use different shard counts {sorted(counts)}.")
    n = counts.pop()
    folder = base / "results/a8" / a8_id / "stimulus_interventions"
    cells, reports, used = [], [], {}
    for k in range(n):
        used[k] = a8_family.ok_record(base, a8_id, "a2_run", weights, {"shard": k, "n_shards": n})["started_at_utc"]
        suffix = f"_shard{k}" if n > 1 else ""
        cells.append(str(folder / f"cells_{key}{suffix}.jsonl"))
        reports.append(str(folder / f"run_{key}{suffix}.json"))
    for step in ("gates", "delivery"):
        used[step] = a8_family.ok_record(base, a8_id, step, weights)["started_at_utc"]
    return cells, reports, used


def run(a8_id: str, rest: Sequence[str], base: Path = ROOT) -> None:
    """Design binding, input checks, A1 view, then the frozen main with fixed inputs."""
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    entry = a8_design.model(design, a8_id)
    key = entry["code_key"]
    check_pass_through(rest)
    folder = (base / "research/reproduction/a8" / a8_id).resolve()
    out = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    out.add_argument("--output", type=Path, required=True)
    known, _ = out.parse_known_args(list(rest))
    target = known.output if known.output.is_absolute() else base / known.output
    roots = [(base / r).resolve() for r in design["output_roots"]]
    if not any(r in target.resolve().parents for r in roots):
        raise SystemExit(f"{known.output} is outside the A8 output roots.")
    if target.exists():
        raise SystemExit(f"{known.output} exists.")
    a3_path, a1_path = base / a8_family.A3_SUMMARY, base / a8_family.A1_SUMMARY
    a8_family.check_provenance(a3_path)
    a8_family.check_provenance(a1_path)
    cells, reports, _ = shard_inputs(design, base, a8_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    view = write_a1_view(a1_path, a8_id, key, folder)
    an = load_frozen()
    sha = hashlib.sha256(a3_path.read_bytes()).hexdigest()
    loader = a3_loader(a3_path, sha, lambda s: {key: s[a8_id]} if a8_id in s else {})
    argv = ["--model", key, "--reproduction", str(folder), "--cells", *cells, "--run_report", *reports, *rest]
    run_frozen_main(an, argv, loader)
    if not target.exists():
        raise SystemExit(f"The frozen analysis wrote no {known.output}.")
    # Provenance (PREREG_A8_DRAFT.md, "Analyses outside the launcher"): output and input hashes, checked by
    # the next consumer and after the pull.
    inputs = [Path(p) for p in (*cells, *reports)] + [a3_path, a1_path, view, folder / f"vision_gates_{key}.json",
                                                       folder / f"input_delivery_{key}.json"]
    a8_family.write_provenance(target, [p for p in inputs if p.exists()], {
        "a8_id": a8_id, "key": key, "frozen_argv": [str(a) for a in argv],
        "wrapper_sha256": a8_design.file_sha256(Path(__file__).resolve()),
        "design_sha256": a8_design.block_hash(),
    })


def main(argv: Optional[Sequence[str]] = None) -> None:
    os.chdir(ROOT)
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("Separate the frozen analysis arguments with --.")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--a8_id", required=True)
    args = ap.parse_args(argv[:split])
    a8_design.check_own_code(require=True)
    run(args.a8_id, argv[split + 1:])


if __name__ == "__main__":
    main()
