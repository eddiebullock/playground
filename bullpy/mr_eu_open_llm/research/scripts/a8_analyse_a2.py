#!/usr/bin/env python
"""A8: the frozen A2 analysis on one A8 model, with one input loader replaced.

`scripts/analyse_stimulus_interventions.py` reads RQ0, gates and delivery by model key from
`--reproduction`, which A8 points at `research/reproduction/a8/<a8_id>/`. It reads A3 only
from a file whose sha256 is the frozen three-model summary's, so an A8 model would always
read as "no A3 result", and every verdict would be Inconclusive by construction. This
wrapper replaces `a3_summary`, and nothing else, with a loader of the A8 A3 summary, whose
sha256 must equal the value given on the command line, presented under the model's code
key. Then the frozen `main` runs with the frozen CLI.

Before that it writes, in the A8 reproduction folder, `rotation_summary.json` as the A8 A1
summary's entry for this model under its code key, the form the frozen code reads.

Equivalence: with the frozen A3 file given as the A8 A3 file, the output equals the frozen
analysis's output byte for byte (tests/test_a8.py).

    python research/scripts/a8_analyse_a2.py --a8_id qwen3vl_32b \
        --a8_a3_summary research/reproduction/a8/face_input_validity_summary_a8.json \
        --a8_a3_sha256 <hex> --a8_a1_summary research/reproduction/a8/rotation_summary_a8.json \
        -- --model qwen3vl --cells ... --run_report ... \
        --reproduction research/reproduction/a8/qwen3vl_32b --output ... --a2_frozen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402

ROOT = a8_design.ROOT
FROZEN = {
    "scripts/analyse_stimulus_interventions.py": ROOT / "scripts/analyse_stimulus_interventions.py",
    "scripts/stimulus_design.py": ROOT / "scripts/stimulus_design.py",
}


def load_frozen():
    pins = json.loads(a8_design.block_bytes())["frozen_code_sha256"]
    for name, path in FROZEN.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
    """The frozen main with this argv; `a3_summary` replaced for the call when given."""
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
    """rotation_summary.json in the A8 folder: this model's A8 A1 entry under its key."""
    entry = json.loads(Path(a1_summary).read_text(encoding="utf-8")).get(a8_id)
    if entry is None:
        raise SystemExit(f"{a1_summary} has no entry for {a8_id}.")
    data = (json.dumps({key: entry}, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = folder / "rotation_summary.json"
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f"{path} exists with other content.")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("Separate the frozen analysis arguments with --.")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a8_id", required=True)
    ap.add_argument("--a8_a3_summary", type=Path, required=True)
    ap.add_argument("--a8_a3_sha256", required=True)
    ap.add_argument("--a8_a1_summary", type=Path, required=True)
    args = ap.parse_args(argv[:split])
    rest = argv[split + 1:]
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    entry = a8_design.model(design, args.a8_id)
    key = entry["code_key"]
    frozen_args = argparse.ArgumentParser(add_help=False)
    frozen_args.add_argument("--model", required=True)
    frozen_args.add_argument("--reproduction", type=Path, required=True)
    frozen_args.add_argument("--output", type=Path, required=True)
    known, _ = frozen_args.parse_known_args(rest)
    folder = (ROOT / "research/reproduction/a8" / args.a8_id).resolve()
    if known.model != key:
        raise SystemExit(f"--model must be {key} for {args.a8_id}.")
    if (ROOT / known.reproduction).resolve() != folder:
        raise SystemExit(f"--reproduction must be {folder}.")
    if known.output.exists():
        raise SystemExit(f"{known.output} exists.")
    write_a1_view(args.a8_a1_summary, args.a8_id, key, folder)
    an = load_frozen()
    loader = a3_loader(args.a8_a3_summary, args.a8_a3_sha256, lambda s: {key: s[args.a8_id]} if args.a8_id in s else {})
    run_frozen_main(an, rest, loader)


if __name__ == "__main__":
    main()
