#!/usr/bin/env python
"""A8: the frozen A3 summary over the A8 family.

The frozen `scripts/face_input_validity.py --summarise` reads the three frozen models'
cell files and applies Holm over them. This runs the same functions, `read_cells`,
`summarise_model` and `decide`, on a given list of (name, cells file) pairs, and writes the
output exactly as the frozen `summarise` does. Holm is then over the given models only.

Equivalence: given the three frozen cell files in the frozen order (qwen3vl, gemma4,
molmo2), `summary_bytes` reproduces the frozen summary byte for byte (sha256
c1c64e69..., `frozen_a3_summary_sha256` in the A8 design).

    python research/scripts/a8_face_input_validity_summary.py \
        --cells qwen3vl_32b=results/a8/qwen3vl_32b/face_input_validity/cells_qwen3vl.jsonl \
        --output research/reproduction/a8/face_input_validity_summary_a8.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402

ROOT = a8_design.ROOT
FROZEN = ROOT / "scripts" / "face_input_validity.py"
LOOKUP = ROOT / "data" / "eu_emotions_human_entropy.json"


def load_frozen():
    pinned = json.loads(a8_design.block_bytes())["frozen_code_sha256"]["scripts/face_input_validity.py"]
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if digest != pinned:
        raise SystemExit(f"{FROZEN} sha256 {digest} is not the frozen {pinned}.")
    for p in (str(ROOT / "src"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from scripts import face_input_validity as fiv

    return fiv


def summary_bytes(pairs: Sequence[Tuple[str, Path]], lookup: Path = LOOKUP) -> bytes:
    """The frozen summarise, over these models in this order, as bytes."""
    fiv = load_frozen()
    human = json.loads(Path(lookup).read_text(encoding="utf-8"))["trials"]
    states = {name: fiv.read_cells(Path(path)) for name, path in pairs}
    out = fiv.decide({m: fiv.summarise_model(state["rows"], human) for m, state in states.items()})
    for model, state in states.items():
        out[model]["n_partial_lines_dropped"] = state["n_partial_dropped"]
    return (json.dumps(out, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_pair(text: str) -> Tuple[str, Path]:
    name, _, path = text.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    return name, Path(path)


def main(argv: Optional[Sequence[str]] = None) -> bytes:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=parse_pair, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--lookup", type=Path, default=LOOKUP)
    ap.add_argument("--allow_any_names", action="store_true", help="Equivalence runs on the frozen files only.")
    args = ap.parse_args(argv)
    if not args.allow_any_names:
        design = a8_design.load_design()
        active = a8_design.active_models(design)
        names = [n for n, _ in args.cells]
        if names != active:
            raise SystemExit(f"The A8 family is {active}, in that order; got {names}.")
        for name, path in args.cells:
            if not any(r in Path(path).parts for r in ("a8",)):
                raise SystemExit(f"{path} is not an A8 cells file.")
    for _, path in args.cells:
        if not Path(path).exists():
            raise SystemExit(f"{path} does not exist; a missing model is not an empty one.")
    if args.output.exists():
        raise SystemExit(f"{args.output} exists.")
    data = summary_bytes(args.cells, args.lookup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(hashlib.sha256(data).hexdigest())
    return data


if __name__ == "__main__":
    main()
