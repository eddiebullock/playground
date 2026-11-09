#!/usr/bin/env python
"""A8: the frozen A3 summary over the A8 family.

The frozen `scripts/face_input_validity.py --summarise` reads the three frozen models'
cell files and applies Holm over them. This runs the same functions, `read_cells`,
`summarise_model` and `decide`, over the A8 A3 family, and writes the output exactly as the
frozen `summarise` does. The family is computed by `a8_family.a3_family` from the A8 files
(active models whose single_face delivery passed); every cell file must be complete.

Equivalence: `summary_bytes` on the three frozen cell files, in the frozen order (qwen3vl,
gemma4, molmo2), reproduces the frozen summary byte for byte (sha256 c1c64e69...,
`frozen_a3_summary_sha256`); on synthetic cells it equals the frozen `summarise` output.

    python research/scripts/a8_face_input_validity_summary.py

Writes research/reproduction/a8/face_input_validity_summary_a8.json and its provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402
import a8_family  # noqa: E402

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


def produce(base: Path = ROOT) -> Path:
    """Family by rule, completeness checked, summary and provenance written once."""
    design = a8_design.load_design()
    a8_design.check_frozen_code(design)
    a8_design.check_data(design, ["data/eu_emotions_human_entropy.json"], root=ROOT)
    family, excluded = a8_family.a3_family(design, base)
    if not family:
        raise SystemExit(f"No A8 model is in the A3 family: {excluded}")
    output = base / a8_family.A3_SUMMARY
    if output.exists():
        raise SystemExit(f"{output} exists; the A8 A3 summary is written once.")
    data = summary_bytes(family)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    a8_family.write_provenance(output, [p for _, p in family] + [LOOKUP], {
        "family": [n for n, _ in family], "excluded": excluded,
        "wrapper_sha256": a8_design.file_sha256(Path(__file__).resolve()),
        "design_sha256": a8_design.block_hash(),
    })
    print(hashlib.sha256(data).hexdigest())
    return output


def main(argv: Optional[Sequence[str]] = None) -> Path:
    os.chdir(ROOT)
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    a8_design.check_own_code(require=True)
    return produce(ROOT)


if __name__ == "__main__":
    main()
