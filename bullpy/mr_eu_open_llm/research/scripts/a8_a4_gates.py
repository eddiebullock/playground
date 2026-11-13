#!/usr/bin/env python
"""A8: the `--gates` file the frozen scripts/a4_analyse.py accepts, from A8 outputs.

Per code key: `a1_passed` (A8 A1 summary, vision_dependent; null for a model outside the A1
family, which the frozen rq0_status reads as no result), `a3_passed` (A8 A3 summary), and
`g2_g4` (A8 gate file, read as the frozen outside_gates reads it). Both summaries are
accepted only as their A8 producers wrote them.

    python research/scripts/a8_a4_gates.py   ->  research/reproduction/a8/a4_gates.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_design  # noqa: E402
import a8_family  # noqa: E402

OUTPUT = "research/reproduction/a8/a4_gates.json"


def gates(base: Path) -> Dict[str, Dict[str, Any]]:
    design = a8_design.load_design()
    a3 = json.loads((base / a8_family.A3_SUMMARY).read_text(encoding="utf-8"))
    a1 = json.loads((base / a8_family.A1_SUMMARY).read_text(encoding="utf-8"))
    out = {}
    for a8_id in a8_design.active_models(design):
        key = design["models"][a8_id]["code_key"]
        out[key] = {
            "a8_id": a8_id,
            "a1_passed": (a1.get(a8_id) or {}).get("vision_dependent"),
            "a3_passed": (a3.get(a8_id) or {}).get("vision_dependent"),
            "g2_g4": a8_family.gate_result(design, base, a8_id),
        }
    return out


def gate_inputs(base: Path) -> List[Path]:
    """The files `gates` reads: both A8 summaries and each active model's gate file."""
    design = a8_design.load_design()
    files = [base / a8_family.A3_SUMMARY, base / a8_family.A1_SUMMARY]
    files += [a8_family.repro(base, m) / f"vision_gates_{design['models'][m]['code_key']}.json"
              for m in a8_design.active_models(design)]
    return [p for p in files if p.exists()]


def produce(base: Path) -> Path:
    """The gates file, written once, with its provenance (PREREG_A8_DRAFT.md, "Analyses outside the
    launcher"), which A8's A4 pre-run check verifies."""
    a8_family.check_provenance(base / a8_family.A3_SUMMARY)
    a8_family.check_provenance(base / a8_family.A1_SUMMARY)
    path = base / OUTPUT
    if path.exists():
        raise SystemExit(f"{path} exists.")
    path.write_text(json.dumps(gates(base), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a8_family.write_provenance(path, gate_inputs(base), {
        "wrapper_sha256": a8_design.file_sha256(Path(__file__).resolve()),
        "design_sha256": a8_design.block_hash(),
    })
    return path


def main() -> Path:
    root = a8_design.ROOT
    os.chdir(root)
    a8_design.check_own_code(require=True)
    return produce(root)


if __name__ == "__main__":
    main()
