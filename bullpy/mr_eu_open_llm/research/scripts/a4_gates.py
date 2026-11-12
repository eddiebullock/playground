#!/usr/bin/env python3
"""A4's `--gates` file, built from the frozen outputs rather than typed (DECISIONS 2026-11-12, A4 ruling 6).

`scripts/a4_analyse.py` reads, per model, `a1_passed`, `a3_passed` and `g2_g4` from the file passed as
`--gates`, and treats anything but the JSON value `true` as not passed. This script fills each value
with A2's frozen `outside_gates` (`scripts/analyse_stimulus_interventions.py`) on `single_face`, the
route A4 sends: `vision_dependent` from the A1 summary and from the A3 summary at its pinned bytes, and
G2 to G4 from the model's vision-gate file. These are locked results (RESULTS_REGISTRY), not A4 data.

    PYTHONPATH=.:src python research/scripts/a4_gates.py   ->  research/reproduction/a4_gates.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUTPUT = Path("research/reproduction/a4_gates.json")
MODELS = ("gemma4", "qwen3vl", "molmo2")  # the Holm family and the descriptive model (scripts/a4_design.py)


def build(root: Path = ROOT) -> Dict[str, Dict[str, Any]]:
    from scripts.analyse_stimulus_interventions import outside_gates  # frozen, tag a2-freeze

    out = {}
    for model in MODELS:
        g = outside_gates(root / "research" / "reproduction", model, "single_face",
                          a3_output_dir=root / "results" / "face_input_validity")
        out[model] = {"a1_passed": g["a1"], "a3_passed": g["a3"], "g2_g4": g["g2_g4"]}
    return out


def main() -> int:
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    (ROOT / OUTPUT).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
