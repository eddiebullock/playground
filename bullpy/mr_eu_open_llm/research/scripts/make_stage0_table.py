#!/usr/bin/env python3
"""Supplementary table: A7 Stage 0 on both eligible models, built from the locked Stage 0 files (2026-11-13).

Reads `research/reproduction/a7_stage0_{gemma4,qwen3vl}.json` (frozen A7 code, 2026-11-10; RESULTS_REGISTRY,
"A7 Stage 0") and the block names from A7's design block, and writes
`research/paper/v2/supplementary_table_a7_stage0.md`. Every cell is a field of those files; nothing is computed
beyond rounding. A field the run did not produce (for example the edit size where the token map failed)
reads "not computed".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "research" / "reproduction"
MODELS = ("gemma4", "qwen3vl")
NAMES = {"gemma4": "Gemma 4", "qwen3vl": "Qwen3-VL"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r3(x: Optional[float]) -> str:
    if x is None:
        return "not computed"
    s = f"{float(x):.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def blocks() -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from scripts import a7_design  # frozen; the block is hash-checked on load, read only for the block names

    return a7_design.load_design()["route_blocks"]


def build() -> str:
    names = blocks()
    rows: List[List[str]] = []
    used = {}
    tables = {}
    for m in MODELS:
        path = REPRO / f"a7_stage0_{m}.json"
        used[str(path.relative_to(ROOT))] = sha256(path)
        tables[m] = json.loads(path.read_text(encoding="utf-8"))

    def row(label: str, fn) -> None:
        rows.append([label] + [fn(tables[m]) for m in MODELS])

    row("Clips (calibration set, outside the analysis set)", lambda t: str(t["n_clips"]))
    row("Completeness tolerance (nats, all 106 words)", lambda t: r3(t["tolerance_nats"]))
    for c, what in (("C1", "determinism, native against a repeat"), ("C2", "exit cut equals the image-free forward"),
                    ("C3", "exit cut does not depend on the image"), ("C4", "holding both blocks is complete")):  # PREREG_A7, Stage 0
        row(f"{c}, {what}: worst change (nats); pass",
            lambda t, c=c: f"{r3(t['checks'][c]['worst_abs_nats'])}; {'pass' if t['checks'][c]['pass'] else 'fail'}")
    row("C5, block A live: ratio of denominator (floor .10)",
        lambda t: f"{r3(t['checks']['C5']['blocks']['A']['ratio'])}; {'live' if t['checks']['C5']['blocks']['A']['live'] else 'dead'}")
    row("C5, block B live: ratio of denominator (floor .10)",
        lambda t: f"{r3(t['checks']['C5']['blocks']['B']['ratio'])}; {'live' if t['checks']['C5']['blocks']['B']['live'] else 'dead'}")
    row("Exit cut complete; blocks exhaust it", lambda t: f"{t['exit_complete']}; {t['blocks_exhaust_exit']}")
    row("Complete cuts shown", lambda t: ", ".join(t["complete_cuts"]) or "none")
    row("Region test: pass; reason", lambda t: f"{t['region_test']['pass']}; {t['region_test'].get('reason')}")
    row("Token map: hit rate over trials", lambda t: f"{r3((t['region_test'].get('token_map') or {}).get('hit_rate'))} of "
                                                     f"{(t['region_test'].get('token_map') or {}).get('n_trials', 'not computed')}")
    row("Leak at buffers 0 and 1: point (upper)", lambda t: "; ".join(
        f"{r3(v.get('point'))} ({r3(v.get('upper'))})" for v in (t["region_test"].get("leaks") or {}).values()) or "not computed")
    row("Arms admitted: route; token", lambda t: f"{t['arms']['route']}; {t['arms']['token']}")
    head = "| Quantity | " + " | ".join(NAMES[m] for m in MODELS) + " |"
    lines = [head, "|---|" + "---|" * len(MODELS)] + ["| " + " | ".join(r) + " |" for r in rows]
    block_note = "; ".join(f"{NAMES[m]}: A = {names[m]['A']}, B = {names[m]['B'] if isinstance(names[m]['B'], str) else ', '.join(names[m]['B'])}"
                           for m in MODELS)
    sources = "\n".join(f"- `{k}` (sha256 `{v[:8]}...`)" for k, v in sorted(used.items()))
    return ("# Supplementary Table: A7 Stage 0, completeness of vision-path cuts\n\n"
            "<!-- Built by research/scripts/make_stage0_table.py from the locked Stage 0 files; do not edit by hand. -->\n\n"
            "On `single_face` with the blind readout, frozen A7 code (tag `a7-freeze`), run on 2026-11-10. A site is a complete cut only if every clip "
            "and every lexicon word stays within the tolerance of the image-free reference. A block is not a cut (DECISIONS 2026-11-08). "
            f"Blocks: {block_note}.\n\n" + "\n".join(lines) + "\n\nInputs:\n\n" + sources + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "paper" / "v2" / "supplementary_table_a7_stage0.md")
    args = ap.parse_args(argv)
    args.out.write_text(build(), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
