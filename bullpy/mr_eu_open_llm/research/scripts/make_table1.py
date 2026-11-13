#!/usr/bin/env python3
"""Table 1: the profile of each model across RQ0, RQ1 and RQ2, built from locked outputs only (2026-11-13).

Every cell copies a class or a value from a locked output; the table adds no rule of its own (results.md,
section 8). An output counts as locked when its analysis has run: A1, A3 and the natural-clip scores are locked
(2026-11-09); A2, A4 and A9 count only once `research/reproduction/<amendment>_run_note.json` reads complete,
and until then their cells read "pending". Molmo2 is outside the A2 and A4 families on its gate failures, so its
cells there read "not eligible". A2 classes are the verdicts after the fill-rule ruling (DECISIONS 2026-11-12),
from `a2_extras_<model>.json`, counted per route.

Writes research/paper/v2/table1.md, with each input's sha256.

Usage (project root):
    python research/scripts/make_table1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3vl", "gemma4", "molmo2")
ELIGIBLE = ("gemma4", "qwen3vl")
NAMES = {"gemma4": "Gemma 4", "qwen3vl": "Qwen3-VL", "molmo2": "Molmo2"}
ROUTES = ("single_face", "video_face")
SHORT = {"matches_prediction": "Matches", "opposes_prediction": "Opposes", "null": "Null", "inconclusive": "Inconclusive"}
READING = {(True, True): "Face use on both inputs", (False, True): "The grid limited A1; single_face from here",
           (True, False): "A1 evidence needs inspection", (False, False): "No demonstrated face use"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r3(x: Optional[float]) -> str:
    s = f"{float(x):.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def p(x: float) -> str:
    """.0002, .00009, .012: five decimals below .001 without trailing zeros, else three."""
    s = f"{x:.5f}".rstrip("0") if x < 0.001 else f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def locked(repro: Path, amendment: str) -> bool:
    note = repro / f"{amendment}_run_note.json"
    return note.exists() and json.loads(note.read_text(encoding="utf-8")).get("status") == "complete"


def build(root: Path = ROOT) -> str:
    repro = root / "research" / "reproduction"
    used: Dict[str, str] = {}

    def load(path: Path) -> Any:
        used[str(path.relative_to(root))] = sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))

    a1 = load(repro / "rotation_summary.json")
    a3 = load(repro / "face_input_validity_summary.json")
    natural = {m: load(repro / f"same_clip_family_model_{m}.json") for m in MODELS}
    a4 = load(root / "results" / "a4" / "summary.json") if locked(repro, "a4") else None
    a9 = load(root / "results" / "a9" / "summary.json") if locked(repro, "a9") else None
    a2: Dict[str, Optional[Dict[str, Any]]] = {m: None for m in ELIGIBLE}
    if locked(repro, "a2"):
        for m in ELIGIBLE:
            out_path, ext_path = repro / f"stimulus_interventions_{m}.json", repro / f"a2_extras_{m}.json"
            out, ext = load(out_path), load(ext_path)
            if ext.get("a2_output_sha256") != sha256(out_path):
                raise SystemExit(f"{ext_path.name} was not computed from {out_path.name}.")
            a2[m] = {"out": out, "ext": ext}

    rows: List[str] = ["| Model | RQ0: A1 delta (Holm *P*) | RQ0: A3 delta (Holm *P*) | RQ0 reading | RQ1, natural clips: pattern score (*P*) | "
                       "RQ1, randomised (A4): basic; conversational | A4 on a second sample (A9) | RQ2 (A2), after the fill rule: Gate B; Matches, Opposes, Null, Inconclusive |",
                       "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        r0 = (bool(a1[m]["vision_dependent"]), bool(a3[m]["vision_dependent"]))
        nat = natural[m]
        status = "confirmatory" if str(nat.get("rq1_status", "")).startswith("confirmatory") else "descriptive"
        nat_cell = f"{r3(nat['composite']['score'])} ({p(nat['composite']['p'])}), {status}"
        if m not in ELIGIBLE:
            a4_cell = a9_cell = a2_cell = "not eligible (vision gates)"
        else:
            if a4 is None:
                a4_cell = "pending"
            else:
                h = (a4.get("models", {}).get(m) or {}).get("headline") or {}
                a4_cell = f"{h.get('basic')}; {h.get('conversational')}"
            if a9 is None:
                a9_cell = "pending"
            else:
                e = a9.get("models", {}).get(m) or {}
                a9_cell = f"{e.get('decision')}; {((e.get('replication') or {}).get('reading'))}"
            if a2[m] is None:
                a2_cell = "pending"
            else:
                parts = []
                for route in ROUTES:
                    block = a2[m]["out"].get(route)
                    if not block:
                        continue
                    contrasts = (a2[m]["ext"].get(route) or {}).get("contrasts", {})
                    counts = Counter(SHORT[c["verdict_after_fill_rule"]] for c in contrasts.values())
                    gate = "pass" if (block.get("gates") or {}).get("gate_b") is True else "fail"
                    parts.append(f"{route}: Gate B {gate}; {counts['Matches']}, {counts['Opposes']}, {counts['Null']}, {counts['Inconclusive']}")
                a2_cell = "<br>".join(parts) or "no route"
        rows.append(f"| {NAMES[m]} | {r3(a1[m]['delta_real_minus_mismatch'])} ({p(a1[m]['p_delta_holm'])}) | "
                    f"{r3(a3[m]['delta_real_minus_mismatch'])} ({p(a3[m]['p_delta_holm'])}) | {READING[r0]} | {nat_cell} | {a4_cell} | {a9_cell} | {a2_cell} |")
    sources = "\n".join(f"- `{k}` (sha256 `{v[:8]}...`)" for k, v in sorted(used.items()))
    return ("# Table 1 | What each model has shown\n\n"
            "<!-- Built by research/scripts/make_table1.py from locked outputs only; do not edit by hand. -->\n\n"
            "Each cell copies a class or value from a locked output; no rule is added (Results, section 8). Models are never pooled. "
            "\"Pending\" means the analysis has not run. RQ1 on natural clips supports alignment with the human pattern, not use of the same cues. "
            "A4 headlines need WC and EA to agree; \"Not aligned\" is always on these renders. A2 counts are of the 16 confirmatory contrasts per route.\n\n"
            + "\n".join(rows) + "\n\nInputs:\n\n" + sources + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "paper" / "v2" / "table1.md")
    args = ap.parse_args(argv)
    args.out.write_text(build(), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
