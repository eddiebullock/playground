#!/usr/bin/env python3
"""Registry tables from locked Phase 3 outputs, so that no number is typed by hand (written 2026-11-12).

Run only after an analysis's run note exists (`research/reproduction/<amendment>_run_note.json`, written by
`post_phase3.py analyse`). It reads the output files and prints markdown tables for RESULTS_REGISTRY.md,
headed by each file's path and sha256. It computes nothing new: every value is a field of an output file,
formatted in the paper's style (no leading zero, three decimals for effects; p-values to five decimals
below .001). It never changes a verdict: A2's table shows the frozen verdict, the verdict after the fill-rule
ruling (`a2_extras_<model>.json`, DECISIONS 2026-11-12) and the .01 reading side by side.

Usage (project root):
    python research/scripts/registry_tables.py a2
    python research/scripts/registry_tables.py a4
    python research/scripts/registry_tables.py a9   (added 2026-11-14, before any A9 output, on A9's frozen schema)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "research" / "reproduction"
MODELS = ("gemma4", "qwen3vl")
NAMES = {"gemma4": "Gemma 4", "qwen3vl": "Qwen3-VL", "molmo2": "Molmo2"}
ROUTES = ("single_face", "video_face")
VERDICTS = {"matches_prediction": "Matches", "opposes_prediction": "Opposes", "null": "Null", "inconclusive": "Inconclusive"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r3(x: Optional[float]) -> str:
    """.402, -.218, 1.25; '-' for missing."""
    if x is None:
        return "-"
    s = f"{float(x):.3f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def p5(x: Optional[float]) -> str:
    if x is None:
        return "-"
    x = float(x)
    s = f"{x:.5f}" if x < 0.001 else f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def interval(ci: Optional[Sequence[float]]) -> str:
    return "-" if not ci else f"{r3(ci[0])} to {r3(ci[1])}"


def header(path: Path) -> str:
    rel = path.relative_to(ROOT) if ROOT in path.parents else path
    return f"Source `{rel}` (sha256 `{sha256(path)[:8]}...`), printed by `research/scripts/registry_tables.py`."


def a2_rows(out: Mapping[str, Any], extras: Optional[Mapping[str, Any]]) -> List[str]:
    lines = ["| Route | Contrast | Actors | Change (predicted sign) | 95% CI | Step-down P, one-sided | Two-sided | Frozen verdict | After fill rule | At .01 | 90% equivalence interval |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for route in ROUTES:
        block = out.get(route)
        if not block:
            continue
        rows = block["results"]["primary_blind_acted"]
        ext = ((extras or {}).get(route) or {}).get("contrasts", {})
        for cid, row in rows.items():
            sign = row.get("direction") or 1
            change = None if row.get("mean_change") is None else row["mean_change"] * sign
            ci = row.get("ci95")
            ci_signed = None if not ci else sorted(v * sign for v in ci)
            after = (ext.get(cid) or {}).get("verdict_after_fill_rule", row["verdict"] if extras is None else None)
            if after is None:
                raise SystemExit(f"{route} {cid}: no fill-rule verdict in the extras.")
            if after == "matches_prediction":
                at01 = "holds" if (row.get("p_maxt") or 1) < 0.01 else "not at .01"
            elif after == "opposes_prediction":
                at01 = "holds" if (row.get("p_maxt_two_sided") or 1) < 0.01 else "not at .01"
            else:
                at01 = "-"
            eq = (ext.get(cid) or {}).get("equivalence_interval_90")
            lines.append(f"| {route} | {cid} | {row.get('n_actors', '-')} | {r3(change)} | {interval(ci_signed)} | {p5(row.get('p_maxt'))} | "
                         f"{p5(row.get('p_maxt_two_sided'))} | {VERDICTS.get(row['verdict'], row['verdict'])} | {VERDICTS.get(after, after)} | {at01} | {interval(eq)} |")
    return lines


def a2_gates(out: Mapping[str, Any]) -> List[str]:
    lines = ["| Route | RQ0 | G2 to G4 | Gate B | Delivery | Sham |", "|---|---|---|---|---|---|"]
    for route in ROUTES:
        g = (out.get(route) or {}).get("gates") or {}
        if g:
            lines.append(f"| {route} | {g.get('rq0')} | {g.get('g2_g4')} | {g.get('gate_b')} | {g.get('delivery')} | {g.get('sham')} |")
    return lines


def table_a2(repro: Path = REPRO) -> str:
    parts = []
    for m in MODELS:
        out_path, ext_path = repro / f"stimulus_interventions_{m}.json", repro / f"a2_extras_{m}.json"
        out = json.loads(out_path.read_text(encoding="utf-8"))
        extras = json.loads(ext_path.read_text(encoding="utf-8"))
        if extras.get("a2_output_sha256") != sha256(out_path):
            raise SystemExit(f"{ext_path.name} was not computed from {out_path.name}.")
        parts += [f"### A2, {NAMES[m]}", "", header(out_path), header(ext_path), "", *a2_gates(out), "", *a2_rows(out, extras), ""]
    return "\n".join(parts)


def table_a4(summary_path: Path = ROOT / "results" / "a4" / "summary.json") -> str:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [header(summary_path), "", f"Statistic: {s.get('statistic')}.", "",
             "| Dataset | Model | Signal P | Residual CEA | 95% CI | R_S | Alignment Holm P | Raw CEA (secondary) | Decision |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds, models in s["datasets"].items():
        for m, e in models.items():
            res, raw = e.get("residual") or {}, e.get("raw") or {}
            lines.append(f"| {ds} | {NAMES.get(m, m)} | {p5(e.get('p_signal'))} | {r3(res.get('cfa'))} | {interval(res.get('ci'))} | "
                         f"{r3(res.get('r_s'))} | {p5(e.get('p_holm'))} | {r3(raw.get('cfa'))} | {e.get('decision')} |")
    lines += ["", "| Model | Basic headline | Conversational headline |", "|---|---|---|"]
    for m, e in s.get("models", {}).items():
        h = e.get("headline") or {}
        lines.append(f"| {NAMES.get(m, m)} | {h.get('basic')} | {h.get('conversational')} |")
    return "\n".join(lines)


def table_a9(summary_path: Path = ROOT / "results" / "a9" / "summary.json") -> str:
    """A9's per-model decision, gates and replication reading (scripts/a9_analyse.py output, `models.<model>`)."""
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [header(summary_path), "", f"Statistic: {s.get('statistic')}; design {s.get('design')}.", "",
             "| Model | Signal P | Residual CEA | 95% CI | R_S | Alignment Holm P (within A9) | Raw CEA (secondary) | "
             "Configurations (share lost) | Decision |",
             "|---|---|---|---|---|---|---|---|---|"]
    models = s.get("models", {})
    for m, e in models.items():
        res, raw = e.get("residual") or {}, e.get("raw") or {}
        lines.append(f"| {NAMES.get(m, m)} | {p5(e.get('p_signal'))} | {r3(res.get('cfa'))} | {interval(res.get('ci'))} | "
                     f"{r3(res.get('r_s'))} | {p5(e.get('p_holm'))} | {r3(raw.get('cfa'))} | "
                     f"{e.get('n_configurations', '-')} ({r3(e.get('share_lost'))}) | {e.get('decision')} |")
    lines += ["", "| Model | Gates not passed | Replication reading | A9 decision | A4 basic WC decision (comparator) | "
                  "A4 basic headline (beside) | Reason |",
              "|---|---|---|---|---|---|---|"]
    for m, e in models.items():
        failed = [k for k, v in (e.get("gates") or {}).items() if v is not True] or ["none"]
        rep = e.get("replication") or {}
        lines.append(f"| {NAMES.get(m, m)} | {', '.join(failed)} | {rep.get('reading')} | {rep.get('a9_decision')} | "
                     f"{rep.get('a4_basic_WC_decision')} | {rep.get('a4_basic_headline_beside')} | {rep.get('reason') or '-'} |")
    lines += ["", "| Model | Over ceiling (descriptive) | Residual correlation across faces (descriptive) | "
                  "Earlier operationalisation of lead decision 2 (descriptive) |",
              "|---|---|---|---|"]
    for m, e in models.items():
        res = e.get("residual") or {}
        bound = " (lower bound only)" if res.get("lower_bound_only") else ""
        lines.append(f"| {NAMES.get(m, m)} | {r3(res.get('over_ceiling'))}{bound} | "
                     f"{r3(e.get('face_shared_residual_correlation'))} | {e.get('earlier_rule_reading_descriptive')} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("amendment", choices=("a2", "a4", "a9"))
    args = ap.parse_args(argv)
    note = REPRO / f"{args.amendment}_run_note.json"
    if not note.exists() or json.loads(note.read_text(encoding="utf-8")).get("status") != "complete":
        raise SystemExit(f"{note.name} is missing or not complete: the analysis has not run, so nothing is printed.")
    print({"a2": table_a2, "a4": table_a4, "a9": table_a9}[args.amendment]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
