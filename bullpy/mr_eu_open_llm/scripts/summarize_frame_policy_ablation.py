#!/usr/bin/env python3
"""Summarize composite_grid vs native_video ablation eval JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _metric_row(data: Dict[str, Any]) -> Dict[str, Any]:
    am = data.get("artifact_metrics") or {}
    sp = am.get("selective_prediction") or {}
    tr = am.get("tolerant_rescore") or {}
    return {
        "path": None,
        "model": data.get("model"),
        "condition": data.get("condition"),
        "frame_mode": data.get("frame_mode"),
        "enforce_multi_frame": data.get("enforce_multi_frame"),
        "n_scored": data.get("n_scored"),
        "strict_4afc": data.get("accuracy"),
        "tolerant_4afc": tr.get("tolerant_4afc_accuracy"),
        "free_response_judge": am.get("free_response_judge_accuracy"),
        "ece": sp.get("expected_calibration_error"),
        "auroc": sp.get("auroc_confidence_vs_correct"),
        "mean_semantic_entropy": data.get("mean_semantic_entropy"),
        "multi_frame_strategy": _dominant_strategy(data),
    }


def _dominant_strategy(data: Dict[str, Any]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for t in data.get("trials") or []:
        strat = t.get("multi_frame_strategy")
        if strat:
            counts[str(strat)] = counts.get(str(strat), 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def summarize(
    composite_path: Path,
    native_path: Path,
) -> Dict[str, Any]:
    composite = json.loads(composite_path.read_text(encoding="utf-8"))
    native = json.loads(native_path.read_text(encoding="utf-8"))
    row_c = _metric_row(composite)
    row_c["path"] = str(composite_path)
    row_n = _metric_row(native)
    row_n["path"] = str(native_path)
    return {
        "composite_grid": row_c,
        "native_video": row_n,
        "delta_native_minus_composite": {
            "strict_4afc": _delta(row_c.get("strict_4afc"), row_n.get("strict_4afc")),
            "tolerant_4afc": _delta(row_c.get("tolerant_4afc"), row_n.get("tolerant_4afc")),
            "free_response_judge": _delta(
                row_c.get("free_response_judge"), row_n.get("free_response_judge")
            ),
            "ece": _delta(row_c.get("ece"), row_n.get("ece")),
            "auroc": _delta(row_c.get("auroc"), row_n.get("auroc")),
            "mean_semantic_entropy": _delta(
                row_c.get("mean_semantic_entropy"), row_n.get("mean_semantic_entropy")
            ),
        },
    }


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{100.0 * v:.1f}%"


def _fmt_delta(v: Optional[float], *, as_pct: bool = True) -> str:
    if v is None:
        return "—"
    if as_pct:
        sign = "+" if v >= 0 else ""
        return f"{sign}{100.0 * v:.1f} pp"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def _num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def to_markdown(summary: Dict[str, Any]) -> str:
    cg = summary["composite_grid"]
    nv = summary["native_video"]
    d = summary["delta_native_minus_composite"]
    model = cg.get("model") or nv.get("model") or "?"
    condition = cg.get("condition") or nv.get("condition") or "?"
    lines = [
        f"## Frame policy ablation ({model}, {condition})",
        "",
        "| Metric | composite_grid | native_video | Δ (native − composite) |",
        "|--------|----------------|--------------|-------------------------|",
        f"| Strict 4AFC | {_pct(cg.get('strict_4afc'))} | {_pct(nv.get('strict_4afc'))} | {_fmt_delta(d.get('strict_4afc'))} |",
        f"| Tolerant 4AFC | {_pct(cg.get('tolerant_4afc'))} | {_pct(nv.get('tolerant_4afc'))} | {_fmt_delta(d.get('tolerant_4afc'))} |",
        f"| Free-response judge (primary) | {_pct(cg.get('free_response_judge'))} | {_pct(nv.get('free_response_judge'))} | {_fmt_delta(d.get('free_response_judge'))} |",
        f"| ECE | {_num(cg.get('ece'))} | {_num(nv.get('ece'))} | {_num(d.get('ece'))} |",
        f"| AUROC (entropy conf.) | {_num(cg.get('auroc'))} | {_num(nv.get('auroc'))} | {_num(d.get('auroc'))} |",
        f"| Mean semantic entropy | {_num(cg.get('mean_semantic_entropy'))} | {_num(nv.get('mean_semantic_entropy'))} | {_num(d.get('mean_semantic_entropy'))} |",
        "",
        f"- composite_grid strategy: `{cg.get('multi_frame_strategy')}` (enforce_multi_frame={cg.get('enforce_multi_frame')})",
        f"- native_video strategy: `{nv.get('multi_frame_strategy')}` (enforce_multi_frame={nv.get('enforce_multi_frame')})",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize frame policy ablation")
    ap.add_argument("--composite", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ablation/frame_policy_summary.json"),
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/ablation/frame_policy_summary.md"),
    )
    args = ap.parse_args()
    summary = summarize(args.composite, args.native)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(to_markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary["delta_native_minus_composite"], indent=2))


if __name__ == "__main__":
    main()
