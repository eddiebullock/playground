#!/usr/bin/env python3
"""Build consolidated artifact results table from augmented eval JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PARADIGM_FOOTNOTE = (
    "Human EU-Emotions benchmarks often report 6AFC group accuracy; this harness uses "
    "deterministic 4AFC foils (protocol v2). Do not headline strict model-vs-human "
    "superiority without paradigm alignment."
)


def _condition_label(data: Dict[str, Any], path: Path) -> str:
    model = data.get("model") or "?"
    condition = data.get("condition") or "?"
    frame_mode = data.get("frame_mode")
    lora = data.get("lora_adapter")
    prompt_mode = data.get("stage2_prompt_mode")
    parts = [f"{model}_{condition}"]
    if frame_mode and frame_mode != "composite_grid":
        parts.append(frame_mode)
    if lora:
        parts.append("finetuned")
    elif prompt_mode == "finetune_label":
        parts.append("finetune_prompt")
    elif "finetune" in str(path).lower() and "post_ft" in str(path).lower():
        parts.append("post_ft")
    return "/".join(parts)


def _row_from_eval(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    am = data.get("artifact_metrics") or {}
    sp = am.get("selective_prediction") or {}
    tr = am.get("tolerant_rescore") or {}
    return {
        "source_file": str(path),
        "model": data.get("model"),
        "condition": data.get("condition"),
        "frame_mode": data.get("frame_mode") or "composite_grid",
        "condition_label": _condition_label(data, path),
        "protocol_version": data.get("protocol_version"),
        "n_scored": data.get("n_scored"),
        "strict_4afc_accuracy": data.get("accuracy"),
        "tolerant_4afc_accuracy": tr.get("tolerant_4afc_accuracy"),
        "free_response_judge_accuracy": am.get("free_response_judge_accuracy"),
        "ece": sp.get("expected_calibration_error"),
        "auroc_selective_prediction": sp.get("auroc_confidence_vs_correct"),
        "mean_semantic_entropy": data.get("mean_semantic_entropy"),
        "low_entropy_subset_accuracy": sp.get("low_entropy_subset_accuracy"),
        "low_entropy_subset_n": sp.get("low_entropy_subset_n"),
    }


def discover_eval_files(
    roots: List[Path],
    *,
    patterns: Optional[List[str]] = None,
) -> List[Path]:
    pats = patterns or ["eval_artifact_*.json", "eval_v2_*.json"]
    found: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pat in pats:
            found.extend(sorted(root.rglob(pat)))
    # Prefer artifact over raw v2 when both exist for same run
    by_stem: Dict[str, Path] = {}
    for p in found:
        key = str(p.parent / p.name.replace("eval_artifact_", "eval_v2_"))
        if "eval_artifact_" in p.name:
            by_stem[key] = p
        elif key not in by_stem:
            by_stem[key] = p
    return sorted(by_stem.values())


def _run_kind(path: Path, data: Dict[str, Any]) -> str:
    s = str(path).lower()
    if "/ablation/" in s or "native_video" in s or "composite_grid" in path.name:
        if "/ablation/" in s:
            return "ablation"
    if "/finetune/" in s or data.get("lora_adapter"):
        return "finetune"
    return "baseline"


def _dedupe_key(row: Dict[str, Any], kind: str) -> str:
    model = row.get("model") or "?"
    condition = row.get("condition") or "?"
    frame_mode = row.get("frame_mode") or "composite_grid"
    label = row.get("condition_label") or ""
    if kind == "finetune":
        return f"{model}|{condition}|{label}"
    if kind == "ablation":
        return f"{model}|{condition}|{frame_mode}|{row.get('n_scored')}"
    return f"{model}|{condition}|{frame_mode}"


def _dedupe_rows(rows: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    """Keep newest file per key; prefer higher n_scored on ties."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = _dedupe_key(row, kind)
        prev = buckets.get(key)
        if prev is None:
            buckets[key] = row
            continue
        prev_n = prev.get("n_scored") or 0
        row_n = row.get("n_scored") or 0
        prev_mtime = Path(prev["source_file"]).stat().st_mtime
        row_mtime = Path(row["source_file"]).stat().st_mtime
        if row_n > prev_n or (row_n == prev_n and row_mtime > prev_mtime):
            buckets[key] = row
    return sorted(buckets.values(), key=lambda r: (r.get("model") or "", r.get("condition") or ""))


def build_table(
    paths: List[Path],
    *,
    min_baseline_n: int = 118,
) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "artifact_metrics" not in data:
            continue
        row = _row_from_eval(data, p)
        row["run_kind"] = _run_kind(p, data)
        all_rows.append(row)

    baseline_raw = [r for r in all_rows if r["run_kind"] == "baseline"]
    baseline_full = [r for r in baseline_raw if (r.get("n_scored") or 0) >= min_baseline_n]
    baseline = _dedupe_rows(baseline_full, "baseline")

    ablation = _dedupe_rows(
        [r for r in all_rows if r["run_kind"] == "ablation"],
        "ablation",
    )
    finetune = _dedupe_rows(
        [r for r in all_rows if r["run_kind"] == "finetune"],
        "finetune",
    )

    return {
        "footnote": PARADIGM_FOOTNOTE,
        "sections": {
            "baselines": baseline,
            "frame_ablation": ablation,
            "finetune_supplementary": finetune,
        },
        "n_rows": len(baseline) + len(ablation) + len(finetune),
        "rows": baseline + ablation + finetune,
    }


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{100.0 * v:.1f}%"


def _num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _render_rows(rows: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for r in rows:
        out.append(
            "| {label} | {strict} | {tol} | {judge} | {ece} | {auroc} | {hsem} | {n} |".format(
                label=r.get("condition_label", "?"),
                strict=_pct(r.get("strict_4afc_accuracy")),
                tol=_pct(r.get("tolerant_4afc_accuracy")),
                judge=_pct(r.get("free_response_judge_accuracy")),
                ece=_num(r.get("ece")),
                auroc=_num(r.get("auroc_selective_prediction")),
                hsem=_num(r.get("mean_semantic_entropy")),
                n=r.get("n_scored") or "—",
            )
        )
    return out


def to_markdown(table: Dict[str, Any]) -> str:
    header = (
        "| Model / condition | Strict 4AFC | Tolerant 4AFC | Free-response judge | "
        "ECE | AUROC | Mean H_sem | n |"
    )
    sep = "|-------------------|-------------|---------------|---------------------|-----|-------|------------|---|"
    sections = table.get("sections") or {}
    lines = ["# EU-Emotions artifact master table", ""]

    baselines = sections.get("baselines") or table.get("rows") or []
    if baselines:
        lines.extend(["## Baselines (118 trials)", "", header, sep])
        lines.extend(_render_rows(baselines))
        lines.append("")

    ablation = sections.get("frame_ablation") or []
    if ablation:
        lines.extend(["## Frame policy ablation", "", header, sep])
        lines.extend(_render_rows(ablation))
        lines.append("")

    finetune = sections.get("finetune_supplementary") or []
    if finetune:
        lines.extend(["## Fine-tuning (supplementary; not headline)", "", header, sep])
        lines.extend(_render_rows(finetune))
        lines.append("")

    lines.append(f"_{table.get('footnote', PARADIGM_FOOTNOTE)}_")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build artifact master results table")
    ap.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Eval JSON or directory (repeatable). Default: results/baseline, results/finetune, results/ablation",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/stats/artifact_master_table.json"),
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/stats/artifact_master_table.md"),
    )
    args = ap.parse_args()
    roots = args.input or [
        Path("results/baseline"),
        Path("results/finetune"),
        Path("results/ablation"),
    ]
    paths = discover_eval_files([Path(r) for r in roots])
    table = build_table(paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(to_markdown(table) + "\n", encoding="utf-8")
    print(f"Wrote {len(table['rows'])} rows to {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
