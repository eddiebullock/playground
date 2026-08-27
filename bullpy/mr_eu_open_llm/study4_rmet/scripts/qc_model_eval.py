"""
QC for study4 RMET model evals: chance vs systematic responding.

Especially useful for GPT-5 (acc≈chance): distinguishes
  - near-random 4AFC (high entropy, modal ≈ chance)
  - systematic but wrong (low entropy / high sample agreement, still ~25% correct)

Usage (repo root):
  python study4_rmet/scripts/qc_model_eval.py --model gpt5
  python study4_rmet/scripts/qc_model_eval.py --model gpt5 --compare claude_opus,gemini_flash,qwen3vl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import binomtest

STUDY4_ROOT = Path(__file__).resolve().parents[1]


def _load_eval(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_eval(model: str, seed: int = 42) -> Path:
    d = STUDY4_ROOT / "results" / "model" / model
    full = sorted(d.glob(f"rmet_eval_{model}_full_seed{seed}.json"))
    if full:
        return full[-1]
    any_json = sorted(d.glob(f"rmet_eval_{model}_*_seed{seed}.json"))
    if not any_json:
        raise FileNotFoundError(f"No eval JSON for model={model} under {d}")
    return max(any_json, key=lambda p: p.stat().st_size)


def _shannon(counts: Counter, k: int) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return float(h)


def qc_one(data: Dict[str, Any]) -> Dict[str, Any]:
    trials = list(data["trials"])
    n = len(trials)
    det_correct = np.array([int(t["deterministic"]["correct"]) for t in trials], dtype=int)
    det_parse_fail = np.array([int(t["deterministic"]["parse_fail"]) for t in trials], dtype=int)
    n_scored = int((1 - det_parse_fail).sum())
    n_correct = int(det_correct.sum())
    acc = float(n_correct / n_scored) if n_scored else float("nan")

    # Binomial vs chance 0.25 (one-sided greater: is accuracy above chance?)
    bt_greater = binomtest(n_correct, n_scored, p=0.25, alternative="greater") if n_scored else None
    bt_two = binomtest(n_correct, n_scored, p=0.25, alternative="two-sided") if n_scored else None

    sample_accs = []
    sample_ents = []
    modal_agree = []  # fraction of samples matching modal label
    det_matches_modal = []
    modal_correct = []
    zero_ent = 0
    n_with_samples = 0
    pred_labels = []
    correct_labels = []

    for t in trials:
        opts = list(t["options"])
        correct = t["correct_label"]
        det = t["deterministic"]["prediction"]
        preds = [p for p in t["samples"]["predictions"] if p is not None]
        correct_labels.append(correct)
        pred_labels.append(det)

        if not preds:
            continue
        n_with_samples += 1
        counts = Counter(preds)
        ent = _shannon(counts, len(opts))
        sample_ents.append(ent)
        if ent == 0.0:
            zero_ent += 1
        modal, modal_n = counts.most_common(1)[0]
        modal_agree.append(modal_n / len(preds))
        sample_accs.append(sum(1 for p in preds if p.lower() == correct.lower()) / len(preds))
        det_matches_modal.append(bool(det is not None and det.lower() == modal.lower()))
        modal_correct.append(bool(modal.lower() == correct.lower()))

    # If responses were uniform random among 4 options, expected entropy = ln(4)≈1.386
    h_max = math.log(4.0)
    mean_ent = float(np.nanmean(sample_ents)) if sample_ents else float("nan")
    mean_modal_agree = float(np.nanmean(modal_agree)) if modal_agree else float("nan")

    # Consistency of wrong answers: among items where modal is wrong, mean modal agreement
    wrong_modal_agree = [
        a for a, mc in zip(modal_agree, modal_correct) if not mc
    ]
    right_modal_agree = [
        a for a, mc in zip(modal_agree, modal_correct) if mc
    ]

    # Item-level: det vs sample majority disagreement rate
    det_modal_disagree = 1.0 - float(np.mean(det_matches_modal)) if det_matches_modal else float("nan")

    # Most common deterministic predictions (are they collapsing to a few labels?)
    det_counter = Counter(p for p in pred_labels if p is not None)
    # Per-option position bias (1..4)
    pos_counts = Counter()
    for t in trials:
        det = t["deterministic"]["prediction"]
        if det is None:
            continue
        opts = [o.lower() for o in t["options"]]
        try:
            pos_counts[opts.index(det.lower()) + 1] += 1
        except ValueError:
            pass

    # Expected under chance: each position ~ n/4
    pos_chi = None
    if sum(pos_counts.values()) == n_scored and n_scored:
        obs = np.array([pos_counts.get(i, 0) for i in range(1, 5)], dtype=float)
        exp = np.full(4, n_scored / 4.0)
        # Pearson chi-square
        pos_chi = float(np.sum((obs - exp) ** 2 / exp))

    verdict_bits = []
    if bt_greater is not None and bt_greater.pvalue >= 0.05:
        verdict_bits.append("accuracy not above chance (binomial greater p>=.05)")
    elif bt_greater is not None:
        verdict_bits.append("accuracy above chance")

    if sample_ents and mean_ent < 0.5 * h_max and mean_modal_agree >= 0.7:
        verdict_bits.append(
            "samples are low-entropy / high modal agreement → systematic (not near-random) responding"
        )
    elif sample_ents and mean_ent > 0.85 * h_max:
        verdict_bits.append("samples near max entropy → near-random among options")
    else:
        verdict_bits.append("sample entropy intermediate")

    if wrong_modal_agree and float(np.mean(wrong_modal_agree)) >= 0.7:
        verdict_bits.append(
            "wrong items still show high within-item agreement → consistently wrong, not guessing"
        )

    out = {
        "model": data.get("model"),
        "api_model": data.get("api_model"),
        "arm": data.get("arm"),
        "n_items": n,
        "n_scored_deterministic": n_scored,
        "n_correct_deterministic": n_correct,
        "accuracy_deterministic": acc,
        "chance": 0.25,
        "binomial_vs_chance": {
            "n_correct": n_correct,
            "n_trials": n_scored,
            "p_greater": float(bt_greater.pvalue) if bt_greater else None,
            "p_two_sided": float(bt_two.pvalue) if bt_two else None,
            "estimate": float(bt_greater.proportion_estimate) if bt_greater else None,
        },
        "parse_failures_deterministic": int(det_parse_fail.sum()),
        "samples": {
            "n_items_with_samples": n_with_samples,
            "mean_sample_accuracy": float(np.nanmean(sample_accs)) if sample_accs else None,
            "mean_entropy": mean_ent,
            "entropy_max_4afc": h_max,
            "entropy_frac_of_max": (mean_ent / h_max) if sample_ents else None,
            "n_zero_entropy_items": zero_ent,
            "mean_modal_agreement": mean_modal_agree,
            "mean_modal_agreement_when_modal_wrong": float(np.mean(wrong_modal_agree))
            if wrong_modal_agree
            else None,
            "mean_modal_agreement_when_modal_right": float(np.mean(right_modal_agree))
            if right_modal_agree
            else None,
            "modal_accuracy": float(np.mean(modal_correct)) if modal_correct else None,
            "det_vs_modal_disagree_rate": det_modal_disagree,
        },
        "response_bias": {
            "det_label_top": det_counter.most_common(8),
            "option_position_counts_1to4": {str(i): int(pos_counts.get(i, 0)) for i in range(1, 5)},
            "option_position_chi2_vs_uniform": pos_chi,
        },
        "verdict": verdict_bits,
        "per_item": [
            {
                "item": int(t["item"]),
                "correct_label": t["correct_label"],
                "det_prediction": t["deterministic"]["prediction"],
                "det_correct": bool(t["deterministic"]["correct"]),
                "sample_accuracy": t["samples"]["accuracy"],
                "sample_entropy": t["samples"]["entropy"],
                "sample_distribution": t["samples"]["distribution"],
            }
            for t in trials
        ],
    }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt5")
    ap.add_argument(
        "--compare",
        default="claude_opus,gemini_flash,qwen3vl,gemma4,molmo2",
        help="Comma-separated other models for a side-by-side table",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    models = [args.model] + [m.strip() for m in args.compare.split(",") if m.strip()]
    # de-dupe preserving order
    seen = set()
    models = [m for m in models if not (m in seen or seen.add(m))]

    summaries: Dict[str, Any] = {"primary": args.model, "models": {}}
    rows = []
    for m in models:
        path = _resolve_eval(m, seed=args.seed)
        data = _load_eval(path)
        qc = qc_one(data)
        qc["source"] = str(path)
        summaries["models"][m] = {k: v for k, v in qc.items() if k != "per_item"}
        summaries["models"][m]["per_item_path_note"] = "see primary detailed JSON"
        rows.append(
            {
                "model": m,
                "acc_det": qc["accuracy_deterministic"],
                "p_greater_chance": qc["binomial_vs_chance"]["p_greater"],
                "mean_entropy": qc["samples"]["mean_entropy"],
                "entropy_frac_max": qc["samples"]["entropy_frac_of_max"],
                "n_zero_ent": qc["samples"]["n_zero_entropy_items"],
                "mean_modal_agree": qc["samples"]["mean_modal_agreement"],
                "modal_agree_when_wrong": qc["samples"]["mean_modal_agreement_when_modal_wrong"],
                "modal_acc": qc["samples"]["modal_accuracy"],
            }
        )

    outdir = args.outdir or (STUDY4_ROOT / "results" / "model" / args.model / "qc")
    outdir.mkdir(parents=True, exist_ok=True)

    primary_path = _resolve_eval(args.model, seed=args.seed)
    primary_qc = qc_one(_load_eval(primary_path))
    primary_qc["source"] = str(primary_path)
    primary_qc["comparison_table"] = rows

    detail = outdir / f"qc_{args.model}_full_seed{args.seed}.json"
    detail.write_text(json.dumps(primary_qc, indent=2) + "\n", encoding="utf-8")
    summary_path = outdir / f"qc_panel_summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps({"comparison_table": rows, "models": summaries["models"]}, indent=2) + "\n")

    # Human-readable brief
    md = outdir / f"qc_{args.model}_brief.md"
    v = primary_qc
    lines = [
        f"# QC: {args.model}",
        "",
        f"- Source: `{v['source']}`",
        f"- Deterministic accuracy: **{v['accuracy_deterministic']:.3f}** "
        f"({v['n_correct_deterministic']}/{v['n_scored_deterministic']}; chance=0.25)",
        f"- Binomial vs chance (greater): p={v['binomial_vs_chance']['p_greater']:.4f}",
        f"- Binomial vs chance (two-sided): p={v['binomial_vs_chance']['p_two_sided']:.4f}",
        f"- Mean sample entropy: {v['samples']['mean_entropy']:.3f} "
        f"({100*(v['samples']['entropy_frac_of_max'] or 0):.0f}% of ln4 max); "
        f"zero-entropy items: {v['samples']['n_zero_entropy_items']}/36",
        f"- Mean modal agreement: {v['samples']['mean_modal_agreement']:.3f}",
        f"- Modal agreement when wrong: {v['samples']['mean_modal_agreement_when_modal_wrong']}",
        f"- Modal accuracy: {v['samples']['modal_accuracy']}",
        "",
        "## Verdict",
    ]
    for bit in v["verdict"]:
        lines.append(f"- {bit}")
    lines += ["", "## Comparison (acc / entropy / modal agree when wrong)", ""]
    lines.append("| model | acc_det | p>chance | mean H | H/Hmax | modal_agree | agree|wrong | modal_acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['acc_det']:.3f} | {r['p_greater_chance']:.3f} | "
            f"{r['mean_entropy']:.3f} | {r['entropy_frac_max']:.2f} | "
            f"{r['mean_modal_agree']:.3f} | {r['modal_agree_when_wrong'] if r['modal_agree_when_wrong'] is not None else float('nan'):.3f} | "
            f"{r['modal_acc']:.3f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md.read_text(encoding="utf-8"))
    print(f"\nwrote {detail}\nwrote {summary_path}\nwrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
