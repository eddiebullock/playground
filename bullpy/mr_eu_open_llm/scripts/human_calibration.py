#!/usr/bin/env python3
"""RQ1.2: are model errors concentrated where humans also disagree?

Stratifies model accuracy by tercile of per-item human response entropy and tests H1.2 --
that a meaningful share of errors falls on high-consensus items, i.e. errors are not
fully explained by genuine stimulus ambiguity.

Reported per model:

  accuracy within each human-agreement tercile, with Wilson intervals
  a two-proportion test of high-consensus vs low-consensus accuracy
  the share of all errors that land on high-consensus items (the H1.2 quantity)

An "ambiguity-tracking" model would fail almost exclusively on the high-disagreement
tercile; a miscalibrated one spreads errors across terciles or concentrates them on items
humans found easy. The error share on the high-consensus tercile is the headline number,
because unlike accuracy it does not depend on how hard the tercile happens to be.

Usage:

  python -m scripts.human_calibration \\
    --eval results/baseline/eu_emotions/*/eval_v2_*.json \\
    --human data/eu_emotions_human_entropy.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from scripts.statistics import two_proportion_ztest_vs_human, wilson_ci

TERCILE_NAMES = ("high_consensus", "mid", "high_disagreement")


def tercile_bounds(values: Sequence[float]) -> Tuple[float, float]:
    """Cut points splitting sorted values into three near-equal groups."""
    ordered = sorted(values)
    n = len(ordered)
    if n < 3:
        raise ValueError(f"need at least 3 items to form terciles, got {n}")
    return ordered[n // 3], ordered[2 * n // 3]


def assign_tercile(value: float, bounds: Tuple[float, float]) -> str:
    """Low human entropy means humans agreed, so it is the high-consensus tercile."""
    low, high = bounds
    if value < low:
        return TERCILE_NAMES[0]
    if value < high:
        return TERCILE_NAMES[1]
    return TERCILE_NAMES[2]


def stratify(
    eval_path: Path,
    human_path: Path,
    *,
    entropy_field: str = "human_entropy",
) -> Dict[str, Any]:
    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))["trials"]

    scored: List[Dict[str, Any]] = []
    for trial in metrics.get("trials", []):
        entry = human.get(trial.get("trial_id"))
        stage2 = trial.get("stage2") or {}
        if entry is None or stage2.get("correct") is None:
            continue
        scored.append(
            {
                "trial_id": trial["trial_id"],
                "human_entropy": float(entry[entropy_field]),
                "correct": bool(stage2["correct"]),
                "prediction": stage2.get("prediction"),
                "forced_choice_entropy": stage2.get("forced_choice_entropy"),
            }
        )

    if len(scored) < 3:
        return {
            "eval": eval_path.name,
            "model": metrics.get("model"),
            "n_scored_with_human_data": len(scored),
            "error": "too few scored trials matched to human data to form terciles",
        }

    bounds = tercile_bounds([row["human_entropy"] for row in scored])
    for row in scored:
        row["human_tercile"] = assign_tercile(row["human_entropy"], bounds)

    n_errors_total = sum(1 for row in scored if not row["correct"])
    terciles: Dict[str, Any] = {}
    for name in TERCILE_NAMES:
        rows = [r for r in scored if r["human_tercile"] == name]
        n_correct = sum(1 for r in rows if r["correct"])
        n = len(rows)
        low_ci, high_ci = wilson_ci(n_correct, n) if n else (float("nan"), float("nan"))
        terciles[name] = {
            "n": n,
            "n_correct": n_correct,
            "n_errors": n - n_correct,
            "accuracy": (n_correct / n) if n else None,
            "accuracy_wilson_ci_95": [low_ci, high_ci],
            "share_of_all_errors": ((n - n_correct) / n_errors_total) if n_errors_total else None,
            "mean_human_entropy": (sum(r["human_entropy"] for r in rows) / n) if n else None,
        }

    easy, hard = terciles[TERCILE_NAMES[0]], terciles[TERCILE_NAMES[2]]
    # The z-test is undefined when the pooled proportion is 0 or 1 (statsmodels returns
    # nan after a divide-by-zero warning), so report None rather than a fake p-value.
    pooled_successes = easy["n_correct"] + hard["n_correct"]
    pooled_n = easy["n"] + hard["n"]
    degenerate = pooled_n == 0 or pooled_successes in (0, pooled_n)
    p_value = (
        None
        if degenerate
        else two_proportion_ztest_vs_human(
            easy["n_correct"], easy["n"], hard["n_correct"], hard["n"]
        )
    )

    return {
        "eval": eval_path.name,
        "model": metrics.get("model"),
        "condition": metrics.get("condition"),
        "protocol_version": metrics.get("protocol_version"),
        "entropy_field": entropy_field,
        "n_scored_with_human_data": len(scored),
        "n_errors_total": n_errors_total,
        "tercile_bounds_human_entropy": list(bounds),
        "terciles": terciles,
        "high_vs_low_consensus_accuracy_p": p_value,
        # H1.2: errors on items humans found easy are calibration failures, not ambiguity.
        "h1_2_error_share_on_high_consensus": easy["share_of_all_errors"],
        "trials": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, nargs="+", help="eval JSON path(s) or glob(s)")
    parser.add_argument("--human", default=Path("data/eu_emotions_human_entropy.json"), type=Path)
    parser.add_argument(
        "--entropy_field",
        default="human_entropy",
        choices=["human_entropy", "human_entropy_forced"],
        help="Full 6-way human distribution, or the 5 named emotions renormalised.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths: List[Path] = []
    for pattern in args.eval:
        matched = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matched or [Path(pattern)])

    reports = []
    for path in paths:
        if not path.exists():
            print(f"missing: {path}")
            continue
        report = stratify(path, args.human, entropy_field=args.entropy_field)
        reports.append(report)
        if report.get("error"):
            print(f"{report.get('model')}: {report['error']}")
            continue
        print(f"\n{report['model']} | {report['condition']} | n={report['n_scored_with_human_data']}")
        for name in TERCILE_NAMES:
            block = report["terciles"][name]
            ci = block["accuracy_wilson_ci_95"]
            print(
                f"   {name:<18} n={block['n']:<4} acc={block['accuracy']:.3f} "
                f"[{ci[0]:.3f}, {ci[1]:.3f}]  errors={block['n_errors']:<4} "
                f"share_of_errors={block['share_of_all_errors']}"
            )
        print(f"   high- vs low-consensus accuracy p = {report['high_vs_low_consensus_accuracy_p']}")
        print(f"   H1.2 error share on high-consensus items = {report['h1_2_error_share_on_high_consensus']}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
