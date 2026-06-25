#!/usr/bin/env python3
"""Reliability diagram and ECE bar chart from augmented eval JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from scripts.selective_prediction import (
    _confidence_from_entropy,
    expected_calibration_error,
)


def _trial_pairs(trials: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[bool]]:
    entropies: List[float] = []
    correct: List[bool] = []
    for t in trials:
        s1 = t.get("stage1") or {}
        s2 = t.get("stage2") or {}
        ent = s1.get("semantic_entropy")
        corr = s2.get("correct")
        if ent is None or corr is None or ent != ent:
            continue
        entropies.append(float(ent))
        correct.append(bool(corr))
    return entropies, correct


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray([1.0 if c else 0.0 for c in correct], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Dict[str, Any]] = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)
        count = int(np.sum(mask))
        if count == 0:
            out.append(
                {
                    "bin": i,
                    "confidence_lo": float(lo),
                    "confidence_hi": float(hi),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        out.append(
            {
                "bin": i,
                "confidence_lo": float(lo),
                "confidence_hi": float(hi),
                "count": count,
                "mean_confidence": float(np.mean(conf[mask])),
                "accuracy": float(np.mean(acc[mask])),
            }
        )
    return out


def plot_reliability(
    bins: List[Dict[str, Any]],
    *,
    title: str,
    ece: float | None,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    xs = [b["mean_confidence"] for b in bins if b["count"] > 0]
    ys = [b["accuracy"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    if xs:
        ax.scatter(xs, ys, s=[max(20, 3 * c) for c in counts], alpha=0.8, label="bins")
    subtitle = f"ECE={ece:.3f}" if ece is not None else "ECE=n/a"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("confidence (1 - norm. entropy)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ece_bars(
    labels: List[str],
    eces: List[float | None],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    vals = [e if e is not None else 0.0 for e in eces]
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    ax.bar(range(len(labels)), vals, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ECE")
    ax.set_title("Expected calibration error by condition")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot calibration from augmented eval JSONs")
    ap.add_argument("--input", type=Path, action="append", required=True)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stats/figures"),
    )
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    labels: List[str] = []
    eces: List[float | None] = []
    for p in args.input:
        data = json.loads(p.read_text(encoding="utf-8"))
        trials = data.get("trials") or []
        ent, corr = _trial_pairs(trials)
        if not ent:
            continue
        conf = _confidence_from_entropy(ent).tolist()
        ece = expected_calibration_error(conf, corr, n_bins=args.n_bins)
        bins = reliability_bins(conf, corr, n_bins=args.n_bins)
        label = p.stem.replace("eval_artifact_", "").replace("eval_v2_", "")
        labels.append(label)
        eces.append(ece)
        plot_reliability(
            bins,
            title=label,
            ece=ece,
            out_path=args.output_dir / f"reliability_{label}.png",
        )

    if labels:
        plot_ece_bars(labels, eces, args.output_dir / "ece_by_condition.png")
        print(f"Wrote calibration figures to {args.output_dir}")


if __name__ == "__main__":
    main()
