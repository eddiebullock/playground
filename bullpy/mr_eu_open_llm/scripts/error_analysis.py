"""
Study 1 error analysis: confusion matrix, per-label stats, top confused pairs for Study 2.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import LOCAL_RESULTS_DIR, PROTOCOL_VERSION


def load_v2_results(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Not protocol v2: {path}")
    return obj


def trial_prediction(trial: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    true_label = trial.get("label")
    s2 = trial.get("stage2") or {}
    pred = s2.get("prediction")
    s1 = trial.get("stage1") or {}
    entropy = s1.get("semantic_entropy") if s1 else None
    return true_label, pred, entropy


def _collect_label_vocab(trials: List[Dict[str, Any]]) -> List[str]:
    """Union of true labels, predictions, and 4AFC options (fixes empty confusion when preds use variant strings)."""
    vocab: set = set()
    for t in trials:
        if t.get("label"):
            vocab.add(str(t["label"]))
        s2 = t.get("stage2") or {}
        if s2.get("prediction"):
            vocab.add(str(s2["prediction"]))
        for opt in s2.get("options") or []:
            vocab.add(str(opt))
    return sorted(vocab)


def build_confusion(trials: List[Dict[str, Any]], labels: List[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for t in trials:
        true_l, pred, _ = trial_prediction(t)
        if true_l not in idx or pred not in idx:
            continue
        mat[idx[true_l], idx[pred]] += 1
    return mat


def confused_pairs_from_trials(
    trials: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Direct (true, pred) counts when matrix indexing drops mismatched label strings."""
    counts: Counter = Counter()
    for t in trials:
        true_l, pred, _ = trial_prediction(t)
        if not true_l or not pred or true_l == pred:
            continue
        counts[(str(true_l), str(pred))] += 1
    pairs: List[Dict[str, Any]] = []
    for (a, b), count in counts.most_common(top_k):
        pairs.append({"label_a": a, "label_b": b, "count": int(count), "mean_entropy": None})
    return pairs


def confused_pairs_from_matrix(
    labels: List[str],
    mat: np.ndarray,
    entropy_by_label: Dict[str, List[float]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    pairs: List[Tuple[str, str, int]] = []
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(mat[i, j])
            if c > 0:
                pairs.append((labels[i], labels[j], c))
    pairs.sort(key=lambda x: -x[2])
    out: List[Dict[str, Any]] = []
    for a, b, count in pairs[:top_k]:
        ent_a = entropy_by_label.get(a, [])
        ent_b = entropy_by_label.get(b, [])
        mean_ent = float(np.mean(ent_a + ent_b)) if (ent_a or ent_b) else None
        out.append(
            {
                "label_a": a,
                "label_b": b,
                "count": count,
                "mean_entropy": mean_ent,
            }
        )
    return out


def run_error_analysis(results_path: Path, top_k: int = 5) -> Dict[str, Any]:
    obj = load_v2_results(results_path)
    trials = obj.get("trials", [])
    labels = _collect_label_vocab(trials)
    mat = build_confusion(trials, labels)

    per_label: Dict[str, Dict[str, Any]] = {}
    entropy_by_label: Dict[str, List[float]] = defaultdict(list)
    for t in trials:
        true_l, pred, ent = trial_prediction(t)
        if true_l is None:
            continue
        if ent is not None and ent == ent:
            entropy_by_label[true_l].append(float(ent))
        rec = per_label.setdefault(
            true_l,
            {"n": 0, "n_correct": 0, "entropies": []},
        )
        rec["n"] += 1
        s2 = t.get("stage2") or {}
        if s2.get("correct") is True:
            rec["n_correct"] += 1
        if ent is not None and ent == ent:
            rec["entropies"].append(float(ent))

    for lbl, rec in per_label.items():
        rec["accuracy"] = rec["n_correct"] / rec["n"] if rec["n"] else None
        ents = rec.pop("entropies", [])
        rec["mean_entropy"] = float(np.mean(ents)) if ents else None
        rec["median_entropy"] = float(np.median(ents)) if ents else None

    pairs = confused_pairs_from_matrix(labels, mat, entropy_by_label, top_k=top_k)
    if not pairs:
        pairs = confused_pairs_from_trials(trials, top_k=top_k)
    return {
        "model": obj.get("model"),
        "results_path": str(results_path),
        "protocol_version": PROTOCOL_VERSION,
        "per_label": per_label,
        "confused_pairs": pairs,
        "n_labels": len(labels),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Error analysis on protocol v2 evaluation JSONs.")
    ap.add_argument("--results", type=Path, required=True, help="Single eval_v2 JSON.")
    ap.add_argument(
        "--output_pairs",
        type=Path,
        default=None,
        help="Write confused_pairs JSON (default: results/stats/confused_pairs_{model}.json).",
    )
    ap.add_argument(
        "--output_summary",
        type=Path,
        default=LOCAL_RESULTS_DIR / "stats" / "error_analysis_summary.json",
    )
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    summary = run_error_analysis(args.results, top_k=args.top_k)
    model = summary.get("model", "unknown")
    if args.output_pairs is None:
        args.output_pairs = LOCAL_RESULTS_DIR / "stats" / f"confused_pairs_{model}.json"
    args.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    args.output_pairs.write_text(
        json.dumps({"confused_pairs": summary["confused_pairs"], "model": model}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_pairs} and {args.output_summary}")


if __name__ == "__main__":
    main()
