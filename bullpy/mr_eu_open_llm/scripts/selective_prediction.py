"""Selective prediction and calibration metrics from semantic entropy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _binary_labels(correct: Sequence[bool]) -> np.ndarray:
    return np.asarray([1 if c else 0 for c in correct], dtype=np.float64)


def _confidence_from_entropy(entropies: Sequence[float], *, max_entropy: Optional[float] = None) -> np.ndarray:
    """Map semantic entropy to confidence in [0, 1] (lower entropy = higher confidence)."""
    ent = np.asarray(entropies, dtype=np.float64)
    if max_entropy is None:
        max_entropy = float(np.nanmax(ent)) if np.any(np.isfinite(ent)) else 1.0
    if max_entropy <= 0:
        return np.ones_like(ent)
    conf = 1.0 - np.clip(ent / max_entropy, 0.0, 1.0)
    return conf


def accuracy_on_subset(
    correct: Sequence[bool],
    mask: Sequence[bool],
) -> Tuple[Optional[float], int]:
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        return None, 0
    hits = sum(1 for i in idx if correct[i])
    return hits / len(idx), len(idx)


def auroc_entropy_vs_correct(
    entropies: Sequence[float],
    correct: Sequence[bool],
) -> Optional[float]:
    """AUROC using (1 - normalized entropy) as confidence score."""
    if len(entropies) != len(correct) or len(correct) < 2:
        return None
    if len(set(correct)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        conf = _confidence_from_entropy(entropies)
        return float(roc_auc_score(_binary_labels(correct), conf))
    except Exception:
        return None


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> Optional[float]:
    """Standard ECE with uniform confidence bins."""
    if len(confidences) != len(correct) or len(correct) == 0:
        return None
    conf = np.asarray(confidences, dtype=np.float64)
    acc = _binary_labels(correct)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)
    return float(ece)


def selective_prediction_report(
    trials: Sequence[Dict[str, Any]],
    *,
    entropy_key: str = "semantic_entropy",
    correct_key: str = "correct",
    low_entropy_quantile: float = 0.33,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Build calibration / selective-prediction summary from eval trial dicts.

    Each trial needs stage2.correct and stage1.semantic_entropy (or top-level keys).
    """
    entropies: List[float] = []
    correct: List[bool] = []
    for t in trials:
        s1 = t.get("stage1") or {}
        s2 = t.get("stage2") or {}
        ent = s1.get(entropy_key) if s1 else t.get(entropy_key)
        corr = s2.get(correct_key) if s2 else t.get(correct_key)
        if ent is None or corr is None:
            continue
        if ent != ent:  # NaN
            continue
        entropies.append(float(ent))
        correct.append(bool(corr))

    n = len(correct)
    overall_acc = sum(correct) / n if n else None
    if n == 0:
        return {
            "n_scored": 0,
            "overall_accuracy": None,
            "note": "no trials with entropy and stage2 correctness",
        }

    ent_arr = np.asarray(entropies)
    threshold = float(np.quantile(ent_arr, low_entropy_quantile))
    low_ent_mask = ent_arr <= threshold
    low_acc, low_n = accuracy_on_subset(correct, low_ent_mask)
    high_acc, high_n = accuracy_on_subset(correct, [not m for m in low_ent_mask])

    confidences = _confidence_from_entropy(entropies).tolist()
    return {
        "n_scored": n,
        "overall_accuracy": overall_acc,
        "low_entropy_quantile": low_entropy_quantile,
        "low_entropy_threshold": threshold,
        "low_entropy_subset_accuracy": low_acc,
        "low_entropy_subset_n": low_n,
        "high_entropy_subset_accuracy": high_acc,
        "high_entropy_subset_n": high_n,
        "auroc_confidence_vs_correct": auroc_entropy_vs_correct(entropies, correct),
        "expected_calibration_error": expected_calibration_error(confidences, correct, n_bins=n_bins),
        "mean_semantic_entropy": float(np.mean(ent_arr)),
        "median_semantic_entropy": float(np.median(ent_arr)),
    }


def format_selective_headline(report: Dict[str, Any]) -> str:
    """One-line summary for README / papers."""
    oa = report.get("overall_accuracy")
    la = report.get("low_entropy_subset_accuracy")
    ln = report.get("low_entropy_subset_n", 0)
    if oa is None or la is None:
        return "insufficient data for selective prediction"
    return (
        f"overall {oa:.1%} accurate; "
        f"low-entropy subset (n={ln}) {la:.1%} accurate"
    )
