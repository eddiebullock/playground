"""
Phase 1 — Build CARD RMET psychometric structure (study4 only).

From card_rmet_item_level.csv + answer key + existing EQ slopes:
  - per-item choice distributions (overall, EQ tertile, ASC vs non-ASC)
  - human Shannon entropy
  - trait_diagnosticity (EQ slope; ASC gap; low-high EQ accuracy gap)
  - profile-conditioned JS/KL divergences
  - RSA-ready feature matrix (NO global label RDM — options are item-specific)
  - pre-registered item classes (median splits)

Outputs under results/card_structure/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STUDY4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD = STUDY4_ROOT / "data" / "processed" / "card_rmet_item_level.csv"
DEFAULT_KEY = STUDY4_ROOT / "data" / "rmet" / "answer_key" / "rmet_adult_answer_key.json"
DEFAULT_HUMAN = STUDY4_ROOT / "results" / "human" / "item_trait_sensitivity.csv"
DEFAULT_OUT = STUDY4_ROOT / "results" / "card_structure"


def _shannon(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p)))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def load_key(path: Path) -> Dict[int, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {int(it["item"]): it for it in obj["items"]}


def choice_dist(series: pd.Series, n_opt: int = 4) -> np.ndarray:
    """Distribution over options 1..n_opt; ignores -1/NaN."""
    counts = np.zeros(n_opt, dtype=float)
    for v in series.dropna():
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n_opt:
            counts[i - 1] += 1.0
    s = counts.sum()
    if s <= 0:
        return counts
    return counts / s


def eq_tertiles(eq: pd.Series) -> pd.Series:
    x = pd.to_numeric(eq, errors="coerce")
    # tertile labels: 0=low, 1=mid, 2=high
    try:
        return pd.qcut(x, 3, labels=[0, 1, 2], duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=eq.index)


def build_structure(
    card: pd.DataFrame,
    key: Dict[int, Dict[str, Any]],
    trait_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    trait = pd.read_csv(trait_csv).sort_values("item")
    eq_slope = dict(zip(trait["item"].astype(int), trait["trait_sensitivity_coef"].astype(float)))

    work = card.copy()
    work["eq_total"] = pd.to_numeric(work["eq_total"], errors="coerce")
    work["asc_diagnosis"] = pd.to_numeric(work.get("asc_diagnosis"), errors="coerce")
    work["eq_tertile"] = eq_tertiles(work["eq_total"])

    rows: List[Dict[str, Any]] = []
    choice_payload: Dict[str, Any] = {"items": {}}

    for item in range(1, 37):
        ch_col = f"rmet_{item:02d}_choice"
        corr_col = f"rmet_{item:02d}_correct"
        if ch_col not in work.columns:
            raise KeyError(ch_col)
        opts = list(key[item]["options"])
        n_opt = len(opts)
        correct_opt = int(key[item]["correct_option"])

        sub = work[[ch_col, corr_col, "eq_total", "eq_tertile", "asc_diagnosis"]].copy()
        sub = sub.rename(columns={ch_col: "choice", corr_col: "correct"})
        sub["correct"] = pd.to_numeric(sub["correct"], errors="coerce")

        p_all = choice_dist(sub["choice"], n_opt=n_opt)
        ent = _shannon(p_all)

        # EQ tertile dists + accuracy
        p_low = p_mid = p_high = np.full(n_opt, np.nan)
        acc_low = acc_mid = acc_high = float("nan")
        for t_lab, name in [(0, "low"), (1, "mid"), (2, "high")]:
            g = sub[sub["eq_tertile"] == t_lab]
            dist = choice_dist(g["choice"], n_opt=n_opt)
            acc = float(pd.to_numeric(g["correct"], errors="coerce").mean()) if len(g) else float("nan")
            if name == "low":
                p_low, acc_low = dist, acc
            elif name == "mid":
                p_mid, acc_mid = dist, acc
            else:
                p_high, acc_high = dist, acc

        # ASC vs non-ASC
        asc = sub[sub["asc_diagnosis"] == 1.0]
        ctrl = sub[sub["asc_diagnosis"] == 0.0]
        p_asc = choice_dist(asc["choice"], n_opt=n_opt)
        p_ctrl = choice_dist(ctrl["choice"], n_opt=n_opt)
        acc_asc = float(pd.to_numeric(asc["correct"], errors="coerce").mean()) if len(asc) else float("nan")
        acc_ctrl = float(pd.to_numeric(ctrl["correct"], errors="coerce").mean()) if len(ctrl) else float("nan")
        asc_gap = acc_asc - acc_ctrl if np.isfinite(acc_asc) and np.isfinite(acc_ctrl) else float("nan")
        eq_gap = acc_low - acc_high if np.isfinite(acc_low) and np.isfinite(acc_high) else float("nan")

        js_eq = _js_divergence(p_low, p_high) if np.isfinite(p_low).all() else float("nan")
        kl_eq = _kl(p_low, p_high) if np.isfinite(p_low).all() else float("nan")
        js_asc = _js_divergence(p_asc, p_ctrl) if np.isfinite(p_asc).all() else float("nan")
        kl_asc = _kl(p_asc, p_ctrl) if np.isfinite(p_asc).all() else float("nan")

        diag = float(eq_slope.get(item, float("nan")))
        acc_all = float(pd.to_numeric(sub["correct"], errors="coerce").mean())

        rows.append(
            {
                "item": item,
                "correct_label": key[item]["correct_label"],
                "correct_option": correct_opt,
                "n_valid_choices": int((sub["choice"].isin([1, 2, 3, 4])).sum()),
                "accuracy_all": acc_all,
                "human_entropy": ent,
                "trait_diagnosticity_eq_slope": diag,
                "asc_accuracy_gap": asc_gap,
                "eq_low_minus_high_accuracy_gap": eq_gap,
                "accuracy_eq_low": acc_low,
                "accuracy_eq_mid": acc_mid,
                "accuracy_eq_high": acc_high,
                "accuracy_asc": acc_asc,
                "accuracy_non_asc": acc_ctrl,
                "js_lowEQ_vs_highEQ": js_eq,
                "kl_lowEQ_vs_highEQ": kl_eq,
                "js_ASC_vs_nonASC": js_asc,
                "kl_ASC_vs_nonASC": kl_asc,
                "n_asc": int(len(asc)),
                "n_non_asc": int(len(ctrl)),
            }
        )
        choice_payload["items"][str(item)] = {
            "options": opts,
            "correct_option": correct_opt,
            "p_all": p_all.tolist(),
            "p_eq_low": p_low.tolist(),
            "p_eq_mid": p_mid.tolist(),
            "p_eq_high": p_high.tolist(),
            "p_asc": p_asc.tolist(),
            "p_non_asc": p_ctrl.tolist(),
        }

    item_df = pd.DataFrame(rows).sort_values("item").reset_index(drop=True)

    # Pre-registered median-split classes (human data only)
    def median_class(col: str, high_name: str, low_name: str) -> pd.Series:
        v = item_df[col].astype(float)
        med = float(np.nanmedian(v.to_numpy()))
        return np.where(v >= med, high_name, low_name)

    item_df["class_trait_diagnosticity"] = median_class(
        "trait_diagnosticity_eq_slope", "high_diagnosticity", "low_diagnosticity"
    )
    item_df["class_human_entropy"] = median_class(
        "human_entropy", "high_entropy", "low_entropy"
    )
    # Secondary ASC-gap class (report; primary remains EQ slope)
    item_df["class_asc_gap"] = median_class(
        "asc_accuracy_gap", "high_asc_gap_mag_pos", "low_asc_gap"
    )

    # Feature matrix for RSA: continuous item features (NOT label RDMs)
    feat_cols = [
        "human_entropy",
        "trait_diagnosticity_eq_slope",
        "asc_accuracy_gap",
        "eq_low_minus_high_accuracy_gap",
        "js_lowEQ_vs_highEQ",
        "js_ASC_vs_nonASC",
        "accuracy_all",
    ]
    feat = item_df[["item"] + feat_cols].copy()
    note = {
        "why_no_global_label_rdm": (
            "RMET foils are item-specific; a shared mental-state label RDM is invalid. "
            "RSA uses comparable-across-items continuous features (entropy, diagnosticity, "
            "stratified divergence), not cross-item label confusion."
        ),
        "feature_columns": feat_cols,
        "class_splits": {
            "trait_diagnosticity": "median of trait_diagnosticity_eq_slope",
            "human_entropy": "median of human_entropy",
            "asc_gap_secondary": "median of asc_accuracy_gap",
        },
        "n_participants": int(len(card)),
        "n_asc": int((pd.to_numeric(card.get("asc_diagnosis"), errors="coerce") == 1).sum()),
        "alexithymia": "not available in CARD export — ASC contrasts limited",
    }
    return item_df, choice_payload, {"features_note": note, "feature_columns": feat_cols}


def rdm_from_feature_matrix(feat_df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Pairwise cosine distance RDM from z-scored feature rows."""
    X = feat_df[list(cols)].to_numpy(dtype=float)
    # z-score columns
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    sim = Xn @ Xn.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--card_csv", type=Path, default=DEFAULT_CARD)
    ap.add_argument("--answer_key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--trait_csv", type=Path, default=DEFAULT_HUMAN)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.card_csv.exists():
        raise SystemExit(f"Missing CARD CSV: {args.card_csv}")
    if not args.trait_csv.exists():
        raise SystemExit(
            f"Missing trait sensitivity CSV: {args.trait_csv} "
            "(run scripts/human_item_difficulty.py first)"
        )

    card = pd.read_csv(args.card_csv)
    key = load_key(args.answer_key)
    item_df, choice_payload, meta = build_structure(card, key, args.trait_csv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    item_path = args.outdir / "item_card_structure.csv"
    item_df.to_csv(item_path, index=False)

    feat_cols = meta["feature_columns"]
    feat_path = args.outdir / "item_feature_matrix.csv"
    item_df[["item"] + feat_cols].to_csv(feat_path, index=False)

    rdm = rdm_from_feature_matrix(item_df, feat_cols)
    np.save(args.outdir / "human_card_feature_rdm.npy", rdm)

    # Scalar RDMs for single-feature RSA targets
    for col, short in (
        ("human_entropy", "entropy"),
        ("trait_diagnosticity_eq_slope", "diagnosticity"),
    ):
        v = item_df[col].to_numpy(dtype=float).reshape(-1, 1)
        d = np.abs(v - v.T).astype(np.float32)
        np.fill_diagonal(d, 0.0)
        np.save(args.outdir / f"human_{short}_rdm.npy", d)

    classes = {
        "trait_diagnosticity": {
            "high": item_df.loc[
                item_df["class_trait_diagnosticity"] == "high_diagnosticity", "item"
            ].astype(int).tolist(),
            "low": item_df.loc[
                item_df["class_trait_diagnosticity"] == "low_diagnosticity", "item"
            ].astype(int).tolist(),
            "rule": "median split on trait_diagnosticity_eq_slope",
        },
        "human_entropy": {
            "high": item_df.loc[
                item_df["class_human_entropy"] == "high_entropy", "item"
            ].astype(int).tolist(),
            "low": item_df.loc[
                item_df["class_human_entropy"] == "low_entropy", "item"
            ].astype(int).tolist(),
            "rule": "median split on human_entropy",
        },
    }
    (args.outdir / "item_classes_preregistered.json").write_text(
        json.dumps(classes, indent=2) + "\n", encoding="utf-8"
    )
    (args.outdir / "choice_distributions.json").write_text(
        json.dumps(choice_payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "n_items": 36,
        "n_participants": int(len(card)),
        "outputs": {
            "item_card_structure": str(item_path),
            "item_feature_matrix": str(feat_path),
            "human_card_feature_rdm": str(args.outdir / "human_card_feature_rdm.npy"),
            "item_classes": str(args.outdir / "item_classes_preregistered.json"),
            "choice_distributions": str(args.outdir / "choice_distributions.json"),
        },
        "meta": meta,
        "mean_human_entropy": float(item_df["human_entropy"].mean()),
        "mean_trait_diagnosticity": float(item_df["trait_diagnosticity_eq_slope"].mean()),
    }
    (args.outdir / "card_structure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
