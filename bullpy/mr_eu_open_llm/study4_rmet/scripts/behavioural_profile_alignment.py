"""
Phase 2 — Behavioural redesign: entropy alignment + profile-conditioned soft-label
alignment (B1/B2), plus demoted legacy H1.

Uses existing model eval JSONs (k samples). Prefer k>=20–50 on re-runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STUDY4_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from alignment_analyses import discover_model_evals, perm_spearman  # noqa: E402

DEFAULT_STRUCT = STUDY4_ROOT / "results" / "card_structure" / "item_card_structure.csv"
DEFAULT_CHOICE = STUDY4_ROOT / "results" / "card_structure" / "choice_distributions.json"
DEFAULT_HUMAN_TRAIT = STUDY4_ROOT / "results" / "human" / "item_trait_sensitivity.csv"
DEFAULT_OUT = STUDY4_ROOT / "results" / "behavioural_v2"
DEFAULT_A1 = STUDY4_ROOT / "results" / "alignment" / "a1_summary.json"


def _shannon_from_counts(counts: Dict[str, int], options: Sequence[str]) -> float:
    total = sum(counts.get(o, 0) for o in options)
    if total <= 0:
        return float("nan")
    h = 0.0
    for o in options:
        c = counts.get(o, 0)
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return float(h)


def _js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def model_soft_from_eval(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for t in data["trials"]:
        opts = list(t["options"])
        preds = [p for p in (t.get("samples") or {}).get("predictions") or [] if p is not None]
        counts = Counter(preds)
        dist = np.array([counts.get(o, 0) for o in opts], dtype=float)
        if dist.sum() > 0:
            dist = dist / dist.sum()
        ent = _shannon_from_counts(counts, opts)
        # also det
        det = t.get("deterministic") or {}
        rows.append(
            {
                "item": int(t["item"]),
                "model_entropy": ent,
                "sample_accuracy": (t.get("samples") or {}).get("accuracy"),
                "det_correct": int(bool(det.get("correct"))),
                "n_samples": len(preds),
                "soft_dist": dist.tolist(),
                "options": opts,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> Dict[str, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    rho, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        boots.append(r)
    boots = np.asarray(boots, float)
    return {
        "rho": float(rho),
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
        "n": float(n),
    }


def run_for_model(
    model: str,
    eval_path: Path,
    struct: pd.DataFrame,
    choice: Dict[str, Any],
    trait: pd.DataFrame,
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    mdf = model_soft_from_eval(eval_path)
    merged = struct.merge(mdf, on="item", how="inner")
    # B1 entropy
    b1 = bootstrap_spearman_ci(
        merged["human_entropy"].to_numpy(),
        merged["model_entropy"].to_numpy(),
        seed=seed,
    )
    b1_perm = perm_spearman(
        merged["human_entropy"].to_numpy(),
        merged["model_entropy"].to_numpy(),
        n_perm=n_perm,
        seed=seed,
    )

    # B2: mean JS to stratified human distributions
    js_targets = {
        "overall": "p_all",
        "eq_low": "p_eq_low",
        "eq_high": "p_eq_high",
        "asc": "p_asc",
        "non_asc": "p_non_asc",
    }
    js_means: Dict[str, float] = {}
    per_item_js: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        item = str(int(row["item"]))
        soft = np.asarray(row["soft_dist"], float)
        entry = {"item": int(item)}
        hum = choice["items"][item]
        for name, key in js_targets.items():
            hp = np.asarray(hum[key], float)
            if soft.size != hp.size or soft.sum() <= 0 or not np.isfinite(hp).all():
                val = float("nan")
            else:
                val = _js(soft, hp)
            entry[f"js_{name}"] = val
        per_item_js.append(entry)
    js_df = pd.DataFrame(per_item_js)
    for name in js_targets:
        js_means[f"mean_js_{name}"] = float(np.nanmean(js_df[f"js_{name}"]))

    # Profile preference: lower JS to low-EQ than high-EQ → "low-EQ-like"
    pref = float(np.nanmean(js_df["js_eq_low"] - js_df["js_eq_high"]))

    # Legacy H1: trait diagnosticity vs sample accuracy / det
    tmerge = trait.merge(mdf, on="item", how="inner")
    h1_acc = perm_spearman(
        tmerge["trait_sensitivity_coef"].to_numpy(),
        pd.to_numeric(tmerge["sample_accuracy"], errors="coerce").to_numpy(),
        n_perm=n_perm,
        seed=seed,
    )
    h1_det = perm_spearman(
        tmerge["trait_sensitivity_coef"].to_numpy(),
        tmerge["det_correct"].to_numpy(dtype=float),
        n_perm=n_perm,
        seed=seed,
    )

    return {
        "model": model,
        "eval_path": str(eval_path),
        "mean_n_samples": float(mdf["n_samples"].mean()),
        "B1_entropy_alignment": {**b1, "p_perm": b1_perm["p_perm"]},
        "B2_profile_js": {**js_means, "mean_js_eq_low_minus_high": pref},
        "legacy_H1_sample_accuracy": h1_acc,
        "legacy_H1_det_correct": h1_det,
        "per_item_js_table": js_df,
        "merged_item_table": merged[
            ["item", "human_entropy", "model_entropy", "sample_accuracy", "det_correct"]
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure_csv", type=Path, default=DEFAULT_STRUCT)
    ap.add_argument("--choice_json", type=Path, default=DEFAULT_CHOICE)
    ap.add_argument("--trait_csv", type=Path, default=DEFAULT_HUMAN_TRAIT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not args.structure_csv.exists():
        raise SystemExit(f"Run build_card_rmet_structure.py first ({args.structure_csv})")

    struct = pd.read_csv(args.structure_csv)
    choice = json.loads(args.choice_json.read_text(encoding="utf-8"))
    trait = pd.read_csv(args.trait_csv)
    evals = discover_model_evals(STUDY4_ROOT / "results" / "model")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"models": {}, "primary_tests": ["B1", "B2"], "legacy": ["H1"]}
    b1_rows, b2_rows, h1_rows = [], [], []

    for model, path in sorted(evals.items()):
        print(f"behavioural_v2: {model}", flush=True)
        res = run_for_model(
            model, path, struct, choice, trait, n_perm=args.n_perm, seed=args.seed
        )
        res["per_item_js_table"].to_csv(args.outdir / f"{model}_item_js.csv", index=False)
        res["merged_item_table"].to_csv(args.outdir / f"{model}_entropy_items.csv", index=False)
        slim = {k: v for k, v in res.items() if k not in ("per_item_js_table", "merged_item_table")}
        summary["models"][model] = slim
        b1 = res["B1_entropy_alignment"]
        b1_rows.append({"model": model, **{k: b1[k] for k in ("rho", "ci_low", "ci_high", "p_perm", "n")}})
        b2 = res["B2_profile_js"]
        b2_rows.append({"model": model, **b2})
        h1_rows.append(
            {
                "model": model,
                "H1_sample_rho": res["legacy_H1_sample_accuracy"]["rho"],
                "H1_sample_p_perm": res["legacy_H1_sample_accuracy"]["p_perm"],
                "H1_det_rho": res["legacy_H1_det_correct"]["rho"],
                "H1_det_p_perm": res["legacy_H1_det_correct"]["p_perm"],
            }
        )

    pd.DataFrame(b1_rows).to_csv(args.outdir / "B1_entropy_alignment.csv", index=False)
    pd.DataFrame(b2_rows).to_csv(args.outdir / "B2_profile_js_summary.csv", index=False)
    pd.DataFrame(h1_rows).to_csv(args.outdir / "legacy_H1_summary.csv", index=False)

    # Pass through prior A1 if present
    if DEFAULT_A1.exists():
        summary["legacy_a1_file"] = str(DEFAULT_A1)

    (args.outdir / "behavioural_v2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"B1": b1_rows, "B2_head": b2_rows[:2], "H1": h1_rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
