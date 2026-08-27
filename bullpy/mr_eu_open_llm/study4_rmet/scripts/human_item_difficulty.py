"""
Step 2 — Human RMET item-difficulty / trait-sensitivity profile (study4 only).

Inputs: data/processed/card_rmet_item_level.csv
Outputs under results/human/:
  item_trait_sensitivity.csv   36 rows: per-item logistic coefs (EQ, SQ, D) + CIs + p
  item_eq_tertile_accuracy.csv per-item accuracy by EQ tertile
  mixed_effects_summary.json   global mixed-effects confirmatory fit
  figures/eq_tertile_heatmap.png (optional)

Collinearity note: d_score = sq_total - eq_total, so EQ + SQ + D cannot enter one
design matrix jointly. Per-item models are fit separately for each trait predictor
(EQ, SQ, D) as univariate logistic regressions of item correctness. A confirmatory
mixed-effects model uses z_EQ + z_SQ with random intercepts for participant and item.

Does not import or modify study3 code.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

STUDY4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = STUDY4_ROOT / "data" / "processed" / "card_rmet_item_level.csv"
DEFAULT_OUTDIR = STUDY4_ROOT / "results" / "human"


def _item_correct_cols(df: pd.DataFrame) -> List[str]:
    cols = [f"rmet_{i:02d}_correct" for i in range(1, 37)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing correct columns: {missing[:5]}")
    return cols


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_vars = [
        "VolunteerID",
        "eq_total",
        "sq_total",
        "d_score",
        "aq_total",
        "spq_total",
        "age",
        "sex",
        "asc_diagnosis",
    ]
    id_vars = [c for c in id_vars if c in df.columns]
    correct_cols = _item_correct_cols(df)
    long = df.melt(
        id_vars=id_vars,
        value_vars=correct_cols,
        var_name="item_col",
        value_name="correct",
    )
    long["item"] = long["item_col"].str.extract(r"rmet_(\d+)_correct").astype(int)
    long["correct"] = pd.to_numeric(long["correct"], errors="coerce")
    long = long.dropna(subset=["correct"]).copy()
    long["correct"] = long["correct"].astype(int)
    return long


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd is None or not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def logistic_univariate(
    y: np.ndarray,
    x: np.ndarray,
) -> Dict[str, float]:
    """Intercept + one predictor logistic via statsmodels; returns coef/CI/p for x."""
    import statsmodels.api as sm

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    if len(y) < 30 or y.min() == y.max():
        return {
            "n": float(len(y)),
            "coef": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p": np.nan,
            "odds_ratio": np.nan,
        }
    X = sm.add_constant(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.Logit(y, X).fit(disp=False, maxiter=200)
        except Exception:
            fit = sm.Logit(y, X).fit_regularized(disp=False, alpha=1e-4)
            # Regularized fit lacks standard errors — mark p/CI missing
            coef = float(fit.params[1])
            return {
                "n": float(len(y)),
                "coef": coef,
                "se": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p": np.nan,
                "odds_ratio": float(np.exp(coef)),
                "fit": "regularized",
            }
    coef = float(fit.params[1])
    se = float(fit.bse[1])
    ci = fit.conf_int(alpha=0.05)
    return {
        "n": float(len(y)),
        "coef": coef,
        "se": se,
        "ci_low": float(ci[1, 0]),
        "ci_high": float(ci[1, 1]),
        "p": float(fit.pvalues[1]),
        "odds_ratio": float(np.exp(coef)),
        "fit": "mle",
    }


def per_item_trait_sensitivity(long: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    work = long.copy()
    work["z_eq"] = _zscore(work["eq_total"])
    work["z_sq"] = _zscore(work["sq_total"])
    work["z_d"] = _zscore(work["d_score"])

    for item in range(1, 37):
        sub = work[work["item"] == item]
        y = sub["correct"].to_numpy()
        row: Dict[str, Any] = {
            "item": item,
            "n": int(np.isfinite(y).sum()),
            "accuracy": float(np.nanmean(y)),
        }
        for trait, col in (("eq", "z_eq"), ("sq", "z_sq"), ("d", "z_d")):
            stats_d = logistic_univariate(y, sub[col].to_numpy())
            for k, v in stats_d.items():
                row[f"{trait}_{k}"] = v
        # Primary "trait-sensitivity" headline = EQ slope (continuous empathizing).
        row["trait_sensitivity_coef"] = row["eq_coef"]
        row["trait_sensitivity_ci_low"] = row["eq_ci_low"]
        row["trait_sensitivity_ci_high"] = row["eq_ci_high"]
        row["trait_sensitivity_p"] = row["eq_p"]
        rows.append(row)
    return pd.DataFrame(rows)


def eq_tertile_accuracy(df_wide: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    eq = pd.to_numeric(df_wide["eq_total"], errors="coerce")
    # Tertiles on participants with EQ; label low/mid/high.
    valid = df_wide.loc[eq.notna(), ["VolunteerID", "eq_total"]].copy()
    valid["eq_tertile"] = pd.qcut(
        pd.to_numeric(valid["eq_total"], errors="coerce"),
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )
    merged = long.merge(valid[["VolunteerID", "eq_tertile"]], on="VolunteerID", how="inner")
    rows = []
    for item in range(1, 37):
        sub = merged[merged["item"] == item]
        row: Dict[str, Any] = {"item": item}
        for t in ("low", "mid", "high"):
            s = sub.loc[sub["eq_tertile"] == t, "correct"]
            row[f"acc_{t}"] = float(s.mean()) if len(s) else np.nan
            row[f"n_{t}"] = int(len(s))
        row["acc_high_minus_low"] = row["acc_high"] - row["acc_low"]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_mixed_effects(long: pd.DataFrame) -> Dict[str, Any]:
    """
    Confirmatory mixed model on a participant-level accuracy summary is too coarse;
    a full crossed logistic with C(VolunteerID) dummies is too large for local VB.

    Here we fit a linear mixed model on the long binary outcomes as an approximate
    confirmatory check: correct ~ z_eq + z_sq + (1|VolunteerID). Item variance is
    absorbed by fitting after residualizing item means (within-item centered outcome
    would remove the trait signal), so instead we include item fixed effects for a
    random sample of participants to keep the matrix tractable.

    Primary scientific output remains the per-item logistic table.
    """
    import statsmodels.formula.api as smf

    work = long.dropna(subset=["eq_total", "sq_total", "correct"]).copy()
    work["z_eq"] = _zscore(work["eq_total"])
    work["z_sq"] = _zscore(work["sq_total"])
    work["VolunteerID"] = work["VolunteerID"].astype(str)
    work["item"] = work["item"].astype(int)

    # Subsample participants for tractable crossed FE approximation
    rng = np.random.default_rng(42)
    pids = work["VolunteerID"].unique()
    if len(pids) > 800:
        keep = set(rng.choice(pids, size=800, replace=False))
        work = work[work["VolunteerID"].isin(keep)].copy()

    try:
        # Random intercept for participant; item as fixed effect (C(item))
        model = smf.mixedlm(
            "correct ~ z_eq + z_sq + C(item)",
            data=work,
            groups=work["VolunteerID"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(reml=False, method="lbfgs")
        fixed = {}
        for name in ("z_eq", "z_sq", "Intercept"):
            if name in result.fe_params.index:
                mu = float(result.fe_params[name])
                se = float(result.bse_fe[name])
                z = mu / se if se > 0 else np.nan
                p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
                fixed[name] = {"coef": mu, "se": se, "approx_p": p}
        return {
            "status": "ok",
            "formula": "correct ~ z_eq + z_sq + C(item) + (1|VolunteerID)",
            "method": "statsmodels.mixedlm (Gaussian LMM on binary; approximate)",
            "n_rows": int(len(work)),
            "n_participants": int(work["VolunteerID"].nunique()),
            "n_items": int(work["item"].nunique()),
            "fixed_effects": fixed,
            "note": (
                "Primary per-item trait-sensitivity is in item_trait_sensitivity.csv. "
                "This LMM is a global confirmatory check only; D omitted (collinear with EQ+SQ). "
                "Participants subsampled to 800 if N is larger."
            ),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": repr(e),
            "n_rows": int(len(work)),
            "fallback": "Use per-item univariate logistic tables as primary Step-2 output.",
        }


def maybe_plot_tertiles(tert: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = tert["item"].to_numpy()
    ax.plot(x, tert["acc_low"], label="EQ low tertile", marker="o", ms=3)
    ax.plot(x, tert["acc_mid"], label="EQ mid tertile", marker="o", ms=3)
    ax.plot(x, tert["acc_high"], label="EQ high tertile", marker="o", ms=3)
    ax.set_xlabel("RMET item")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Human RMET item accuracy by EQ tertile")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(input_csv: Path, outdir: Path, *, fit_mixed: bool = True) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    # Require EQ+SQ for trait analyses
    n_all = len(df)
    df_trait = df.dropna(subset=["eq_total", "sq_total"]).copy()
    long = to_long(df_trait)

    print(f"Fitting per-item logistics on N={len(df_trait)} participants...", flush=True)
    sens = per_item_trait_sensitivity(long)
    tert = eq_tertile_accuracy(df_trait, long)
    if fit_mixed:
        print("Fitting confirmatory mixed model (subsampled)...", flush=True)
        mixed = fit_mixed_effects(long)
    else:
        mixed = {"status": "skipped"}

    sens_path = outdir / "item_trait_sensitivity.csv"
    tert_path = outdir / "item_eq_tertile_accuracy.csv"
    mixed_path = outdir / "mixed_effects_summary.json"
    sens.to_csv(sens_path, index=False)
    tert.to_csv(tert_path, index=False)
    mixed_path.write_text(json.dumps(mixed, indent=2) + "\n", encoding="utf-8")

    fig_path = outdir / "figures" / "eq_tertile_accuracy.png"
    maybe_plot_tertiles(tert, fig_path)

    summary = {
        "n_participants_raw": n_all,
        "n_participants_with_eq_sq": int(len(df_trait)),
        "n_long_rows": int(len(long)),
        "mean_accuracy": float(long["correct"].mean()),
        "outputs": {
            "item_trait_sensitivity": str(sens_path),
            "item_eq_tertile_accuracy": str(tert_path),
            "mixed_effects_summary": str(mixed_path),
            "figure": str(fig_path) if fig_path.exists() else None,
        },
        "mixed_effects_status": mixed.get("status"),
        "top_eq_sensitive_items": sens.nlargest(5, "eq_coef")[
            ["item", "accuracy", "eq_coef", "eq_p"]
        ].to_dict(orient="records"),
        "bottom_eq_sensitive_items": sens.nsmallest(5, "eq_coef")[
            ["item", "accuracy", "eq_coef", "eq_p"]
        ].to_dict(orient="records"),
    }
    (outdir / "step2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--no_mixed", action="store_true", help="Skip confirmatory mixed model")
    args = ap.parse_args(argv)
    summary = run(args.input, args.outdir, fit_mixed=not args.no_mixed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
