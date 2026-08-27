"""
Trial-level mixed-effects re-analysis of H1.

Fits (Gaussian LMM approximation to binary outcomes — same pragmatic choice as
study4 human Step-2 confirmatory LMM; pymer4/R and bambi/PyMC are not in the
active venv):

  correct ~ eq_sensitivity * C(agent_type)
            + (1 + eq_sensitivity | item)   # fall back to (1|item) if needed
            + subject/repetition clustered via variance component when possible

Human trials: CARD long format (VolunteerID × item).
Model trials: sampled completions (k≈10 per item) expanded to long format.

Data inventory (this repo):
  - Human trial-level: AVAILABLE (card_rmet_item_level.csv → to_long)
  - Model repeats: AVAILABLE (k=10 in full eval JSON; not 20–50; no log-probs)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .data_io import (
        load_human_item_sensitivity,
        load_human_trials,
        model_sample_matrix,
    )
except ImportError:
    from data_io import (  # type: ignore
        load_human_item_sensitivity,
        load_human_trials,
        model_sample_matrix,
    )


def build_human_trial_frame() -> pd.DataFrame:
    sens = load_human_item_sensitivity()[["item", "trait_sensitivity_coef"]].rename(
        columns={"trait_sensitivity_coef": "eq_sensitivity"}
    )
    # z-score item sensitivity for stable RE slopes
    mu, sd = sens["eq_sensitivity"].mean(), sens["eq_sensitivity"].std(ddof=0)
    sens["eq_sensitivity_z"] = (sens["eq_sensitivity"] - mu) / (sd if sd else 1.0)
    hum = load_human_trials()
    df = hum.merge(sens, on="item", how="inner")
    df["agent_type"] = "human"
    df["subject_or_rep"] = df["VolunteerID"].astype(str)
    df["correct"] = df["correct"].astype(float)
    return df[["item", "correct", "eq_sensitivity", "eq_sensitivity_z", "agent_type", "subject_or_rep"]]


def build_model_trial_frame(model: str) -> pd.DataFrame:
    mat, items, meta = model_sample_matrix(model)
    if mat.size == 0:
        return pd.DataFrame()
    sens = load_human_item_sensitivity()[["item", "trait_sensitivity_coef"]].rename(
        columns={"trait_sensitivity_coef": "eq_sensitivity"}
    )
    mu, sd = sens["eq_sensitivity"].mean(), sens["eq_sensitivity"].std(ddof=0)
    sens["eq_sensitivity_z"] = (sens["eq_sensitivity"] - mu) / (sd if sd else 1.0)
    rows = []
    for i, item in enumerate(items):
        for k in range(mat.shape[1]):
            if not np.isfinite(mat[i, k]):
                continue
            rows.append(
                {
                    "item": int(item),
                    "correct": float(mat[i, k]),
                    "agent_type": model,
                    "subject_or_rep": f"{model}_rep{k}",
                }
            )
    df = pd.DataFrame(rows).merge(sens, on="item", how="inner")
    df.attrs["model_meta"] = meta
    return df


def _fit_mixedlm(df: pd.DataFrame, formula: str, re_formula: str) -> Tuple[Any, str]:
    import statsmodels.formula.api as smf

    # Cluster by item; try maximal random slope of eq_sensitivity_z
    try:
        md = smf.mixedlm(formula, df, groups=df["item"], re_formula=re_formula)
        fit = md.fit(method="lbfgs", maxiter=200, reml=False)
        if not fit.converged:
            raise RuntimeError("not converged")
        return fit, re_formula
    except Exception as e:
        if re_formula.strip() != "1":
            md = smf.mixedlm(formula, df, groups=df["item"], re_formula="1")
            fit = md.fit(method="lbfgs", maxiter=200, reml=False)
            return fit, f"1  # fallback after: {type(e).__name__}: {e}"
        raise


def _interaction_rows(fit: Any, agent_levels: List[str]) -> List[Dict[str, Any]]:
    """Extract eq_sensitivity_z × agent interaction terms from params."""
    rows = []
    params = fit.params
    bse = fit.bse
    pvals = fit.pvalues
    for name in params.index:
        if "eq_sensitivity_z" in name and "agent_type" in name:
            rows.append(
                {
                    "term": name,
                    "coef": float(params[name]),
                    "se": float(bse[name]),
                    "p": float(pvals[name]),
                }
            )
    # also main effect of eq_sensitivity_z (human reference if Treatment coding)
    if "eq_sensitivity_z" in params.index:
        rows.insert(
            0,
            {
                "term": "eq_sensitivity_z",
                "coef": float(params["eq_sensitivity_z"]),
                "se": float(bse["eq_sensitivity_z"]),
                "p": float(pvals["eq_sensitivity_z"]),
            },
        )
    return rows


def fit_human_vs_model(model: str, max_human_trials: Optional[int] = 80_000, seed: int = 42) -> Dict[str, Any]:
    """One-vs-one: human vs model_X, Treatment contrast with human as reference."""
    hum = build_human_trial_frame()
    mod = build_model_trial_frame(model)
    if mod.empty:
        return {
            "status": "stub",
            "model": model,
            "message": (
                "Need model trial matrix (n_items × k) from "
                "results/model/<model>/rmet_eval_*_full_*.json samples.predictions"
            ),
        }
    if max_human_trials is not None and len(hum) > max_human_trials:
        hum = hum.sample(n=max_human_trials, random_state=seed)
    df = pd.concat([hum, mod], ignore_index=True)
    df["agent_type"] = pd.Categorical(df["agent_type"], categories=["human", model])
    formula = "correct ~ eq_sensitivity_z * C(agent_type)"
    fit, re_used = _fit_mixedlm(df, formula, re_formula="~eq_sensitivity_z")
    # BLUPs for item slopes if available
    blups = None
    try:
        re = fit.random_effects  # dict item -> array
        blup_rows = []
        for item, vec in re.items():
            v = np.asarray(vec, dtype=float).ravel()
            if v.size >= 2:
                blup_rows.append(
                    {"item": int(item), "blup_intercept": float(v[0]), "blup_slope": float(v[1])}
                )
            else:
                blup_rows.append(
                    {"item": int(item), "blup_intercept": float(v[0]), "blup_slope": np.nan}
                )
        blups = pd.DataFrame(blup_rows)
        sens = load_human_item_sensitivity()[["item", "trait_sensitivity_coef"]]
        blups = blups.merge(sens, on="item", how="left")
    except Exception:
        blups = None

    return {
        "status": "ok",
        "model": model,
        "backend": "statsmodels.MixedLM (Gaussian approx on binary)",
        "formula": formula,
        "re_formula_used": re_used,
        "n_human_trials": int(len(hum)),
        "n_model_trials": int(len(mod)),
        "n_items": int(df["item"].nunique()),
        "n_total_trials": int(len(df)),
        "converged": bool(getattr(fit, "converged", True)),
        "interaction_terms": _interaction_rows(fit, ["human", model]),
        "aic": float(fit.aic) if np.isfinite(getattr(fit, "aic", np.nan)) else None,
        "blups_head": blups.head(5).to_dict(orient="records") if blups is not None else None,
        "blups": blups,
        "power_contrast_note": (
            f"Item-level Spearman/Pearson tests use n_items≈36 (df≈34). "
            f"This trial-level LMM uses N_human={len(hum)} and N_model={len(mod)} "
            f"trials with item as grouping factor for random effects."
        ),
    }


def fit_omnibus(models: List[str], max_human_trials: Optional[int] = 60_000, seed: int = 42) -> Dict[str, Any]:
    """Single model with agent_type = human + all models; LRT vs no-interaction model."""
    import statsmodels.formula.api as smf

    hum = build_human_trial_frame()
    if max_human_trials is not None and len(hum) > max_human_trials:
        hum = hum.sample(n=max_human_trials, random_state=seed)
    parts = [hum]
    for m in models:
        mf = build_model_trial_frame(m)
        if not mf.empty:
            parts.append(mf)
    df = pd.concat(parts, ignore_index=True)
    levels = ["human"] + [m for m in models if m in set(df["agent_type"])]
    df["agent_type"] = pd.Categorical(df["agent_type"], categories=levels)
    f_full = "correct ~ eq_sensitivity_z * C(agent_type)"
    f_red = "correct ~ eq_sensitivity_z + C(agent_type)"
    fit_full, re_used = _fit_mixedlm(df, f_full, re_formula="1")  # omnibus: intercept RE for stability
    fit_red = smf.mixedlm(f_red, df, groups=df["item"], re_formula="1").fit(method="lbfgs", maxiter=200, reml=False)
    # Likelihood-ratio test (ML fits)
    lr = float(2 * (fit_full.llf - fit_red.llf))
    df_diff = int(max(1, len(fit_full.params) - len(fit_red.params)))
    from scipy.stats import chi2

    p_lrt = float(chi2.sf(lr, df_diff))
    return {
        "status": "ok",
        "backend": "statsmodels.MixedLM (Gaussian approx)",
        "formula_full": f_full,
        "formula_reduced": f_red,
        "re_formula_used": re_used,
        "n_human_trials": int((df.agent_type == "human").sum()),
        "n_model_trials": int((df.agent_type != "human").sum()),
        "n_total_trials": int(len(df)),
        "n_items": int(df["item"].nunique()),
        "agent_levels": levels,
        "interaction_terms": _interaction_rows(fit_full, levels),
        "lrt_interaction": {"LR": lr, "df": df_diff, "p": p_lrt},
        "power_contrast_note": (
            f"Item-level tests: n=36 items. Trial-level omnibus: "
            f"N={len(df)} trials (human={int((df.agent_type=='human').sum())}, "
            f"model={int((df.agent_type!='human').sum())})."
        ),
    }


def run_stub() -> Dict[str, Any]:
    msg = (
        "trial_level_model needs:\n"
        "  human: long table columns [VolunteerID, item, correct, eq_total] "
        "(from card_rmet_item_level.csv)\n"
        "  model: (n_items × k) correctness from samples.predictions with k>=2\n"
        "Optional upgrade: k=20–50 samples or log-prob of correct option for open models."
    )
    print(msg)
    return {"status": "stub", "message": msg}
