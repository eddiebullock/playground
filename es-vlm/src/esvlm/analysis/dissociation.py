"""RQ2: direction x domain x alpha mixed-effects (item + template RE)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def fit_dissociation(df: pd.DataFrame) -> Any:
    """``correct ~ direction * domain * alpha + (1|item) + (1|template)``.

    Requires statsmodels. Item-bootstrap CIs are computed separately.
    Always also report generic-VQA accuracy and perplexity per alpha.
    """

    import statsmodels.formula.api as smf

    needed = {"correct", "direction", "domain", "alpha", "item_id", "template_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    model = smf.mixedlm(
        "correct ~ direction * domain * alpha",
        df,
        groups=df["item_id"],
        re_formula="1",
        vc_formula={"template": "0 + C(template_id)"},
    )
    return model.fit(method="lbfgs")


def bootstrap_item_ci(
    df: pd.DataFrame,
    *,
    n: int = 1000,
    seed: int = 0,
    col: str = "correct",
) -> dict[str, float]:
    rng = __import__("numpy").random.default_rng(seed)
    items = df["item_id"].unique()
    means = []
    for _ in range(n):
        draw = rng.choice(items, size=len(items), replace=True)
        part = df.set_index("item_id").loc[draw]
        means.append(float(part[col].mean()))
    arr = __import__("numpy").asarray(means)
    return {
        "mean": float(arr.mean()),
        "lo": float(__import__("numpy").quantile(arr, 0.025)),
        "hi": float(__import__("numpy").quantile(arr, 0.975)),
    }
