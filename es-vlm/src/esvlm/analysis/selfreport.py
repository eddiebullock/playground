"""RQ3: standardised slopes of questionnaire log-prob vs task accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardised_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.polyfit(x, y, 1)[0])


def slopes_by_direction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, domain), part in df.groupby(["direction", "domain"]):
        if part["alpha"].nunique() < 3:
            continue
        acc = part.groupby("alpha")["correct"].mean()
        q = part.groupby("alpha")["questionnaire_score"].mean()
        shared = acc.index.intersection(q.index)
        rows.append(
            {
                "direction": direction,
                "domain": domain,
                "slope": standardised_slope(q.loc[shared].to_numpy(), acc.loc[shared].to_numpy()),
            }
        )
    return pd.DataFrame(rows)
