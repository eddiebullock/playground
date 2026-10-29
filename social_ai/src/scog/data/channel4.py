"""Channel 4 raw export: load, exclusions, and group labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scog.scoring.codebook import AGREE_CODES, DISAGREE_CODES
from scog.scoring.short_forms import (
    AQ10_AGREE,
    AQ10_CUTOFF,
    EQ10_AGREE,
    SQR10_AGREE,
    binary_total,
)

DIAGNOSIS_COLS = [f"diagnosis_{i}" for i in range(9)]
SUBTYPE_COLS = [f"autism_diagnosis_{i}" for i in range(3)]
SPQ_COLS = [f"spq_{i}" for i in range(1, 11)]
EQ_COLS = [f"eq_{i}" for i in range(1, 11)]
SQR_COLS = [f"sqr_{i}" for i in range(1, 11)]
AQ_COLS = [f"aq_{i}" for i in range(1, 11)]
ITEM_COLS = SPQ_COLS + EQ_COLS + SQR_COLS + AQ_COLS
COLUMNS = (
    ["userid", "age", "sex", "handedness", "education", "occupation", "country_region", "repeat"]
    + DIAGNOSIS_COLS
    + SUBTYPE_COLS
    + ITEM_COLS
)
TEST_USER_IDS = set(range(174269, 174284))
SUBTYPE_CODES = {1, 2, 3}


def require_frame(df: pd.DataFrame, what: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise RuntimeError(f"{what}: zero rows")
    return df


def load_channel4(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Channel 4 file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    require_frame(df, f"Channel 4 file {path.name}")
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Channel 4 file {path.name} is missing columns: {missing}")
    if df.shape[1] < 60:
        raise RuntimeError(f"Channel 4 file {path.name} has {df.shape[1]} columns, expected 60")
    return df


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _filled_subtype(df: pd.DataFrame) -> pd.Series:
    filled = pd.Series(False, index=df.index)
    for col in SUBTYPE_COLS:
        values = _numeric(df[col])
        filled = filled | values.isin(SUBTYPE_CODES)
    return filled


def _empty(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    return values.isna()


def group_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label autistic, no-diagnosis, and ambiguous rows.

    Autistic: any diagnosis column equals 2.
    No diagnosis: diagnosis_0 equals 9 and every other diagnosis column is empty.
    Ambiguous: an ASD subtype in {1, 2, 3} but no diagnosis code 2.
    Ambiguous rows are in neither primary group.
    """
    diag = {col: _numeric(df[col]) for col in DIAGNOSIS_COLS}
    autistic = pd.Series(False, index=df.index)
    for values in diag.values():
        autistic = autistic | values.eq(2)
    others_empty = pd.Series(True, index=df.index)
    for col in DIAGNOSIS_COLS[1:]:
        others_empty = others_empty & _empty(df[col])
    no_diagnosis = diag["diagnosis_0"].eq(9) & others_empty & ~autistic
    ambiguous = _filled_subtype(df) & ~autistic
    out = pd.DataFrame(index=df.index)
    # A subtype without diagnosis code 2 cannot also be autistic.
    out["autistic"] = autistic
    out["no_diagnosis"] = no_diagnosis & ~ambiguous
    out["ambiguous_subtype"] = ambiguous
    out["primary_group"] = np.where(
        out["ambiguous_subtype"],
        "ambiguous",
        np.where(out["autistic"], "autistic", np.where(out["no_diagnosis"], "no_diagnosis", "other")),
    )
    return out


def clean_channel4(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the exclusions from the brief. Age 0 and age > 90 stay in, with age missing."""
    require_frame(df, "Channel 4 clean")
    work = df.copy()
    work["userid"] = _numeric(work["userid"])
    n_start = int(len(work))
    test_rows = work["userid"].isin(TEST_USER_IDS)
    work = work.loc[~test_rows].copy()
    repeat = _numeric(work["repeat"]).eq(1)
    n_repeat = int(repeat.sum())
    work = work.loc[~repeat].copy()
    items = work[ITEM_COLS].apply(pd.to_numeric, errors="coerce")
    identical = items.notna().all(axis=1) & items.nunique(axis=1).eq(1)
    n_identical = int(identical.sum())
    work = work.loc[~identical].copy()
    items = items.loc[work.index]
    age = _numeric(work["age"])
    n_age0 = int(age.eq(0).sum())
    n_age_over90 = int(age.gt(90).sum())
    age = age.mask(age.eq(0) | age.gt(90))
    work["age"] = age
    labels = group_labels(work)
    work = pd.concat([work, labels], axis=1)
    work["aq10_binary"] = binary_total(items[AQ_COLS], AQ10_AGREE).to_numpy()
    work["eq10_binary"] = binary_total(items[EQ_COLS], EQ10_AGREE).to_numpy()
    work["sqr10_binary"] = binary_total(items[SQR_COLS], SQR10_AGREE).to_numpy()
    # Reversed response codes (4 = strongly agree). Used only to test the inference.
    flipped = items.replace({1: 4, 2: 3, 3: 2, 4: 1})
    work["aq10_binary_flipped_coding"] = binary_total(flipped[AQ_COLS], AQ10_AGREE).to_numpy()
    if len(work) == 0:
        raise RuntimeError("Channel 4 cleaning removed every row")
    audit = {
        "n_start": n_start,
        "n_test_users_dropped": int(test_rows.sum()),
        "n_repeat_dropped": n_repeat,
        "n_identical_items_dropped": n_identical,
        "n_age0_set_missing": n_age0,
        "n_age_over90_set_missing": n_age_over90,
        "n_after": int(len(work)),
        "n_autistic": int(work["autistic"].sum()),
        "n_no_diagnosis": int(work["no_diagnosis"].sum()),
        "n_ambiguous": int(work["ambiguous_subtype"].sum()),
        "item_coding_status": "inferred",
        "agree_codes": list(AGREE_CODES),
        "disagree_codes": list(DISAGREE_CODES),
        "aq10_cutoff": AQ10_CUTOFF,
    }
    return work, audit


def aq_group_summary(cleaned: pd.DataFrame) -> dict:
    """AQ-10 binary means for the two primary groups. Cells are not printed."""
    def block(mask: pd.Series) -> dict:
        scores = cleaned.loc[mask, "aq10_binary"].dropna()
        n = int(scores.shape[0])
        if n == 0:
            raise RuntimeError("AQ summary: zero rows in a primary group")
        return {
            "n": n,
            "mean": float(scores.mean()),
            "sd": float(scores.std(ddof=1)),
            "pct_below_6": float((scores < AQ10_CUTOFF).mean() * 100),
            "pct_at_or_above_6": float((scores >= AQ10_CUTOFF).mean() * 100),
        }

    return {
        "autistic": block(cleaned["autistic"] & ~cleaned["ambiguous_subtype"]),
        "no_diagnosis": block(cleaned["no_diagnosis"] & ~cleaned["ambiguous_subtype"]),
    }
