"""Draw adult personas from cells with n >= 10.

The caller passes one split. Counts and the draw stay inside that split.
Persona strings are the participant-facing labels in the A1 preregistration
appendix, not the shortened codebook forms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scog.data.channel4 import DIAGNOSIS_COLS, ITEM_COLS, SUBTYPE_COLS
from scog.data.splits import age_band
from scog.scoring.codebook import OCCUPATION

MIN_CELL = 10
AGE_TEXT = {
    "18_29": "18-29",
    "30_44": "30-44",
    "45_59": "45-59",
    "60_90": "60-90",
}

SEX = {1: "Male", 2: "Female", 3: "Other", 4: "Prefer not to say"}
EDUCATION = {
    1: "Did not complete High School (or A-levels)",
    2: "High School (or A-levels) Diploma",
    3: "Undergraduate degree",
    4: "Postgraduate degree",
    5: "Prefer not to say",
}
REGION = {
    1: "Wales",
    2: "Scotland",
    3: "Northern Ireland",
    4: "London (England)",
    5: "North East (England)",
    6: "North West (England)",
    7: "Yorkshire and Humber (England)",
    8: "West Midlands (England)",
    9: "East Midlands (England)",
    10: "South East (England)",
    11: "South West (England)",
    12: "Other (outside of the United Kingdom)",
    13: "Other (in the United Kingdom)",
    14: "Prefer not to say",
}
DIAGNOSIS = {
    1: "Attention Deficit / Hyperactivity Disorder",
    2: "Autism Spectrum Disorder",
    3: "Bipolar Disorder",
    4: "Depression",
    5: "Learning disability",
    6: "Obsessive-Compulsive Disorder",
    7: "Schizophrenia",
    8: "I prefer not to say",
    9: "I have not been diagnosed with any of these conditions",
}
ASD_SUBTYPE = {
    1: "Autism (classical autism)",
    2: "Asperger Syndrome (AS)",
    3: "Other",
}


def _code(value) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _label(mapping: dict[int, str], code: int | None, field: str) -> str | None:
    if code is None:
        return None
    if code not in mapping:
        raise RuntimeError(f"unmapped {field} code {code}")
    return mapping[code]


def _persona_from_row(row: pd.Series) -> dict:
    diagnoses = []
    for col in DIAGNOSIS_COLS:
        label = _label(DIAGNOSIS, _code(row[col]), col)
        if label is not None:
            diagnoses.append(label)
    detail = None
    for col in SUBTYPE_COLS:
        label = _label(ASD_SUBTYPE, _code(row[col]), col)
        if label is not None:
            detail = label
            break
    if pd.isna(row["age"]):
        band = "age_missing"
        shown = None
    else:
        band = age_band(float(row["age"]))
        shown = AGE_TEXT.get(band)
    sex = _label(SEX, _code(row["sex"]), "sex")
    education = _label(EDUCATION, _code(row["education"]), "education")
    occupation = _label(OCCUPATION, _code(row["occupation"]), "occupation")
    region = _label(REGION, _code(row["country_region"]), "region")
    return {
        "persona_id": str(row["userid"]),
        "group": str(row["group"]),
        "age_band": shown,
        "sex": sex,
        "education": education,
        "occupation": occupation,
        "region": region,
        "diagnoses": diagnoses,
        "autism_detail": detail,
        "cell": "|".join(
            [
                band,
                sex or "sex_missing",
                education or "edu_missing",
                occupation or "occ_missing",
                region or "region_missing",
            ]
        ),
    }


def eligible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Adults in the two primary groups, complete 40-item rows, cells n >= 10."""
    if len(frame) == 0:
        raise RuntimeError("persona sample: zero rows")
    work = frame.copy()
    autistic = work["autistic"].astype(bool) & ~work["ambiguous_subtype"].astype(bool)
    no_diagnosis = work["no_diagnosis"].astype(bool) & ~work["ambiguous_subtype"].astype(bool)
    work["group"] = np.where(autistic, "autistic", np.where(no_diagnosis, "no_diagnosis", "other"))
    items = work[ITEM_COLS].apply(pd.to_numeric, errors="coerce")
    complete = items.notna().all(axis=1) & items.isin([1, 2, 3, 4]).all(axis=1)
    age = pd.to_numeric(work["age"], errors="coerce")
    adult = age.isna() | (age >= 18)
    work = work.loc[complete & adult & work["group"].isin(["autistic", "no_diagnosis"])].copy()
    if len(work) == 0:
        raise RuntimeError("persona sample: zero eligible rows before the cell rule")
    if work["userid"].isna().any():
        raise RuntimeError("persona sample: an eligible row has no userid")
    work["userid"] = work["userid"].astype("int64").astype(str)
    personas = pd.DataFrame([_persona_from_row(row) for _, row in work.iterrows()])
    counts = personas.groupby(["group", "cell"]).size()
    keep = counts[counts >= MIN_CELL]
    if len(keep) == 0:
        raise RuntimeError("persona sample: every cell has n < 10")
    keep_index = set(keep.index)
    mask = [
        (group, cell) in keep_index
        for group, cell in zip(personas["group"], personas["cell"])
    ]
    kept = personas.loc[mask].copy()
    if len(kept) == 0:
        raise RuntimeError("persona sample: zero rows after the cell rule")
    return kept


def draw_ids(ids: list[str], n: int, seed: int) -> list[str]:
    if n < 1:
        raise ValueError("draw size must be positive")
    ordered = sorted(set(ids))
    if len(ordered) < n:
        raise RuntimeError(f"requested {n} personas but only {len(ordered)} are eligible")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(ordered))[:n]
    return [ordered[int(i)] for i in pick]


def draw_group(frame: pd.DataFrame, group: str, n: int, seed: int) -> pd.DataFrame:
    eligible = eligible_frame(frame)
    part = eligible.loc[eligible["group"] == group]
    if len(part) == 0:
        raise RuntimeError(f"persona sample: zero eligible {group} rows")
    chosen = set(draw_ids(part["persona_id"].tolist(), n, seed))
    out = part.loc[part["persona_id"].isin(chosen)].copy()
    if len(out) != n:
        raise RuntimeError(f"draw for {group} returned {len(out)}, expected {n}")
    return out
