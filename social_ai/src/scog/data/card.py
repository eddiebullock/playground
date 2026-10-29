"""CARD long-format export.

Questionnaire totals must be rescored from Itemised Score. The published
AQ-50 / EQ-40 / SQ item-to-column map is not confirmed yet, so those
totals are not invented here. The Eyes Test parser follows the brief:
four header fields, then 36 (choice, other) pairs. The other value is ignored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scog.scoring.eyes import EYES_KEY, score_choices

COLUMNS = [
    "VolunteerID",
    "TestName",
    "Score",
    "LastModified",
    "YearOfBirth",
    "AgeWhenTestCompleted",
    "Ethnicity",
    "Occupation",
    "SchoolLeavingAge",
    "Country",
    "Sex",
    "Handedness",
    "ASC diagnosis",
    "Autism detail",
    "ASC suspected",
    "Additional diagnoses",
    "Itemised Score",
]


def load_card(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CARD file not found: {path}")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    if len(df) == 0:
        raise RuntimeError(f"CARD file {path.name}: zero rows")
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"CARD file {path.name} is missing columns: {missing}")
    return df


def normalise_country(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in {"", "0", "nan", "None", "NULL"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text or None


def normalise_sex(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in {"", "0", "nan", "None"}:
        return None
    key = text.lower()
    if key in {"f", "female"}:
        return "F"
    if key in {"m", "male"}:
        return "M"
    return text


def parse_eyes_itemised(raw) -> dict:
    """Parse one Eyes Test Itemised Score. Does not use the second value of each pair."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError("Eyes Test Itemised Score is missing")
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) < 4:
        raise ValueError("Eyes Test Itemised Score has fewer than 4 header fields")
    header = parts[:4]

    def _header_int(text: str):
        if text == "":
            return None
        return int(float(text))

    rest = parts[4:]
    if len(rest) < 72:
        rest = rest + [""] * (72 - len(rest))
    choices: list[int | None] = []
    for i in range(36):
        choice = rest[2 * i]
        if choice in {"", "-1"}:
            choices.append(None)
        else:
            try:
                choices.append(int(float(choice)))
            except ValueError as exc:
                raise ValueError(f"Eyes Test choice {i + 1} is not numeric") from exc
    scored = score_choices(choices, EYES_KEY)
    scored["choices"] = choices
    scored["header_n_correct"] = _header_int(header[1])
    scored["header_matches_key"] = scored["header_n_correct"] == scored["n_correct"]
    return scored


def prepare_card(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        raise RuntimeError("CARD prepare: zero rows")
    work = df.copy()
    work["Country_norm"] = work["Country"].map(normalise_country)
    work["Sex_norm"] = work["Sex"].map(normalise_sex)
    return work
