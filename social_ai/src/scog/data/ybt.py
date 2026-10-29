"""YourBrainType Qualtrics export.

Row 1 is variable names, row 2 is question text, row 3 is ImportId JSON.
Those two rows are skipped. Two columns are both named Q311; they are renamed
from the question text when it is available, and otherwise to Q311_a / Q311_b.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

TEXT_TO_CODE = {
    "strongly agree": 1,
    "slightly agree": 2,
    "slightly disagree": 3,
    "strongly disagree": 4,
}


def _read_preamble(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
            questions = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"YourBrainType file {path.name} has no header rows") from exc
    if not any(cell.strip() for cell in header):
        raise RuntimeError(f"YourBrainType file {path.name} has an empty header")
    return header, questions


def _rename_duplicate_q311(header: list[str], questions: list[str]) -> list[str]:
    names = list(header)
    indexes = [i for i, name in enumerate(names) if name == "Q311"]
    if len(indexes) < 2:
        return names
    for n, idx in enumerate(indexes):
        question = questions[idx].strip() if idx < len(questions) else ""
        if question:
            names[idx] = f"Q311__{question}"
        else:
            names[idx] = f"Q311_{'ab'[n] if n < 2 else n}"
    return names


def load_ybt(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YourBrainType file not found: {path}")
    header, questions = _read_preamble(path)
    names = _rename_duplicate_q311(header, questions)
    if len(names) != len(set(names)):
        raise RuntimeError("YourBrainType header still has duplicate names after the Q311 rename")
    df = pd.read_csv(path, skiprows=[1, 2], header=0, low_memory=False)
    if len(df) == 0:
        raise RuntimeError(f"YourBrainType file {path.name}: zero rows")
    if len(df.columns) != len(names):
        raise RuntimeError(
            f"YourBrainType column count {len(df.columns)} != header count {len(names)}"
        )
    df.columns = names
    if "Finished" not in df.columns:
        raise RuntimeError("YourBrainType file has no Finished column")
    return df, questions


def code_likert(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    coded = text.map(TEXT_TO_CODE)
    missing = series.isna() | text.isin({"", "nan", "none"})
    coded = coded.mask(missing)
    unknown = ~missing & coded.isna() & series.notna()
    if int(unknown.sum()) > 0:
        raise RuntimeError(
            f"YourBrainType likert column has {int(unknown.sum())} values outside the four labels"
        )
    return coded


def parse_multiselect(value, options: list[str]) -> list[str]:
    """Match known option labels, longest first, so commas inside labels survive."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return []
    remaining = text
    found: list[str] = []
    for option in sorted(options, key=len, reverse=True):
        if option and option in remaining:
            found.append(option)
            remaining = remaining.replace(option, "", 1)
    leftover = remaining.replace(",", " ").strip()
    if leftover:
        raise ValueError(f"unparsed diagnosis text remains: {leftover[:80]}")
    return found
