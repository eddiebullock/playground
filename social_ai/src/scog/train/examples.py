"""Persona prompts and answer targets for one Channel 4 respondent.

The cluster job writes the JSONL. Tests use synthetic frames only.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from scog.data.channel4 import DIAGNOSIS_COLS, ITEM_COLS
from scog.scoring.codebook import (
    DIAGNOSIS,
    EDUCATION,
    ITEM_CODING_INFERRED,
    OCCUPATION,
    REGION,
    SEX,
)
from scog.scoring.short_forms import AQ10_ITEMS, EQ10_ITEMS, SPQ10_ITEMS, SQR10_ITEMS

ITEM_TEXTS = tuple(SPQ10_ITEMS + EQ10_ITEMS + SQR10_ITEMS + AQ10_ITEMS)
if len(ITEM_TEXTS) != len(ITEM_COLS):
    raise RuntimeError("short-form wording count does not match Channel 4 item columns")

SHUFFLE_SEED = 20261029
INSTRUCTIONS = (
    "Answer the questionnaire as the person described below. "
    "For each item, use one of these labels: strongly agree, slightly agree, "
    "slightly disagree, strongly disagree. "
    "Keep the item numbers. Do not add any other text.\n\n"
)


def _code(value) -> int | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return int(number)


def _mapped(mapping: dict, value, field: str) -> str:
    code = _code(value)
    if code is None:
        return "missing"
    if code not in mapping:
        raise RuntimeError(f"unknown {field} code {code}")
    return mapping[code]


def is_autistic_woman(sex, primary_group: str) -> bool:
    return primary_group == "autistic" and _code(sex) == 2


def held_out_mask(frame: pd.DataFrame) -> pd.Series:
    sex = pd.to_numeric(frame["sex"], errors="coerce")
    return frame["primary_group"].astype(str).eq("autistic") & sex.eq(2)


def _sorted_ids(ids: Iterable[str]) -> list[str]:
    """Integer order for numeric respondent ids. Other ids sort as text."""
    values = [str(i) for i in ids]
    try:
        return sorted(values, key=lambda item: int(item))
    except ValueError:
        return sorted(values)


def dev_id_set(ids: Iterable[str]) -> set[str]:
    """About 10% of integer-sorted ids: positions 9, 19, 29, ..."""
    return {item for index, item in enumerate(_sorted_ids(ids)) if index % 10 == 9}


def shuffled_id_set(ids: Iterable[str]) -> set[str]:
    """Odd positions in the integer-sorted id list."""
    return {item for index, item in enumerate(_sorted_ids(ids)) if index % 2 == 1}


def item_permutation(example_id: str, n_items: int = 40) -> list[int]:
    digest = hashlib.sha256(f"{SHUFFLE_SEED}:{example_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    draw = np.random.default_rng(seed).permutation(n_items)
    return [int(i) for i in draw]


def format_persona(row: pd.Series) -> str:
    diagnoses = []
    saw_none = False
    for col in DIAGNOSIS_COLS:
        code = _code(row[col])
        if code is None:
            continue
        if code == 9:
            saw_none = True
            continue
        if code not in DIAGNOSIS:
            raise RuntimeError(f"unknown diagnosis code {code}")
        if code not in diagnoses:
            diagnoses.append(code)
    if diagnoses:
        diagnosis_text = ", ".join(DIAGNOSIS[code] for code in diagnoses)
    elif saw_none:
        diagnosis_text = DIAGNOSIS[9]
    else:
        diagnosis_text = "missing"
    age = _code(row["age"])
    age_text = "missing" if age is None else str(age)
    lines = [
        f"Age: {age_text}",
        f"Sex: {_mapped(SEX, row['sex'], 'sex')}",
        f"Education: {_mapped(EDUCATION, row['education'], 'education')}",
        f"Occupation: {_mapped(OCCUPATION, row['occupation'], 'occupation')}",
        f"Region: {_mapped(REGION, row['country_region'], 'region')}",
        f"Diagnoses: {diagnosis_text}",
    ]
    return "\n".join(lines)


def answer_labels(row: pd.Series) -> list[str] | None:
    labels = []
    for col in ITEM_COLS:
        code = _code(row[col])
        if code not in ITEM_CODING_INFERRED:
            return None
        labels.append(ITEM_CODING_INFERRED[code])
    return labels


def render_example(persona: str, order: list[int], labels: list[str]) -> tuple[str, str]:
    if len(order) != len(labels) or len(order) != len(ITEM_TEXTS):
        raise RuntimeError("example order and labels must cover every item")
    item_lines = [f"{number}. {ITEM_TEXTS[index]}" for number, index in enumerate(order, start=1)]
    answer_lines = [f"{number}. {label}" for number, label in enumerate(labels, start=1)]
    prompt = INSTRUCTIONS + persona + "\n\nItems:\n" + "\n".join(item_lines) + "\n\nAnswers:\n"
    answer = "\n".join(answer_lines) + "\n"
    return prompt, answer


def build_examples(frame: pd.DataFrame, ids: Iterable[str], *, shuffled: set[str]) -> tuple[list[dict], dict]:
    """One dict per complete respondent. Incomplete item rows are counted and skipped."""
    wanted = {str(i) for i in ids}
    if not wanted:
        raise RuntimeError("example build received zero ids")
    kept = []
    skipped = 0
    id_column = frame["userid"].map(lambda value: str(int(value)) if pd.notna(value) else "")
    for position in frame.index[id_column.isin(wanted)]:
        row = frame.loc[position]
        labels = answer_labels(row)
        example_id = str(int(row["userid"]))
        if labels is None:
            skipped += 1
            continue
        order = item_permutation(example_id) if example_id in shuffled else list(range(len(ITEM_TEXTS)))
        ordered_labels = [labels[index] for index in order]
        prompt, answer = render_example(format_persona(row), order, ordered_labels)
        kept.append({"id": example_id, "prompt": prompt, "answer": answer})
    if not kept:
        raise RuntimeError("example build kept zero rows")
    if skipped / (skipped + len(kept)) > 0.01:
        raise RuntimeError(f"example build skipped {skipped} incomplete rows, above 1%")
    return kept, {"n_kept": len(kept), "n_skipped_incomplete": skipped}
