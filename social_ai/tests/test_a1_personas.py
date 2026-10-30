"""Synthetic checks for A1 prompts, parsing, and the cell rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scog.a1.parse import parse_completion
from scog.a1.prompts import channel4_items, user_message
from scog.a1.sample import draw_group, draw_ids
from scog.data.channel4 import DIAGNOSIS_COLS, ITEM_COLS, SUBTYPE_COLS


def _people(n: int, *, group: str, age: float = 30, sex: int = 2, item: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {col: np.nan for col in DIAGNOSIS_COLS + SUBTYPE_COLS}
        row.update({col: item for col in ITEM_COLS})
        row["userid"] = 1000 + i + (0 if group == "autistic" else 5000)
        row["age"] = age
        row["sex"] = sex
        row["education"] = 3
        row["occupation"] = 9
        row["country_region"] = 4
        row["autistic"] = group == "autistic"
        row["no_diagnosis"] = group == "no_diagnosis"
        row["ambiguous_subtype"] = False
        if group == "autistic":
            row["diagnosis_0"] = 2
            row["autism_diagnosis_0"] = 2
        else:
            row["diagnosis_0"] = 9
        rows.append(row)
    return pd.DataFrame(rows)


def test_prompt_is_forty_items_and_condition_3_adds_the_diagnosis():
    assert len(channel4_items()) == 40
    persona = {
        "group": "autistic",
        "age_band": "30-44",
        "sex": "Female",
        "education": "Undergraduate degree",
        "occupation": "Healthcare",
        "region": "London (England)",
        "diagnoses": ["Autism Spectrum Disorder"],
        "autism_detail": "Asperger Syndrome (AS)",
    }
    demo = user_message(persona, 2)
    full = user_message(persona, 3)
    assert "Diagnoses:" not in demo
    assert "Autism Spectrum Disorder" in full
    assert "Asperger Syndrome (AS)" in full
    assert "1. " in full and "40. " in full
    assert "Answer as yourself." in user_message(None, 1)
    assert user_message({"group": "autistic"}, 4) == user_message({"group": "autistic"}, 4)
    assert "You are an autistic adult." in user_message({"group": "autistic"}, 4)


def test_parser_keeps_refusals_and_rejects_extra_words():
    lines = [f"{i}. strongly agree" for i in range(1, 41)]
    parsed = parse_completion("\n".join(lines))
    assert parsed["status"] == "complete"
    assert parsed["n_parsed"] == 40
    partial = parse_completion("1. strongly agree\n2. I agree strongly")
    assert partial["status"] == "unparseable"
    assert partial["answers"][0] == "strongly agree"
    assert partial["answers"][1] is None
    refused = parse_completion("I'm sorry, I can't answer that.")
    assert refused["status"] == "refusal"
    with pytest.raises(ValueError):
        parse_completion("   ")


def test_cell_rule_drops_small_cells_children_and_is_deterministic():
    adults = _people(12, group="autistic")
    children = _people(12, group="autistic", age=10)
    children["userid"] = children["userid"] + 100
    rare = _people(3, group="autistic", sex=3)
    rare["userid"] = rare["userid"] + 200
    controls = _people(12, group="no_diagnosis")
    frame = pd.concat([adults, children, rare, controls], ignore_index=True)
    first = draw_group(frame, "autistic", 5, 20261029)
    second = draw_group(frame, "autistic", 5, 20261029)
    assert first["persona_id"].tolist() == second["persona_id"].tolist()
    assert set(first["age_band"]) == {"30-44"}
    assert set(first["sex"]) == {"Female"}
    assert "Asperger Syndrome (AS)" in set(first["autism_detail"])
    controls_drawn = draw_group(frame, "no_diagnosis", 4, 20261029)
    assert controls_drawn["diagnoses"].iloc[0] == [
        "I have not been diagnosed with any of these conditions"
    ]
    blank = _people(12, group="autistic")
    blank["userid"] = blank["userid"] + 300
    blank["age"] = np.nan
    blank["sex"] = np.nan
    blank["education"] = np.nan
    blank["occupation"] = np.nan
    blank["country_region"] = np.nan
    frame = pd.concat([frame, blank], ignore_index=True)
    with pytest.raises(RuntimeError, match="only"):
        draw_group(frame, "autistic", 13, 20261029)
    assert len(draw_ids(["b", "a", "c"], 2, 1)) == 2


def test_condition_3_message_from_a_drawn_persona():
    frame = pd.concat(
        [_people(10, group="autistic"), _people(10, group="no_diagnosis")],
        ignore_index=True,
    )
    drawn = draw_group(frame, "autistic", 2, 20261029)
    text = user_message(drawn.iloc[0].to_dict(), 3)
    assert "Diagnoses: Autism Spectrum Disorder" in text
    assert "Age: 30-44" in text
