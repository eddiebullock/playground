"""Synthetic checks for scoring, exclusions, Eyes parsing, and YBT text handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scog.data.card import load_card, normalise_country, normalise_sex, parse_eyes_itemised
from scog.data.channel4 import (
    AQ_COLS,
    COLUMNS,
    EQ_COLS,
    ITEM_COLS,
    SQR_COLS,
    clean_channel4,
    load_channel4,
)
from scog.data.ybt import code_likert, load_ybt, parse_multiselect
from scog.scoring.eyes import EYES_ITEMS, EYES_KEY
from scog.scoring.short_forms import (
    AQ10_AGREE,
    EQ10_AGREE,
    SPQ10_AGREE,
    SQR10_AGREE,
    score_items,
    texts_match,
)


def _channel_row(**overrides):
    row = {col: np.nan for col in COLUMNS}
    row.update(
        {
            "userid": 1,
            "age": 30,
            "sex": 2,
            "handedness": 1,
            "education": 3,
            "occupation": 1,
            "country_region": 1,
            "repeat": 0,
            "diagnosis_0": 9,
        }
    )
    for i, col in enumerate(ITEM_COLS):
        row[col] = 1 if i % 2 == 0 else 3
    row.update(overrides)
    return row


def _trait_items(*, high_autistic_traits: bool):
    """Score 1 on every item of AQ and SQ-R, and 0 on EQ, or the reverse."""
    out = {}
    for col, agree in zip(AQ_COLS, AQ10_AGREE):
        out[col] = (1 if agree else 4) if high_autistic_traits else (4 if agree else 1)
    for col, agree in zip(EQ_COLS, EQ10_AGREE):
        out[col] = (4 if agree else 1) if high_autistic_traits else (1 if agree else 4)
    for col, agree in zip(SQR_COLS, SQR10_AGREE):
        out[col] = (1 if agree else 4) if high_autistic_traits else (4 if agree else 1)
    return out


def test_published_polarity_matches_the_keys_used_so_far():
    assert AQ10_AGREE == (1, 0, 0, 0, 0, 0, 1, 1, 0, 1)
    assert EQ10_AGREE == (1, 1, 0, 1, 0, 1, 0, 0, 1, 0)
    assert SQR10_AGREE == (1, 0, 1, 1, 0, 1, 1, 0, 1, 1)
    assert SPQ10_AGREE == (1,) * 10


def test_binary_and_graded_schemes_differ_and_missing_fails_the_row():
    responses = pd.DataFrame([[1, 4], [2, 3]])
    binary = score_items(responses, (1, 0), scheme="binary")
    graded = score_items(responses, (1, 0), scheme="graded")
    assert list(binary) == [2, 2]
    assert list(graded) == [4, 2]
    missing = score_items(pd.DataFrame([[1, np.nan]]), (1, 0), scheme="binary")
    assert np.isnan(missing.iloc[0])


def test_inferred_coding_direction_not_the_reverse():
    rows = [
        _channel_row(userid=10, diagnosis_0=2, **_trait_items(high_autistic_traits=True)),
        _channel_row(userid=11, diagnosis_0=2, **_trait_items(high_autistic_traits=True)),
        _channel_row(userid=12, diagnosis_0=9, **_trait_items(high_autistic_traits=False)),
    ]
    cleaned, audit = clean_channel4(pd.DataFrame(rows))
    autistic = cleaned.loc[cleaned["autistic"], "aq10_binary"].mean()
    control = cleaned.loc[cleaned["no_diagnosis"], "aq10_binary"].mean()
    flipped = cleaned.loc[cleaned["autistic"], "aq10_binary_flipped_coding"].mean()
    assert autistic > control
    assert cleaned.loc[cleaned["no_diagnosis"], "eq10_binary"].mean() > cleaned.loc[
        cleaned["autistic"], "eq10_binary"
    ].mean()
    assert not (flipped > control)
    assert audit["n_test_users_dropped"] == 0


def test_channel4_exclusions_and_ambiguous_group():
    rows = [
        _channel_row(userid=174269, diagnosis_0=2),
        _channel_row(userid=20, repeat=1, diagnosis_0=2),
        _channel_row(userid=21, age=0, diagnosis_0=2),
        _channel_row(userid=22, age=95, diagnosis_0=9),
        _channel_row(userid=23, diagnosis_0=9, autism_diagnosis_0=1),
        _channel_row(userid=24, **{col: 2 for col in ITEM_COLS}),
        _channel_row(userid=25, diagnosis_0=2),
    ]
    cleaned, audit = clean_channel4(pd.DataFrame(rows))
    assert audit["n_test_users_dropped"] == 1
    assert audit["n_repeat_dropped"] == 1
    assert audit["n_identical_items_dropped"] == 1
    assert audit["n_age0_set_missing"] == 1
    assert audit["n_age_over90_set_missing"] == 1
    assert set(cleaned["userid"]) == {21, 22, 23, 25}
    assert cleaned.loc[cleaned["userid"] == 21, "age"].isna().all()
    assert cleaned.loc[cleaned["userid"] == 22, "age"].isna().all()
    ambiguous = cleaned.loc[cleaned["userid"] == 23].iloc[0]
    assert ambiguous["ambiguous_subtype"]
    assert not ambiguous["autistic"]
    assert not ambiguous["no_diagnosis"]
    assert int(cleaned["autistic"].sum()) == 2


def test_channel4_load_fails_on_empty_and_bad_header(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero rows"):
        load_channel4(empty)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing columns"):
        load_channel4(bad)
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        load_channel4(missing)


def test_eyes_key_shape_and_caption_checks():
    assert len(EYES_KEY) == 36
    assert EYES_KEY == (
        1, 2, 3, 2, 3, 2, 3, 1, 4, 1, 3, 3, 2, 4, 1, 2, 1, 1, 4, 2, 2, 1, 3, 1,
        4, 3, 2, 1, 4, 2, 2, 1, 4, 3, 2, 3,
    )
    by_item = {item["item"]: item for item in EYES_ITEMS}
    assert by_item[29]["options"][3] == "reflective"
    assert by_item[32]["options"][0] == "serious"


def test_eyes_parser_ignores_second_value_and_matches_header():
    pairs = []
    for answer in EYES_KEY:
        pairs.extend([str(answer), "999"])
    # Two missing items still kept; a third would be excluded.
    pairs[0] = "-1"
    pairs[1] = ""
    pairs[2] = ""
    pairs[3] = ""
    raw = ",".join(["0", "34", "2", "100"] + pairs)
    parsed = parse_eyes_itemised(raw)
    assert parsed["n_missing"] == 2
    assert parsed["n_correct"] == 34
    assert parsed["header_matches_key"]
    assert not parsed["excluded_missing"]
    assert normalise_country("(United Kingdom)") == "United Kingdom"
    assert normalise_country("0") is None
    assert normalise_sex("f") == "F"


def test_card_load_uses_latin1(tmp_path: Path):
    from scog.data.card import COLUMNS

    path = tmp_path / "card.csv"
    header = ",".join(COLUMNS)
    # A non-ascii byte that is legal in latin-1 and illegal as a lone utf-8 start.
    row = ["1", "EyesTest", "1", "t", "1990", "30", "x", "y", "18", "Caf\u00e9", "f", "R", "0", "", "0", "", ""]
    path.write_bytes((header + "\n" + ",".join(row) + "\n").encode("latin-1"))
    df = load_card(path)
    assert len(df) == 1
    assert df.loc[0, "Sex"] == "f"


def test_ybt_skips_preamble_and_renames_duplicate_q311(tmp_path: Path):
    path = tmp_path / "YBT.csv"
    path.write_text(
        "Finished,Q311,Q311,diagnosis,aq_1\n"
        "finished?,IQ,genetic,dx,item\n"
        '{"ImportId":"a"},{"ImportId":"b"},{"ImportId":"c"},{"ImportId":"d"},{"ImportId":"e"}\n'
        "1,120,none,Autism Spectrum Disorder,strongly agree\n"
        "0,110,none,,slightly disagree\n",
        encoding="utf-8",
    )
    df, _questions = load_ybt(path)
    assert list(df["Finished"]) == [1, 0]
    assert "Q311__IQ" in df.columns
    assert "Q311__genetic" in df.columns
    coded = code_likert(df["aq_1"])
    assert list(coded) == [1, 3]


def test_diagnosis_parser_keeps_commas_inside_labels():
    options = [
        "Anxiety Disorder (e.g., Generalized Anxiety Disorder)",
        "Autism Spectrum Disorder",
        "Depression",
    ]
    text = (
        "Anxiety Disorder (e.g., Generalized Anxiety Disorder),"
        "Autism Spectrum Disorder"
    )
    found = parse_multiselect(text, options)
    assert set(found) == {
        "Anxiety Disorder (e.g., Generalized Anxiety Disorder)",
        "Autism Spectrum Disorder",
    }
    with pytest.raises(ValueError):
        parse_multiselect("Not a real option", options)


def test_item_text_normalisation():
    assert texts_match(["Hello, world."], ["hello world"])
    assert not texts_match(["one", "two"], ["two", "one"])
