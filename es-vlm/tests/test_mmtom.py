from __future__ import annotations

import json
from pathlib import Path

from esvlm.tasks.mmtom import (
    load_mmtom_items,
    load_mmtom_jsonl,
    mmtom_to_trial,
    parse_ab_options,
)

SHORT_Q = (
    "Question: If Jen wants a plate, which is more likely to be true? "
    "(a) Jen thinks the plate is in the fridge. "
    "(b) Jen thinks the plate is not in the fridge. "
    "Please respond with either a or b."
)


def test_parse_ab_options() -> None:
    left, right = parse_ab_options(SHORT_Q)
    assert left == "Jen thinks the plate is in the fridge."
    assert right == "Jen thinks the plate is not in the fridge."


def test_mmtom_to_trial() -> None:
    row = {
        "question": SHORT_Q,
        "answer": "b",
        "question_type": 1.3,
        "episode": 6,
        "start_time": 0,
        "end_time": 5,
        "answer_list": ["a", "b"],
    }
    trial = mmtom_to_trial(row, idx=0)
    assert trial["item_id"] == "mmtom_0_6"
    assert trial["correct_idx"] == 1
    assert trial["media"] == ""
    assert trial["media_type"] == "none"
    assert trial["question_type"] == 1.3
    assert trial["episode"] == 6
    assert trial["end_time"] == 5
    assert trial["options"] == [
        "Jen thinks the plate is in the fridge.",
        "Jen thinks the plate is not in the fridge.",
    ]
    assert trial["question"] == "If Jen wants a plate, which is more likely to be true?"

    labeled = mmtom_to_trial(
        {
            **row,
            "answer": "A",
            "answer_list": [
                "the agent saw the cupcake",
                "the agent did not see the cupcake",
            ],
        },
        idx=3,
    )
    assert labeled["item_id"] == "mmtom_3_6"
    assert labeled["correct_idx"] == 0
    assert labeled["options"] == [
        "the agent saw the cupcake",
        "the agent did not see the cupcake",
    ]


def test_load_mmtom_items_from_jsonl(tmp_path: Path) -> None:
    rows = [
        {
            "question": SHORT_Q,
            "answer": "b",
            "question_type": 1.3,
            "episode": 6,
            "end_time": 5,
            "answer_list": ["a", "b"],
        },
        {
            "question": (
                "Question: If Mark wants a salmon, which is more likely? "
                "(a) Mark thinks the salmon is in the microwave. "
                "(b) Mark thinks the salmon is not in the microwave. "
                "Please respond with either a or b."
            ),
            "answer": "a",
            "question_type": 1.1,
            "episode": 23,
            "end_time": 21,
            "answer_list": ["a", "b"],
        },
    ]
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert len(load_mmtom_jsonl(path)) == 2

    items = load_mmtom_items(path)
    assert len(items) == 2
    assert items[0].item_id == "mmtom_0_6"
    assert items[1].item_id == "mmtom_1_23"
    assert items[0].domain == "E"
    assert items[0].task == "mmtom"
    assert items[0].media == ""
    assert items[0].media_type == "none"
    assert items[0].correct_idx == 1
    assert items[1].correct_idx == 0
    assert items[0].options[0] == "Jen thinks the plate is in the fridge."
    assert items[0].prompt_text.startswith("If Jen wants a plate")
    assert "Jen thinks the plate is in the fridge." in items[0].prompt_text
    assert items[0].meta["episode"] == 6
    assert items[1].meta["question_type"] == 1.1
    assert items[1].meta["end_time"] == 21
