from __future__ import annotations

import json
from pathlib import Path

from esvlm.tasks.sdomain import (
    clevrer_to_trials,
    keep_physbench_item,
    load_sdomain_items,
    parse_lettered_options,
    physbench_to_trial,
    piqa_to_trial,
    write_manifest,
)


def test_parse_lettered_options() -> None:
    stem, options = parse_lettered_options(
        "What happens next?\n A. It accelerates\nB. It stops\nC. It floats\nD. It vanishes\n"
    )
    assert stem == "What happens next?"
    assert options == ["It accelerates", "It stops", "It floats", "It vanishes"]


def test_physbench_keeps_dynamics_text_drops_image_options(tmp_path: Path) -> None:
    keep = {
        "idx": 1,
        "question": "After the collision, the block\n A. speeds up\nB. stays still\nC. disappears\nD. melts\n",
        "answer": "A",
        "task_type": "phenomena",
        "sub_type": "collision",
        "ability_type": "prediction",
        "file_name": ["clip.mp4"],
    }
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "clip.mp4").write_bytes(b"x")
    assert keep_physbench_item(keep)
    trial = physbench_to_trial(keep, media_root=tmp_path)
    assert trial is not None
    assert trial["correct_idx"] == 0
    assert trial["media"] == "video/clip.mp4"

    image_opts = {
        **keep,
        "question": "Which frame?\n A. <image>\nB. <image>\nC. <image>\nD. <image>\n",
    }
    assert physbench_to_trial(image_opts, media_root=tmp_path) is None

    property_row = {**keep, "task_type": "property", "sub_type": "color", "ability_type": "recognition"}
    assert keep_physbench_item(property_row) is False


def test_clevrer_keeps_mcq_skips_descriptive() -> None:
    row = {
        "video_index": 10000,
        "video_filename": "video_10000.mp4",
        "questions": [
            {
                "question_id": "d1",
                "question_type": "descriptive",
                "question": "How many spheres?",
                "choices": [{"choice": "one", "answer": "correct"}, {"choice": "two", "answer": "wrong"}],
            },
            {
                "question_id": "e1",
                "question_type": "explanatory",
                "question": "Which event caused the collision?",
                "choices": [
                    {"choice": "the cube hit the sphere", "answer": "wrong"},
                    {"choice": "the sphere hit the cube", "answer": "correct"},
                ],
            },
        ],
    }
    trials = clevrer_to_trials(row)
    assert len(trials) == 1
    assert trials[0]["item_id"] == "clevrer_e1"
    assert trials[0]["correct_idx"] == 1


def test_piqa_and_loader(tmp_path: Path) -> None:
    trial = piqa_to_trial(
        {"goal": "To hang a picture", "sol1": "use a nail", "sol2": "use soup", "label": 0},
        split="valid",
        idx=0,
    )
    assert trial is not None
    man = tmp_path / "manifest.json"
    write_manifest(man, "piqa", [trial])
    items = load_sdomain_items(man, tmp_path, task="piqa")
    assert len(items) == 1
    assert items[0].domain == "S"
    assert items[0].media_type == "none"
    assert "physical or mechanical" in items[0].prompt_text
    payload = json.loads(man.read_text(encoding="utf-8"))
    assert payload["n"] == 1
