from __future__ import annotations

import json
from pathlib import Path

import pytest

from esvlm.tasks.items import load_eu_emotions_items, load_rmet_items, parse_manifest_trial
from esvlm.tasks.prompts import RMET_TEMPLATES, render_prompt
from esvlm.types import MCQItem


def test_mcqitem_validates() -> None:
    with pytest.raises(ValueError):
        MCQItem(
            item_id="x",
            task="t",
            domain="E",
            media="",
            media_type="image",
            prompt_text="",
            options=("a",),
            correct_idx=0,
        )
    with pytest.raises(ValueError):
        MCQItem(
            item_id="x",
            task="t",
            domain="E",
            media="",
            media_type="none",
            prompt_text="",
            options=("a", "b"),
            correct_idx=2,
        )


def test_parse_manifest_trial(tmp_path: Path) -> None:
    trial = {
        "item_id": "03",
        "options": ["a", "b", "c"],
        "correct_label": "b",
        "image": "03.png",
    }
    item = parse_manifest_trial(
        trial,
        task="rmet",
        domain="E",
        media_root=tmp_path,
        default_media_type="image",
        template_id="classic",
        prompt_text="p",
    )
    assert item.correct_idx == 1
    assert item.media.endswith("03.png")


def test_rmet_loader_and_classic_prompt(tmp_path: Path) -> None:
    stim = tmp_path / "stim"
    stim.mkdir()
    (stim / "01.png").write_bytes(b"x")
    manifest = {
        "trials": [
            {"item": 1, "options": ["worried", "friendly", "calm"], "image": "01.png"},
        ]
    }
    key = {"answers": {"1": "worried"}}
    man_path = tmp_path / "manifest.json"
    key_path = tmp_path / "answer_key.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    key_path.write_text(json.dumps(key), encoding="utf-8")
    items = load_rmet_items(man_path, key_path, stim, template_id="classic")
    assert len(items) == 1
    assert items[0].correct_idx == 0
    assert "thinking or feeling" in items[0].prompt_text
    classic = RMET_TEMPLATES["classic"]
    assert "Choose which word best describes what the person in the picture is thinking or feeling" in classic
    rendered = render_prompt("rmet", "classic", ["a", "b"])
    assert "Choose which word best describes" in rendered


def test_eu_emotions_refuses_subset(tmp_path: Path) -> None:
    path = tmp_path / "eu_emotions_118_manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="118"):
        load_eu_emotions_items(path, tmp_path)


def test_synthetic_items(synthetic_dir: Path) -> None:
    from esvlm.tasks.items import load_synthetic_items

    items = load_synthetic_items(synthetic_dir)
    assert len(items) >= 12
    assert all(Path(i.media).is_file() for i in items)
    assert {i.domain for i in items} == {"control"}
