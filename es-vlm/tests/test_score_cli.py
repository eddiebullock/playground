from __future__ import annotations

from pathlib import Path

import pytest

from esvlm.cli.score import load_task_items, slice_items
from esvlm.tasks.media import uniform_indices
from esvlm.tasks.sdomain import load_sdomain_items, piqa_to_trial, write_manifest


def test_uniform_indices() -> None:
    assert uniform_indices(1, 8) == [0]
    assert uniform_indices(8, 8) == list(range(8))
    idx = uniform_indices(100, 4)
    assert idx[0] == 0
    assert idx[-1] == 99
    assert len(idx) == 4


def test_slice_items_and_public_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trials = []
    for i in range(5):
        trial = piqa_to_trial(
            {"goal": f"g{i}", "sol1": "a", "sol2": "b", "label": 0},
            split="valid",
            idx=i,
        )
        assert trial is not None
        trials.append(trial)
    man = tmp_path / "manifest.json"
    write_manifest(man, "piqa", trials)
    monkeypatch.setenv("ESVLM_PIQA_MANIFEST", str(man))
    monkeypatch.setenv("ESVLM_PIQA_DIR", str(tmp_path))
    items, used = load_task_items("piqa", None)
    assert used == man
    assert len(items) == 5
    assert items[0].domain == "S"
    assert [i.item_id for i in slice_items(items, 2, 2)] == ["piqa_valid_2", "piqa_valid_3"]
    assert slice_items(items, 4, 0)[-1].item_id == "piqa_valid_4"


def test_clevrer_loader_rewrites_colliding_ids(tmp_path: Path) -> None:
    trials = [
        {
            "item_id": "clevrer_13",
            "question": "What happens next?",
            "options": ["collide", "miss"],
            "correct_idx": 0,
            "media": f"video_{idx}.mp4",
            "media_type": "video",
            "question_type": "predictive",
        }
        for idx in (10000, 10001)
    ]
    man = tmp_path / "manifest.json"
    write_manifest(man, "clevrer", trials)
    items = load_sdomain_items(man, tmp_path, task="clevrer")
    ids = [it.item_id for it in items]
    assert ids == ["clevrer_video_10000_13", "clevrer_video_10001_13"]
    assert len(set(ids)) == 2


def test_rmet_and_eu_load_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stim = tmp_path / "stim"
    stim.mkdir()
    (stim / "item_01.png").write_bytes(b"x")
    man = tmp_path / "rmet_manifest.json"
    key = tmp_path / "rmet_key.json"
    man.write_text(
        '{"trials":[{"item":1,"options":["alpha","beta"],"correct_label":"alpha",'
        '"image":"nested/path/item_01.png"}]}',
        encoding="utf-8",
    )
    key.write_text('{"items":[]}', encoding="utf-8")
    monkeypatch.setenv("ESVLM_RMET_MANIFEST", str(man))
    monkeypatch.setenv("ESVLM_RMET_ANSWER_KEY", str(key))
    monkeypatch.setenv("ESVLM_RMET_DIR", str(stim))
    items, used = load_task_items("rmet", None)
    assert used == man
    assert items[0].domain == "E"
    assert items[0].media.endswith("item_01.png")
    assert items[0].correct_idx == 0

    eu_stim = tmp_path / "eu"
    eu_stim.mkdir()
    clip = eu_stim / "clip.mp4"
    clip.write_bytes(b"x")
    eu_man = tmp_path / "eu_emotions_full_manifest.json"
    eu_ent = tmp_path / "eu_emotions_human_entropy.json"
    eu_man.write_text(
        '{"trials":[{"trial_id":"t1","label":"alpha","stimulus_path":"clip.mp4"}]}',
        encoding="utf-8",
    )
    eu_ent.write_text(
        '{"trials":{"t1":{"human_options":["alpha","beta","gamma"]}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("ESVLM_EU_EMOTIONS_MANIFEST", str(eu_man))
    monkeypatch.setenv("ESVLM_EU_EMOTIONS_DIR", str(eu_stim))
    monkeypatch.setenv("ESVLM_EU_EMOTIONS_ENTROPY", str(eu_ent))
    eu_items, eu_used = load_task_items("eu_emotions", None)
    assert eu_used == eu_man
    assert eu_items[0].domain == "E"
    assert eu_items[0].options == ("alpha", "beta", "gamma")

    with pytest.raises(SystemExit, match="unknown task"):
        load_task_items("not_a_task", None)


def test_mindreading_requires_mcq_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    man = tmp_path / "val_full.jsonl"
    man.write_text('{"trial_id":"x","label":"alpha","stimulus_path":"x.mov"}\n', encoding="utf-8")
    monkeypatch.setenv("ESVLM_MINDREADING_MANIFEST", str(man))
    monkeypatch.setenv("ESVLM_MINDREADING_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="no MCQ options"):
        load_task_items("mindreading", None)
