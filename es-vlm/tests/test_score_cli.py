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


def test_licensed_task_refused() -> None:
    with pytest.raises(SystemExit, match="licensed"):
        load_task_items("rmet", None)
