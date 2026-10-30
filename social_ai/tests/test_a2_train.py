"""Synthetic checks for A2 example formatting and hyperparameter selection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scog.data.channel4 import DIAGNOSIS_COLS, ITEM_COLS
from scog.data.splits import assert_split_allowed, frozen_studies
from scog.train.examples import (
    build_examples,
    dev_id_set,
    format_persona,
    held_out_mask,
    item_permutation,
    shuffled_id_set,
)
from scog.train.prepare import hash_mmlu, plan_splits, sample_mmlu
from scog.train.select import choose, load_grid_records


def _row(userid: int, *, sex: int = 1, group: str = "no_diagnosis", diagnosis: int = 9) -> dict:
    row = {col: np.nan for col in ["userid", "age", "sex", "education", "occupation", "country_region", "primary_group", *DIAGNOSIS_COLS, *ITEM_COLS]}
    row.update(
        {
            "userid": userid,
            "age": 30 + (userid % 20),
            "sex": sex,
            "education": 3,
            "occupation": 1,
            "country_region": 4,
            "primary_group": group,
            "diagnosis_0": diagnosis,
        }
    )
    for index, col in enumerate(ITEM_COLS):
        row[col] = 1 if index % 2 == 0 else 4
    return row


def _frame(n: int = 40) -> pd.DataFrame:
    rows = []
    for userid in range(1, n + 1):
        if userid <= 8:
            rows.append(_row(userid, sex=2, group="autistic", diagnosis=2))
        else:
            rows.append(_row(userid, sex=1, group="no_diagnosis", diagnosis=9))
    return pd.DataFrame(rows)


def test_a2_freeze_lists_only_a2():
    assert "A2" in frozen_studies()
    assert_split_allowed("confirmatory", study="A2")
    assert_split_allowed("exploratory")
    with pytest.raises(RuntimeError, match="locked"):
        assert_split_allowed("confirmatory", study="A1")
    with pytest.raises(RuntimeError, match="locked"):
        assert_split_allowed("confirmatory")


def test_persona_uses_confirmed_labels():
    row = pd.Series(_row(3, sex=2, group="autistic", diagnosis=2))
    text = format_persona(row)
    assert "Sex: Female" in text
    assert "Diagnoses: Autism Spectrum Disorder" in text
    assert "Education: Undergraduate" in text
    none = format_persona(pd.Series(_row(4)))
    assert "Diagnoses: None of these" in none


def test_shuffle_and_dev_rules():
    ids = [str(i) for i in range(20)]
    shuffled = shuffled_id_set(ids)
    assert shuffled == {ids[i] for i in range(20) if i % 2 == 1}
    dev = dev_id_set(ids)
    assert dev == {"9", "19"}
    order = item_permutation("9")
    assert order == item_permutation("9")
    assert sorted(order) == list(range(40))
    kept, counts = build_examples(_frame(6), ["1", "2"], shuffled={"2"})
    assert counts["n_kept"] == 2
    by_id = {row["id"]: row for row in kept}
    assert "\n1. I would be able to distinguish" in by_id["1"]["prompt"]
    assert by_id["2"]["prompt"] != by_id["1"]["prompt"]
    assert "strongly agree" in by_id["1"]["answer"]
    assert "strongly disagree" in by_id["1"]["answer"]


def test_held_out_women_leave_the_training_pools():
    frame = _frame(80)
    pools = plan_splits(frame, check_hash=False)
    held = set(frame.loc[held_out_mask(frame), "userid"].astype(int).astype(str))
    assert held
    assert held.isdisjoint(pools["explore_train"])
    assert held.isdisjoint(pools["dev"])
    assert held.isdisjoint(pools["confirmatory"])
    assert set(pools["dev"]).isdisjoint(pools["explore_train"])


def test_incomplete_items_are_counted():
    frame = _frame(120)
    frame.loc[0, ITEM_COLS[0]] = np.nan
    ids = [str(i) for i in range(1, 121)]
    kept, counts = build_examples(frame, ids, shuffled=set())
    assert counts["n_skipped_incomplete"] == 1
    assert counts["n_kept"] == 119
    tiny = _frame(4)
    tiny.loc[0, ITEM_COLS[0]] = np.nan
    with pytest.raises(RuntimeError, match="above 1%"):
        build_examples(tiny, ["1", "2", "3", "4"], shuffled=set())


def test_selection_tie_breaks_toward_the_smaller_run():
    records = [
        {"run_id": "a", "epoch": 2, "rank": 16, "lr": 1e-4, "nll": 1.0, "n_examples": 2000},
        {"run_id": "b", "epoch": 1, "rank": 64, "lr": 2e-4, "nll": 1.0, "n_examples": 2000},
        {"run_id": "c", "epoch": 1, "rank": 16, "lr": 2e-4, "nll": 1.0, "n_examples": 2000},
        {"run_id": "d", "epoch": 1, "rank": 16, "lr": 5e-5, "nll": 1.2, "n_examples": 2000},
    ]
    winner = choose(records)
    assert winner["run_id"] == "c"
    with pytest.raises(RuntimeError, match="fewer than 10"):
        choose([{"run_id": "a", "epoch": 1, "rank": 16, "lr": 1e-4, "nll": 1.0, "n_examples": 3}])


def test_grid_records_wait_for_every_epoch(tmp_path):
    grid = [
        {"run_id": "r16_lr5e-05", "rank": 16, "lr": 5e-5},
        {"run_id": "r16_lr1e-04", "rank": 16, "lr": 1e-4},
    ]
    assert load_grid_records(tmp_path, grid) is None
    for cell in grid:
        run = tmp_path / cell["run_id"]
        run.mkdir()
        for epoch in (1, 2):
            (run / f"dev_nll_epoch{epoch}.json").write_text(
                json.dumps({"nll": 1.0 + epoch / 100, "n_examples": 2000, "n_tokens": 100}),
                encoding="utf-8",
            )
    records = load_grid_records(tmp_path, grid)
    assert len(records) == 4
    assert choose(records)["epoch"] == 1


def test_resume_asks_for_a_smaller_disk_reserve():
    from scog.train.prepare import headroom_requirement

    assert headroom_requirement(True, True) == (1.0, 2000)
    assert headroom_requirement(False, True) == (30.0, 80000)
    assert headroom_requirement(True, False) == (30.0, 80000)


def test_mmlu_skips_configs_without_a_test_split():
    from scog.train.prepare import mmlu_subject_names

    splits = {
        "all": ["test"],
        "abstract_algebra": ["test", "validation", "dev"],
        "auxiliary_train": ["train"],
    }
    assert mmlu_subject_names(list(splits), splits.get) == ["abstract_algebra"]
    with pytest.raises(RuntimeError, match="no subject with a test split"):
        mmlu_subject_names(["all", "auxiliary_train"], splits.get)


def test_mmlu_sample_is_stable_and_hashed():
    pairs = [(f"s{i % 3}", i) for i in range(50)]
    first = sample_mmlu(pairs, n=20, seed=20261029)
    assert first == sample_mmlu(pairs, n=20, seed=20261029)
    assert len(first) == 20
    assert len({(row["subject"], row["index"]) for row in first}) == 20
    assert hash_mmlu(first) == hash_mmlu(list(first))
    with pytest.raises(RuntimeError, match="MMLU pool"):
        sample_mmlu(pairs[:5], n=20)
