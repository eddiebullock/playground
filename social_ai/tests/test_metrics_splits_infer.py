"""Metrics, frozen-split hashes, and the inference harness."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scog.data.eu_emotion import match_videos, response_entropy_bits, stimulus_table
from scog.data.splits import (
    assign_splits,
    assert_split_allowed,
    hash_ids,
    read_lock,
    write_id_list,
)
from scog.infer.api_client import ClosedModelClient
from scog.infer.jsonl_store import JsonlStore
from scog.infer.permutations import all_orders, orders_for_six
from scog.infer.smoke import smoke_calls
from scog.infer.vllm_server import multiple_choice_prompt, server_command
from scog.metrics.bootstrap import bootstrap_mean_ci
from scog.metrics.divergence import js_divergence, mean_item_js


def test_js_is_zero_for_identical_and_one_bit_for_disjoint_mass():
    assert js_divergence([1, 0], [1, 0]) == pytest.approx(0.0)
    # Equal mixture of two point masses is 1 bit.
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    summary = mean_item_js(np.array([[10, 0], [5, 5]]), np.array([[10, 0], [5, 5]]))
    assert summary["mean"] == pytest.approx(0.0)


def test_entropy_of_a_sure_response_is_zero():
    props = np.array([[1, 0, 0, 0, 0, 0], [1 / 6] * 6])
    entropy = response_entropy_bits(props)
    assert entropy[0] == pytest.approx(0.0)
    assert entropy[1] == pytest.approx(np.log2(6))


def test_video_match_is_case_insensitive_and_reports_collisions(tmp_path):
    (tmp_path / "AfraidSF3.mp4").write_bytes(b"")
    (tmp_path / "afraidSF3.mov").write_bytes(b"")
    (tmp_path / "other.mp4").write_bytes(b"")
    report = match_videos(["afraidsf3", "missingID"], list(tmp_path.iterdir()))
    assert report["n_matched"] == 1
    assert report["missing_stimuli"] == ["missingID"]
    assert report["extra_files"] == ["other.mp4"]
    assert report["collisions"]


def test_splits_are_deterministic_and_hash_stable(tmp_path):
    ids = pd.Series([f"p{i}" for i in range(40)])
    strata = pd.Series(["autistic|F|30_44"] * 40)
    first = assign_splits(ids, strata, seed=20261029, with_a2=True)
    second = assign_splits(ids, strata, seed=20261029, with_a2=True)
    assert first["split"].tolist() == second["split"].tolist()
    counts = first["split"].value_counts().to_dict()
    assert counts["exploratory"] == 8
    assert counts["a2_test"] == 4
    assert counts["confirmatory"] == 28
    digest = hash_ids(first.loc[first["split"] == "exploratory", "id"].tolist())
    path = tmp_path / "exploratory.txt"
    assert write_id_list(path, first.loc[first["split"] == "exploratory", "id"].tolist()) == digest
    tiny = assign_splits(pd.Series(["a", "b"]), pd.Series(["g|s|age"] * 2), with_a2=True)
    assert set(tiny["split"]) == {"confirmatory"}


def test_confirmatory_split_is_locked_until_prereg():
    assert_split_allowed("exploratory")
    with pytest.raises(RuntimeError, match="locked"):
        assert_split_allowed("confirmatory")
    with pytest.raises(RuntimeError, match="locked"):
        assert_split_allowed("confirmatory", study="A1")


def test_emotion_intensity_is_not_the_mean_rating():
    frame = pd.DataFrame(
        {
            "Expression modality": ["Face"] * 6,
            "Expressed emotion": ["Afraid"] * 6,
            "Emotion intensity": ["High"] * 6,
            "Stimulus identity": [f"s{i}" for i in range(6)],
            "Number of interviews per item": [100] * 6,
            "Target emotion": [0.8] * 6,
            "Control emotion 1": [0.04] * 6,
            "Control emotion 2": [0.04] * 6,
            "Control emotion 3": [0.04] * 6,
            "Control emotion 4": [0.04] * 6,
            "Control 5 - I don't know": [0.04] * 6,
            "Intensity": [3.5] * 6,
        }
    )
    table = stimulus_table(frame)
    assert set(table["intensity"]) == {"High"}


def test_lock_parser_rejects_a_short_digest():
    parsed = read_lock("channel4.exploratory " + "ab" * 32 + "\n")
    assert len(parsed) == 1
    with pytest.raises(RuntimeError):
        read_lock("channel4.exploratory abc\n")


def test_jsonl_store_skips_finished_keys(tmp_path):
    store = JsonlStore(tmp_path / "out.jsonl")
    record = {"model": "m", "item": "1", "condition": "c", "permutation": 0, "seed": 1, "answer": "A"}
    assert store.append(record)
    assert not store.append(record)
    reopened = JsonlStore(tmp_path / "out.jsonl")
    assert reopened.contains(record)
    assert sum(1 for _ in (tmp_path / "out.jsonl").open()) == 1


def test_option_orders():
    orders = all_orders(["a", "b", "c", "d"])
    assert len(orders) == 24
    assert len(set(orders)) == 24
    six = orders_for_six(["a", "b", "c", "d", "e", "f"], n_random=4, seed=1)
    assert len(six) == 10
    assert len(set(six)) == 10
    for row in six[:6]:
        assert sorted(row) == ["a", "b", "c", "d", "e", "f"]


def test_client_caches_and_retries(tmp_path):
    calls = {"n": 0}

    def transport(request):
        calls["n"] += 1
        if calls["n"] < 2:
            return {"status": 429}
        return {"status": 200, "text": "ok", "echo": request["seed"]}

    slept = []
    client = ClosedModelClient(
        transport,
        tmp_path,
        min_interval_s=0.0,
        sleeper=slept.append,
    )
    first = client.complete("m", "p", ["a", "b"], 3)
    second = client.complete("m", "p", ["a", "b"], 3)
    assert first == second
    assert calls["n"] == 2
    assert slept  # the 429 backed off
    # Option order is part of the key.
    client.complete("m", "p", ["b", "a"], 3)
    assert calls["n"] == 3


def test_client_does_not_retry_a_hard_failure(tmp_path):
    def transport(_request):
        return {"status": 400}

    client = ClosedModelClient(transport, tmp_path, sleeper=lambda _s: None)
    with pytest.raises(RuntimeError, match="400"):
        client.complete("m", "p", ["a"], 1)


def test_smoke_plan_and_server_command():
    calls = smoke_calls("example-model")
    assert len(calls) == 2
    assert {call["item"] for call in calls} == {"smoke-mcq", "smoke-gen"}
    prompt = multiple_choice_prompt("Stem", ["one", "two"])
    assert "A. one" in prompt
    command = server_command(model_path="/weights/example")
    assert command[0] == "vllm"
    assert json.dumps(command)


def test_bootstrap_ci_covers_a_constant():
    result = bootstrap_mean_ci([1, 1, 1, 1], n_boot=20, seed=1)
    assert result["mean"] == 1
    assert result["ci_low"] == result["ci_high"] == 1
