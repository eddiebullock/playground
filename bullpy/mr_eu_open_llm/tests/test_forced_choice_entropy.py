import json
import math

import pytest

from scripts.forced_choice_entropy import (
    UNPARSED,
    correlate_with_human,
    entropy_of_distribution,
    forced_choice_distribution,
    sample_forced_choice,
)

OPTIONS = ["Afraid", "Ashamed", "Unfriendly", "Disappointed", "Kind", "None of the above"]


def test_distribution_counts_all_options_including_zeros():
    dist = forced_choice_distribution(["Afraid"] * 4, OPTIONS)
    assert dist["Afraid"] == 1.0
    assert dist["Kind"] == 0.0
    assert set(OPTIONS).issubset(dist)


def test_unparsed_draws_are_retained_not_dropped():
    """Dropping them would understate entropy exactly where the model equivocates."""
    dist = forced_choice_distribution(["Afraid", None, "gibberish", "Kind"], OPTIONS)
    assert dist[UNPARSED] == 0.5
    assert dist["Afraid"] == 0.25
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_entropy_matches_hand_computed_values():
    assert entropy_of_distribution(forced_choice_distribution(["Afraid"] * 8, OPTIONS)) == 0.0
    split = forced_choice_distribution(["Afraid"] * 5 + ["Kind"] * 5, OPTIONS)
    assert abs(entropy_of_distribution(split) - math.log(2)) < 1e-9
    uniform = forced_choice_distribution(list(OPTIONS), OPTIONS)
    assert abs(entropy_of_distribution(uniform) - math.log(6)) < 1e-9


def test_sample_forced_choice_records_draws_and_mode():
    answers = iter(["Afraid", "Afraid", "Kind", "Afraid", "unreadable"])
    result = sample_forced_choice(
        lambda temperature: next(answers),
        lambda text: text if text in OPTIONS else None,
        OPTIONS,
        n_samples=5,
        temperature=1.0,
        seed=42,
        trial_id="emotions/EDITED/Afraid/IF3_cut.mp4",
    )
    assert result["n_samples"] == 5
    assert result["modal_prediction"] == "Afraid"
    assert result["n_unparsed"] == 1
    assert result["response_distribution"]["Afraid"] == 0.6
    assert result["forced_choice_entropy"] > 0


def test_sample_forced_choice_seeds_each_draw_distinctly():
    seen = []
    sample_forced_choice(
        lambda temperature: "Afraid",
        lambda text: text,
        OPTIONS,
        n_samples=3,
        temperature=1.0,
        seed=42,
        trial_id="trial-a",
        seed_fn=lambda seed, trial_id, draw: seen.append((seed, trial_id, draw)),
    )
    assert seen == [(42, "trial-a", 0), (42, "trial-a", 1), (42, "trial-a", 2)]


def test_temperature_is_passed_to_the_generator():
    used = []
    sample_forced_choice(
        lambda temperature: used.append(temperature) or "Kind",
        lambda text: text,
        OPTIONS,
        n_samples=2,
        temperature=0.7,
        seed=42,
        trial_id="trial-b",
    )
    assert used == [0.7, 0.7]


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_correlate_with_human_pairs_on_trial_id(tmp_path):
    human = _write(
        tmp_path,
        "human.json",
        {
            "trials": {
                "t1": {"human_entropy": 0.1},
                "t2": {"human_entropy": 0.9},
                "t3": {"human_entropy": 1.5},
                "t4": {"human_entropy": 1.7},
            }
        },
    )
    evaluation = _write(
        tmp_path,
        "eval.json",
        {
            "model": "qwen3vl",
            "condition": "video_only",
            "trials": [
                {"trial_id": "t1", "stage1": {"semantic_entropy": 0.2}, "stage2": {"forced_choice_entropy": 0.0}},
                {"trial_id": "t2", "stage1": {"semantic_entropy": 0.4}, "stage2": {"forced_choice_entropy": 0.5}},
                {"trial_id": "t3", "stage1": {"semantic_entropy": 0.6}, "stage2": {"forced_choice_entropy": 1.0}},
                {"trial_id": "t4", "stage1": {"semantic_entropy": 0.8}, "stage2": {"forced_choice_entropy": 1.4}},
                {"trial_id": "not_in_human", "stage2": {"forced_choice_entropy": 9.9}},
            ],
        },
    )
    report = correlate_with_human(evaluation, human)
    assert report["n_trials_matched_to_human"] == 4
    assert abs(report["rq1_1b_forced_choice_vs_human"]["spearman_rho"] - 1.0) < 1e-9
    assert abs(report["rq1_1a_semantic_vs_human"]["spearman_rho"] - 1.0) < 1e-9


def test_correlate_reports_zero_when_forced_choice_absent(tmp_path):
    human = _write(tmp_path, "human.json", {"trials": {"t1": {"human_entropy": 0.1}}})
    evaluation = _write(
        tmp_path,
        "eval.json",
        {"model": "gemma4", "trials": [{"trial_id": "t1", "stage2": {"prediction": "Kind"}}]},
    )
    report = correlate_with_human(evaluation, human)
    assert report["rq1_1b_forced_choice_vs_human"]["n"] == 0
    assert report["rq1_1b_forced_choice_vs_human"]["spearman_rho"] is None


def test_batched_sampling_uses_one_call_and_seeds_once():
    calls = []
    seeds = []
    result = sample_forced_choice(
        lambda temperature: pytest.fail("sequential generate must not be used when batching"),
        lambda text: text if text in OPTIONS else None,
        OPTIONS,
        n_samples=4,
        temperature=1.0,
        seed=42,
        trial_id="trial-c",
        seed_fn=lambda seed, trial_id, draw: seeds.append((seed, trial_id, draw)),
        generate_batch=lambda temperature, n: calls.append((temperature, n))
        or ["Afraid", "Afraid", "Kind", "Ashamed"],
    )
    assert calls == [(1.0, 4)]
    assert seeds == [(42, "trial-c", 0)]
    assert result["sampling_mode"] == "batched"
    assert result["response_distribution"]["Afraid"] == 0.5
    assert result["modal_prediction"] == "Afraid"


def test_batched_and_sequential_agree_on_the_same_draws():
    answers = ["Afraid", "Kind", "Afraid", "Ashamed", "Afraid"]
    shared = dict(
        options=OPTIONS, n_samples=5, temperature=1.0, seed=42, trial_id="trial-d"
    )
    it = iter(answers)
    sequential = sample_forced_choice(
        lambda temperature: next(it), lambda text: text, **shared
    )
    batched = sample_forced_choice(
        lambda temperature: "unused",
        lambda text: text,
        generate_batch=lambda temperature, n: list(answers),
        **shared,
    )
    assert sequential["response_distribution"] == batched["response_distribution"]
    assert sequential["forced_choice_entropy"] == batched["forced_choice_entropy"]


def test_short_batch_raises_rather_than_reporting_confident_zero_entropy():
    """A backend ignoring num_return_sequences would give 1 draw and H=0, not an error."""
    with pytest.raises(RuntimeError, match="incomplete sample"):
        sample_forced_choice(
            lambda temperature: "Afraid",
            lambda text: text,
            OPTIONS,
            n_samples=20,
            temperature=1.0,
            seed=42,
            trial_id="trial-e",
            generate_batch=lambda temperature, n: ["Afraid"],
        )
