import json

import pytest

from scripts.human_calibration import TERCILE_NAMES, assign_tercile, stratify, tercile_bounds


def test_tercile_bounds_split_into_thirds():
    assert tercile_bounds([0.0, 1.0, 2.0]) == (1.0, 2.0)
    with pytest.raises(ValueError):
        tercile_bounds([0.1, 0.2])


def test_low_human_entropy_is_the_high_consensus_tercile():
    bounds = (0.5, 1.2)
    assert assign_tercile(0.1, bounds) == "high_consensus"
    assert assign_tercile(0.8, bounds) == "mid"
    assert assign_tercile(1.5, bounds) == "high_disagreement"


def _fixtures(tmp_path, correct_flags, entropies):
    human = tmp_path / "human.json"
    human.write_text(
        json.dumps({"trials": {f"t{i}": {"human_entropy": e} for i, e in enumerate(entropies)}}),
        encoding="utf-8",
    )
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(
        json.dumps(
            {
                "model": "qwen3vl",
                "condition": "video_only",
                "trials": [
                    {"trial_id": f"t{i}", "stage2": {"correct": c, "prediction": "X"}}
                    for i, c in enumerate(correct_flags)
                ],
            }
        ),
        encoding="utf-8",
    )
    return evaluation, human


def test_ambiguity_tracking_model_puts_all_errors_on_disagreement_tercile(tmp_path):
    entropies = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.5, 1.6, 1.7]
    correct = [True] * 6 + [False] * 3
    evaluation, human = _fixtures(tmp_path, correct, entropies)

    report = stratify(evaluation, human)
    assert report["n_scored_with_human_data"] == 9
    assert report["terciles"]["high_consensus"]["accuracy"] == 1.0
    assert report["terciles"]["high_disagreement"]["accuracy"] == 0.0
    assert report["h1_2_error_share_on_high_consensus"] == 0.0


def test_miscalibrated_model_puts_errors_on_easy_items(tmp_path):
    """H1.2: errors on high-consensus items are calibration failure, not ambiguity."""
    entropies = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.5, 1.6, 1.7]
    correct = [False] * 3 + [True] * 6
    evaluation, human = _fixtures(tmp_path, correct, entropies)

    report = stratify(evaluation, human)
    assert report["terciles"]["high_consensus"]["accuracy"] == 0.0
    assert report["h1_2_error_share_on_high_consensus"] == 1.0
    assert report["n_errors_total"] == 3


def test_error_shares_sum_to_one_across_terciles(tmp_path):
    entropies = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.5, 1.6, 1.7]
    correct = [True, False, True, False, True, False, True, False, True]
    evaluation, human = _fixtures(tmp_path, correct, entropies)

    report = stratify(evaluation, human)
    shares = [report["terciles"][name]["share_of_all_errors"] for name in TERCILE_NAMES]
    assert abs(sum(shares) - 1.0) < 1e-9


def test_trials_without_human_data_are_excluded(tmp_path):
    evaluation, human = _fixtures(tmp_path, [True] * 9, [0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.5, 1.6, 1.7])
    obj = json.loads(evaluation.read_text())
    obj["trials"].append({"trial_id": "not_in_human", "stage2": {"correct": False}})
    # An unscored trial (stage 1 only) must not be counted as an error either.
    obj["trials"].append({"trial_id": "t0", "stage2": {"correct": None}})
    evaluation.write_text(json.dumps(obj), encoding="utf-8")

    report = stratify(evaluation, human)
    assert report["n_scored_with_human_data"] == 9
    assert report["n_errors_total"] == 0


def test_too_few_matches_reports_error_rather_than_crashing(tmp_path):
    evaluation, human = _fixtures(tmp_path, [True, False], [0.1, 1.5])
    report = stratify(evaluation, human)
    assert "error" in report
    assert report["n_scored_with_human_data"] == 2
