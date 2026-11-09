"""Metric functions of research/scripts/benchmark_leak_audit.py on synthetic inputs (no network)."""

import importlib.util
import random
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "research/scripts/benchmark_leak_audit.py"
spec = importlib.util.spec_from_file_location("benchmark_leak_audit", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LABELS = ["happy", "sad", "angry", "fear", "disgust", "surprise"]


def confounded_items(n_per_label=10):
    """Our EU-Emotion design: target always first, one fixed option set per label."""
    items = []
    for j, lab in enumerate(LABELS):
        distractors = [LABELS[(j + d) % 6] for d in (1, 2, 3)]
        for _ in range(n_per_label):
            items.append({"options": [lab] + distractors, "answer_idx": 0})
    return items


def balanced_items(n=600, seed=0):
    """Random 4-option subsets of six labels, answer position uniform."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        opts = rng.sample(LABELS, 4)
        items.append({"options": opts, "answer_idx": rng.randrange(4)})
    return items


def test_position_test_all_first_is_extreme():
    res = mod.position_test([0] * 40, [4] * 40, n_sim=2000)
    assert res["position_counts"] == [40, 0, 0, 0]
    assert res["expected_counts"] == [10.0, 10.0, 10.0, 10.0]
    assert res["chi2"] == pytest.approx(120.0)
    assert res["df"] == 3
    assert res["p_mc"] == pytest.approx(1 / 2001)
    assert res["p_chi2"] < 1e-20


def test_position_test_uniform_counts_are_null():
    res = mod.position_test([0, 1, 2, 3] * 25, [4] * 100, n_sim=2000)
    assert res["chi2"] == 0
    assert res["p_mc"] == 1.0


def test_position_test_mixed_option_counts_expected():
    res = mod.position_test([0, 1, 0, 4], [2, 2, 5, 5], n_sim=0)
    assert res["expected_counts"] == [1.4, 1.4, 0.4, 0.4, 0.4]
    assert res["position_counts_by_n_options"] == {"2": [1, 1], "5": [1, 0, 0, 0, 1]}


def test_position_test_rejects_out_of_range():
    with pytest.raises(ValueError):
        mod.position_test([4], [4])


def test_option_set_leak_perfect_on_confounded_design():
    items = confounded_items()
    labels = [mod.norm(it["options"][0]) for it in items]
    sets = [tuple(sorted(it["options"])) for it in items]
    res = mod.option_set_leak(labels, sets)
    assert res["n_option_sets"] == 6
    assert res["n_labels"] == 6
    assert res["labels_per_option_set_max"] == 1
    assert res["n_labels_with_one_option_set"] == 6
    assert res["loo_option_set_accuracy"] == 1.0
    assert res["share_items_in_repeated_single_label_sets"] == 1.0


def test_option_set_leak_no_peer_falls_back_to_one_over_k():
    labels = ["a", "b"]
    sets = [("a", "x", "y"), ("b", "z")]
    res = mod.option_set_leak(labels, sets)
    assert res["share_items_no_loo_peer"] == 1.0
    assert res["loo_option_set_accuracy"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_option_set_leak_ties_scored_in_expectation():
    # Same set, three items: labels a, a, b. Item 3 (b) sees peers {a:2} -> 0.
    # Item 1 (a) sees {a:1, b:1} tie -> 0.5. Same for item 2.
    s = ("a", "b", "c")
    res = mod.option_set_leak(["a", "a", "b"], [s, s, s])
    assert res["loo_option_set_accuracy"] == pytest.approx((0.5 + 0.5 + 0) / 3)


def test_fixed_position_baseline_in_sample_and_loo():
    res = mod.fixed_position_baseline([0] * 9 + [1], [4] * 10)
    assert res["best_position"] == 0
    assert res["in_sample_accuracy"] == pytest.approx(0.9)
    assert res["loo_accuracy"] == pytest.approx(0.9)


def test_position_given_k_catches_last_option_leak():
    # 4-option items balanced; whenever there are 5 options the answer is the 5th.
    answers = [0, 1, 2, 3] * 10 + [4] * 8
    ks = [4] * 40 + [5] * 8
    res = mod.position_given_k_baseline(answers, ks)
    four = sum(res["_scores"][:40]) / 40
    five = sum(res["_scores"][40:]) / 8
    assert five == 1.0
    assert four < 0.25


def test_answer_prior_detects_never_correct_distractors():
    # Correct label always 'happy' or 'sad'; 'fear' and 'disgust' offered but never correct.
    items = []
    for i in range(40):
        lab = "happy" if i % 2 else "sad"
        items.append(([lab, "fear", "disgust"], lab))
    labels = [lab for _, lab in items]
    res = mod.answer_prior_baseline(labels, [o for o, _ in items])
    assert res["loo_accuracy"] == 1.0


def test_longest_option_baseline():
    res = mod.longest_option_baseline([["a", "bbb", "cc"], ["aa", "bb"]], [1, 0])
    assert res["accuracy"] == pytest.approx((1 + 0.5) / 2)


def test_versus_chance_zero_excess_at_chance():
    res = mod.versus_chance([0.25] * 100, [4] * 100)
    assert res["excess"] == pytest.approx(0.0)
    assert res["z"] == pytest.approx(0.0)


def test_audit_items_flags_confounded_design():
    res = mod.audit_items(confounded_items(), n_sim=500, n_null=200)
    assert res["position"]["max_position_share"] == 1.0
    assert res["position"]["p_mc"] < 0.01
    b = res["baselines"]
    assert b["always_same_position_loo"]["accuracy"] == 1.0
    assert b["option_set_only_loo"]["accuracy"] == 1.0
    assert b["option_set_only_loo"]["p_null"] < 0.01
    assert b["option_set_only_loo"]["null_mean"] < 0.4


def test_audit_items_balanced_design_is_null():
    res = mod.audit_items(balanced_items(), n_sim=500, n_null=200)
    assert res["position"]["p_mc"] > 0.01
    b = res["baselines"]
    assert abs(b["option_set_only_loo"]["accuracy"] - 0.25) < 0.06
    assert b["option_set_only_loo"]["p_null"] > 0.01
    assert b["always_same_position_in_sample"]["p_null"] > 0.01


def test_null_simulation_calibrates_loo_bias_on_balanced_binary():
    # Perfectly balanced 2-option items: LOO majority scores 0 but that is not evidence of anything.
    items = [{"options": ["yes", "no"], "answer_idx": i % 2} for i in range(100)]
    res = mod.audit_items(items, n_sim=200, n_null=200)
    loo = res["baselines"]["always_same_position_loo"]
    assert loo["accuracy"] == 0.0
    assert loo["p_null"] == 1.0


def test_holm_matches_hand_calculation():
    assert mod.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_parse_inline_options_mmstar_style():
    q = "Question: Which mood does this image convey?\nOptions: A: Sad, B: Cozy, C: Happy, D: Angry"
    stem, opts = mod.parse_inline_options(q)
    assert stem.startswith("Question:")
    assert opts == ["Sad", "Cozy", "Happy", "Angry"]


def test_face_human_bench_replay_matches_rotation_rule():
    data = {name: [] for name, _ in mod.FHB_EVAL_ORDER}
    data["face_expression_basic"] = [{"QA": {"options": ["w", "x", "y", "z"], "answer": 0}}] * 3
    out = mod.fhb_presented_positions(data)
    rng = random.Random(100)
    for (name, version), rows in out.items():
        if name != "face_expression_basic":
            assert rows == []
            continue
        for opts, ans in rows:
            rk = rng.randint(0, 3)
            assert opts == [["w", "x", "y", "z"][(i + rk) % 4] for i in range(4)]
            assert opts[ans] == "w"


def test_emobench_m_option_parser():
    p = "Choose one of the following labels as your final answer: neutral, negative, positive. Respond in the format"
    assert mod._emobm_options(p) == ["neutral", "negative", "positive"]
    p2 = "Choose one emotion: happy, surprise, sad, and neutral. Choose one intent: a, b."
    assert mod._emobm_options(p2) == ["happy", "surprise", "sad", "neutral"]
