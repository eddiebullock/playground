"""Readout ladder, A2 analysis, recovery observers and predictor. No model or clip."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paper_v2.schema import actor_id_from_eu_filename  # noqa: E402
from scripts import analyse_stimulus_interventions as an  # noqa: E402
from scripts import human_response_predictor as hrp  # noqa: E402
from scripts import readout_ladder as ladder  # noqa: E402
from scripts import recovery_stimulus_interventions as rec  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402


def items():
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    out = []
    for key, value in human.items():
        named = {k: v for k, v in value["human_response_distribution"].items() if k != run.ABSTAIN}
        out.append(
            {
                "trial_id": key,
                "actor": actor_id_from_eu_filename(key),
                "label": value["human_target_label"],
                "human_modal_label": max(sorted(named), key=lambda k: named[k]),
                "weak_label": value["human_target_label"] in run.WEAK_LABELS,
            }
        )
    return out


# Ladder ---------------------------------------------------------------------


def test_calibration_set_is_balanced_outside_analysis_and_fixed():
    excluded = (ROOT / "research/reproduction/crema_d_selection.csv").read_text().split("\n")
    excluded = [line.split(",")[0] for line in excluded[1:] if line]
    clips = ladder.select_calibration(excluded)
    assert clips == ladder.select_calibration(excluded)
    assert len(clips) == 24 and not set(clips) & set(excluded)
    assert all(sum(c.split("_")[2] == e for c in clips) == 4 for e in ladder.EMOTIONS)
    assert len({c.split("_")[0] for c in clips}) == 24
    assert not set(clips) & ladder.DEFECTS


def test_sensitivity_fails_when_a_shift_moves_the_readout_as_much_as_a_new_clip():
    rng = np.random.default_rng(0)
    clips = rng.normal(size=(24, 21))
    good = ladder.image_sensitivity(clips, clips + 0.01 * rng.normal(size=clips.shape), rng.normal(size=21))
    assert good["pass"] and good["n_pairs"] == 276
    bad = ladder.image_sensitivity(clips, rng.normal(size=clips.shape), rng.normal(size=21))
    assert not bad["pass"]


def test_sensitivity_needs_real_clips_to_differ_from_grey():
    rng = np.random.default_rng(1)
    clips = rng.normal(size=(24, 21))
    grey_like = clips.mean(0) * 0 + clips[0]
    same = np.tile(clips[0], (24, 1))
    out = ladder.image_sensitivity(same, same, grey_like)
    assert not out["pass"]


def test_share_rule_is_inclusive():
    assert ladder.share_above([0.05, 0.04], 0.05) == 0.5


def test_shared_prompt_maximises_the_lowest_median():
    medians = {"a": {"p": 0.9, "q": 0.5, "r": None}, "b": {"p": 0.1, "q": 0.4, "r": 0.9}}
    assert ladder.choose_shared(medians) == "q"
    assert ladder.choose_shared({"a": {"p": None}, "b": {"p": 0.2}}) is None


def test_ladder_output_has_no_label_terms():
    source = (ROOT / "scripts/readout_ladder.py").read_text()
    assert "tabulatedVotes" not in source and "votes_A" not in source and "human_response_distribution" not in source


# Analysis -------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    return items()


def actors_of(data):
    return sorted({i["actor"] for i in data})


def hits(observer, data, seed=3):
    cells = an.Cells(rec.synthetic_rows(observer, data, seed))
    actors = actors_of(data)
    won = set()
    for route in ("single_face", "video_face"):
        family = [c for c in an.CONFIRMATORY if c.route == route]
        result = an.test_family(cells, family, actors, flip_args={"max_exact_actors": 0, "draws": 2047})
        won |= {k for k, r in result.items() if r["p_maxt"] < 0.05 and r["mean_change"] * r["direction"] > 0}
    return won


def test_every_contrast_uses_conditions_that_exist(data):
    for contrast in an.CONFIRMATORY:
        conditions = run.INPUTS[contrast.route]["conditions"]
        for side in (contrast.a, contrast.b, *contrast.other_fills):
            assert set(side) <= set(conditions), contrast.id


def test_aperture_stays_out_of_the_confirmatory_family():
    assert not [c for c in an.CONFIRMATORY if any("aperture" in s for s in c.a + c.b)]


def test_interaction_is_inside_one_actor_set(data):
    happy = {i["actor"] for i in data if i["label"] == "Happy"}
    afraid = {i["actor"] for i in data if i["label"] == "Afraid"}
    assert happy == afraid


def test_mouth_observer_is_recovered_and_eyes_observer_is_not_confused(data):
    won = hits("mouth_only", data)
    assert {"P1", "P2"} <= won and not {"P3", "P4"} & won
    won = hits("eyes_only", data)
    assert {"P3", "P4"} <= won and not {"P1", "P2"} & won


def test_human_like_observer_passes_all(data):
    assert hits("human_like", data) == rec.EXPECTED["human_like"]


def test_prior_only_observer_passes_none(data):
    assert hits("prior_only", data) == set()


def test_a_fall_in_the_acted_label_cannot_raise_another_label():
    names = rec.FAMILIES
    before = {k: float(i) * 0.1 for i, k in enumerate(names)}
    after = {**before, "Happy": before["Happy"] - 5.0}
    item = {"trial_id": "t", "actor": "A", "label": "Happy", "human_modal_label": "Happy", "weak_label": False}
    rows = [
        {**item, "input_name": "single_face", "condition": c, "status": "ok", "blind": {"family_logprob": f}}
        for c, f in (("original", before), ("mask_mouth_neutral", after))
    ]
    cells = an.Cells(rows)
    p7 = next(c for c in an.CONFIRMATORY if c.id == "P7")
    p1 = next(c for c in an.CONFIRMATORY if c.id == "P1")
    pair = (p7.a, p7.b)
    assert an.item_delta(cells, "t", p7, pair, "blind", "acted") == pytest.approx(0.0)
    assert an.item_delta(cells, "t", p1, (p1.a, p1.b), "blind", "acted") < -4


def test_mcq_reference_also_leaves_the_acted_label_out():
    options = ["Happy", "Neutral", "Sad"]
    row = lambda v: {"mcq": [{"options": options, "joint_logprob": v}]}  # noqa: E731
    assert an.mcq_score(row([-5.0, -1.0, -2.0]), "Neutral", "Happy") == pytest.approx(
        an.mcq_score(row([-1.0, -1.0, -2.0]), "Neutral", "Happy")
    )


def test_failed_cells_are_counted_not_used(data):
    rows = rec.synthetic_rows("prior_only", data[:20], 0)
    rows[0].update(status="failed", reason="eye_closed")
    assert len(an.Cells(rows).failed) == 1


def test_verdict_table():
    row = {"mean_change": -1.0, "direction": -1, "p_maxt": 0.01, "fills_agree": True, "p_maxt_two_sided": 0.02}
    assert an.verdict(row, True, False) == "matches_prediction"
    assert an.verdict(row, False, False) == "inconclusive"
    assert an.verdict({**row, "fills_agree": False, "p_maxt": 0.01}, True, False) == "inconclusive"
    opposite = {"mean_change": 1.0, "direction": -1, "p_maxt": 0.9, "fills_agree": None, "p_maxt_two_sided": 0.01}
    assert an.verdict(opposite, True, False) == "opposes_prediction"
    flat = {"mean_change": 0.0, "direction": -1, "p_maxt": 0.9, "fills_agree": None, "p_maxt_two_sided": 0.9}
    assert an.verdict(flat, True, True) == "null"
    assert an.verdict(flat, True, False) == "inconclusive"


def test_equivalence_needs_a_narrow_interval():
    assert an.equivalence_pass(np.array([0.01, -0.01, 0.0, 0.02, -0.02]), 0.1)
    assert not an.equivalence_pass(np.array([1.0, -1.0, 0.5, -0.5, 0.0]), 0.1)


def test_gate_b_passes_a_face_reader_and_fails_a_prior_observer(data):
    partners = run_partners(data)
    free = dict(zip(rec.FAMILIES, np.zeros(21).tolist()))
    good = an.gate_b(an.Cells(rec.synthetic_rows("human_like", data, 1)), "single_face", partners, free)
    bad = an.gate_b(an.Cells(rec.synthetic_rows("prior_only", data, 1)), "single_face", partners, free)
    assert good["B4"] and good["B5"]
    assert not bad["B4"] and not bad["pass"]


def run_partners(data):
    from paper_v2.option_score import vision_mismatch_partners

    return vision_mismatch_partners(
        [i["trial_id"] for i in data], [i["label"] for i in data], [i["actor"] for i in data], seed=42
    )


def test_analysis_refuses_until_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--cells", "a", "--run_report", "b", "--output", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="not frozen"):
        an.main()


# Predictor ------------------------------------------------------------------


def test_predict_renormalises_and_uses_slopes():
    coefficients = np.zeros((2, 2 + 33))
    coefficients[0, 2] = 4.0
    au = np.zeros((2, 33))
    au[1, 0] = 1.0
    p = hrp.predict(au, np.zeros(2), coefficients)
    assert p.sum(1) == pytest.approx([1, 1])
    assert p[0, 0] == pytest.approx(0.5) and p[1, 0] > 0.6


def test_scores_reward_a_better_model():
    y = np.array([0, 1] * 30)
    stim = np.array(sum(([f"s{i}"] * 4 for i in range(15)), []))
    good = np.eye(2)[y] * 0.9 + 0.05
    flat = np.full((60, 2), 0.5)
    base = np.array([0.5, 0.5])
    assert hrp.scores(good, y, base, stim, 4)["log_loss"] < hrp.scores(flat, y, base, stim, 4)["log_loss"]
