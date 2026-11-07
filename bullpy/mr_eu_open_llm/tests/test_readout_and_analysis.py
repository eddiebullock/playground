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


PASSED = {k: True for k in an.GATE_KEYS}


def test_verdict_table():
    row = {"mean_change": -1.0, "direction": -1, "p_maxt": 0.01, "p_maxt_two_sided": 0.02, "fills_agree": True}
    assert an.verdict(row, PASSED, False) == "matches_prediction"
    assert an.verdict(row, {**PASSED, "gate_b": False}, False) == "inconclusive"
    assert an.verdict({**row, "fills_agree": False}, PASSED, False) == "inconclusive"
    opposite = {**row, "mean_change": 1.0, "p_maxt": 0.9}
    assert an.verdict(opposite, PASSED, False) == "opposes_prediction"
    flat = {**row, "mean_change": -0.01, "p_maxt": 0.5, "p_maxt_two_sided": 0.6}
    assert an.verdict(flat, PASSED, True) == "null"
    assert an.verdict(flat, PASSED, False) == "inconclusive"


@pytest.mark.parametrize("gate", an.GATE_KEYS)
def test_every_gate_is_required_and_missing_means_not_passed(gate):
    row = {"mean_change": -1.0, "direction": -1, "p_maxt": 0.01, "p_maxt_two_sided": 0.02, "fills_agree": None}
    assert an.verdict(row, {**PASSED, gate: False}, True) == "inconclusive"
    assert an.verdict(row, {k: v for k, v in PASSED.items() if k != gate}, True) == "inconclusive"


def test_a_significant_contrast_is_never_a_null():
    row = {"mean_change": -0.01, "direction": -1, "p_maxt": 0.01, "p_maxt_two_sided": 0.02, "fills_agree": False}
    assert an.verdict(row, PASSED, True) == "inconclusive"


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


def test_b5_is_not_b4_and_fails_a_prior_observer(data):
    partners = run_partners(data)
    free = dict(zip(rec.FAMILIES, np.zeros(21).tolist()))
    good = an.gate_b(an.Cells(rec.synthetic_rows("human_like", data, 1)), "single_face", partners, free)
    bad = an.gate_b(an.Cells(rec.synthetic_rows("prior_only", data, 1)), "single_face", partners, free)
    assert good["B5_mean"] != pytest.approx(good["B4_mean"])
    assert good["B5_p"] != good["B4_p"] or good["B5_ci"] != good["B4_ci"]
    assert not bad["B5"]


def test_b4_repeats_are_reported_and_do_not_decide_the_gate(data):
    partners = run_partners(data)
    free = dict(zip(rec.FAMILIES, np.zeros(21).tolist()))
    gate = an.gate_b(an.Cells(rec.synthetic_rows("human_like", data, 1)), "single_face", partners, free)
    assert gate["B4_without_flagged_items"]["n_items"] < gate["B4_against_human_mode"]["n_items"]
    assert gate["B4_against_human_mode"]["n_items"] <= 243


def blind_row(trial, actor, label, condition, scores, route="single_face", **extra):
    return {
        "trial_id": trial, "actor": actor, "label": label, "human_modal_label": label, "input_name": route,
        "condition": condition, "status": "ok", "blind": {"family_logprob": scores, "lexicon_mass": 0.5}, **extra,
    }


def test_p6_takes_the_fill_rule_on_both_sides():
    """The other fills must replace the eyes mask too, or the rule tests something else."""
    rows = []
    for n, (trial, label) in enumerate((("h", "Happy"), ("a", "Afraid"))):
        target = {"Happy": 0.0, "Afraid": 0.0, "Neutral": 0.0}
        base = {**target, label: 2.0}
        for condition in an.CONFIRMATORY[5].a + an.CONFIRMATORY[5].b + ("original",):
            rows.append(blind_row(trial, "C", label, condition, base))
        for region in ("mouth", "eyes"):
            for fill, drop in (("first", 1.0), ("blur", 1.0)):
                wrong = {"Happy": {"mouth": 0.0, "eyes": drop}, "Afraid": {"mouth": drop, "eyes": 0.0}}[label][region]
                rows.append(blind_row(trial, "C", label, f"mask_{region}_{fill}", {**target, label: 2.0 - wrong}))
    p6 = next(c for c in an.CONFIRMATORY if c.id == "P6")
    assert p6.other_fills_b
    effect = an.actor_effects(an.Cells(rows), p6, ["C"], a=p6.other_fills[0], b=p6.other_fills_b[0])
    assert effect[0] == pytest.approx(an.actor_effects(an.Cells(rows), p6, ["C"], a=p6.other_fills[1], b=p6.other_fills_b[1])[0])
    assert effect[0] * p6.direction < 0


def test_contrast_intervals_are_reported(data):
    actors = actors_of(data)
    result = an.analyse_route(an.Cells([dict(r, mcq=[]) for r in rec.synthetic_rows("human_like", data, 2)]), "single_face", actors, PASSED, 0.1, flip_args={"max_exact_actors": 0, "draws": 255})
    row = result["primary_blind_acted"]["P1"]
    assert row["ci95"][0] < row["mean_change"] < row["ci95"][1]


def null_setup(data):
    """An observer that ignores every edit, with a dose control that moves it on request."""
    rows = rec.synthetic_rows("prior_only", data, 4)
    originals = {(r["trial_id"], r["input_name"]): r for r in rows if r["condition"] == "original"}
    out = []
    for row in rows:
        base = originals[(row["trial_id"], row["input_name"])]
        scores = dict(base["blind"]["family_logprob"])
        if row["condition"].endswith("_blur") or row["condition"] in ("sf_low", "sf_high", "static_apex"):
            scores[run.OPTION_TO_FAMILY[row["label"]]] -= 3.0
        out.append({**row, "blind": {**row["blind"], "family_logprob": scores}, "edit": {"mask_fraction": 0.05}, "mcq": []})
    return an.Cells(out)


def test_null_needs_all_six_conditions_and_a_cue_with_a_joint_ablation(data):
    cells = null_setup(data)
    actors = actors_of(data)
    tokens = {t: 2000.0 for t in cells.items}
    result = an.analyse_route(cells, "single_face", actors, PASSED, 0.1, frame_tokens=tokens, flip_args={"max_exact_actors": 0, "draws": 255})
    rows = result["primary_blind_acted"]
    assert rows["P1"]["null_rule"]["null_ok"] and rows["P1"]["verdict"] == "null"
    for name in ("P5", "P6", "P9", "P10"):
        assert not rows[name]["null_rule"]["can_reach_null"] and rows[name]["verdict"] == "inconclusive"
    assert rows["P5"]["descriptive_label"] == "single_region_null"
    assert rows["P6"]["descriptive_label"] is None and rows["P1"]["descriptive_label"] is None


def test_order_cues_reach_null_only_where_frame_order_is_verified(data):
    cells = null_setup(data)
    actors = actors_of(data)
    flips = {"max_exact_actors": 0, "draws": 255}
    gemma = an.analyse_route(cells, "video_face", actors, PASSED, 0.1, flip_args=flips, frame_order_verified=True)["primary_blind_acted"]
    qwen = an.analyse_route(cells, "video_face", actors, PASSED, 0.1, flip_args=flips, frame_order_verified=False)["primary_blind_acted"]
    for name in ("P12", "P13"):
        assert gemma[name]["null_rule"]["joint_inapplicable"] and gemma[name]["verdict"] == "null"
        assert not qwen[name]["null_rule"]["can_reach_null"] and qwen[name]["verdict"] == "inconclusive"
    for name in ("P14a", "P14b", "P15a", "P15b"):
        assert gemma[name]["verdict"] == "inconclusive" and not gemma[name]["null_rule"]["can_reach_null"]


def test_counts_of_contrasts_that_cannot_reach_null():
    capped = [c.id for c in an.CONFIRMATORY if c.joint is None and not c.joint_inapplicable]
    assert capped == ["P5", "P6", "P9", "P10", "P14a", "P14b", "P15a", "P15b"]
    assert len(an.CONFIRMATORY) == 16
    assert len({c.rstrip("ab") for c in capped}) == 6


def test_frame_order_needs_every_row_to_show_it():
    shown = {"delivery": [{"video_face": {"pixels_arrive_in_sent_order": True}}]}
    absent = {"delivery": [{"video_face": {"pixels_change_under_shuffle": True}}]}
    assert an.frame_order_verified([shown, shown])
    assert not an.frame_order_verified([shown, absent]) and not an.frame_order_verified([])


def test_a_blur_with_no_signed_effect_is_not_a_dose(data):
    """Large item changes that cancel within actors are noise, not a working dose."""
    p1 = an.CONFIRMATORY[0]
    rows = []
    for actor in ("C", "E"):
        for n, sign in enumerate((1.0, -1.0)):
            trial = f"{actor}{n}"
            for condition, shift in (("original", 0.0), ("mask_mouth_blur", 3.0 * sign)):
                rows.append(blind_row(trial, actor, "Happy", condition, {"Happy": shift, "Neutral": 0.0, "Sad": 0.0}))
    assert an.dose_change(an.Cells(rows), p1, "mask_mouth_blur", ["C", "E"]) == pytest.approx(0.0)
    row = {"mean_change": 0.0, "ci95": [-0.01, 0.01], "null_equivalence": True}
    checks = an.null_conditions(an.Cells(rows), p1, row, ["C", "E"], PASSED, 0.1, {f"{a}{n}": 2000.0 for a in "CE" for n in (0, 1)})
    assert not checks["c3_dose"]


def test_each_null_condition_blocks_on_its_own(data):
    cells = null_setup(data)
    actors = actors_of(data)
    tokens = {t: 2000.0 for t in cells.items}
    p1 = an.CONFIRMATORY[0]
    row = {"mean_change": 0.0, "ci95": [-0.01, 0.01], "null_equivalence": True}
    args = (cells, p1, row, actors)
    assert an.null_conditions(*args, PASSED, 0.1, tokens)["null_ok"]
    assert not an.null_conditions(*args, {**PASSED, "gate_b": False}, 0.1, tokens)["c1_rq0_gates_gate_b"]
    assert not an.null_conditions(*args, {**PASSED, "delivery": False}, 0.1, tokens)["c2_delivery_and_region"]
    assert not an.null_conditions(*args, PASSED, 0.1, {t: 1.0 for t in tokens})["c2_delivery_and_region"]
    assert not an.null_conditions(*args, PASSED, 0.1, None)["c2_delivery_and_region"]
    assert not an.null_conditions(*args, PASSED, 50.0, tokens)["c3_dose"]
    assert not an.null_conditions(*args, PASSED, None, tokens)["null_ok"]
    assert not an.null_conditions(cells, p1, {**row, "null_equivalence": False}, actors, PASSED, 0.1, tokens)["c5_equivalence"]
    assert not an.null_conditions(cells, p1, {**row, "ci95": None}, actors, PASSED, 0.1, tokens)["c6_divergence_reported"]
    joint_moves = an.Cells([
        {**r, "blind": {**r["blind"], "family_logprob": {**r["blind"]["family_logprob"], "Happy": r["blind"]["family_logprob"]["Happy"] - 5}}}
        if r["condition"] == "mask_eyesmouth_neutral" else r
        for r in cells.rows
    ])
    assert not an.null_conditions(joint_moves, p1, row, actors, PASSED, 0.1, tokens)["c4_joint_ablation"]


def test_sensitivity_filters_remove_the_named_cells():
    keep = {"edit": {"fill": "neutral", "align_flagged": False, "neutral_donor": {"self": False}}}
    assert an.align_flagged({"edit": {"align_flagged": True}})
    assert an.donor_is_sent_frame({"edit": {"donor_is_sent_frame": True}})
    assert an.own_neutral_donor({"edit": {"neutral_donor": {"self": True}}})
    for check in (an.align_flagged, an.donor_is_sent_frame, an.own_neutral_donor):
        assert not check(keep) and not check({"status": "failed"})
    scores = {"Happy": 0.0, "Neutral": 0.0}
    rows = [blind_row("t", "A", "Happy", "original", scores), blind_row("t", "A", "Happy", "mask_mouth_neutral", scores, edit={"align_flagged": True})]
    assert ("t", "single_face", "mask_mouth_neutral") not in an.Cells(rows).without(an.align_flagged).ok


def test_smoke_items_are_hardcoded_and_exist(data):
    ids = {i["trial_id"] for i in data}
    assert len(an.SMOKE_ITEMS) == 3 and set(an.SMOKE_ITEMS) <= ids


def test_canonical_word_replaces_the_family_sum():
    words = run.lexicon_words()
    logprobs = [-9.0] * len(words)
    logprobs[words.index(("Happy", "happy"))] = -1.0
    logprobs[words.index(("Happy", "joy"))] = -0.5
    row = {"blind": {"word_logprob": logprobs, "family_logprob": {"Happy": -0.2}}}
    assert an.canonical_rows([row])[0]["blind"]["family_logprob"]["Happy"] == -1.0
    assert an.canonical_rows([{"status": "failed"}])[0] == {"status": "failed"}


def test_token_floor_and_frame_tokens():
    geometry = {
        "small": {"face_box": [0, 0, 400, 400], "face_width_px": 100.0, "width": 800, "height": 400},
        "large": {"face_box": [0, 0, 400, 400], "face_width_px": 300.0, "width": 800, "height": 400},
    }
    items = {"small": {}, "large": {}, "absent": {}}
    assert sorted(an.below_token_floor(items, 16.0, geometry)) == ["absent", "small"]
    assert len(an.below_token_floor(items, None, geometry)) == 3
    assert an.frame_tokens_by_clip(items, 16.0, geometry)["small"] == pytest.approx(16.0**2 * 2)


def test_rq0_follows_the_a1_and_a3_table():
    assert an.rq0_status(True, True, "video_face")
    assert an.rq0_status(False, True, "single_face")
    assert not an.rq0_status(False, True, "video_face")
    assert not an.rq0_status(True, False, "single_face")
    assert not an.rq0_status(None, True, "single_face")


def test_a3_default_path_is_the_one_a3_writes_and_only_pinned_bytes_count(tmp_path):
    from scripts import face_input_validity as fiv

    assert an.A3_OUTPUT_DIR == fiv.OUTCOME_DIR
    assert "/ \"" + an.A3_SUMMARY_NAME + "\"" in (ROOT / "scripts/face_input_validity.py").read_text()
    committed = ROOT / "research/reproduction"
    assert an.a3_summary(committed, tmp_path)["gemma4"]["vision_dependent"] is True
    frozen = (committed / an.A3_SUMMARY_NAME).read_bytes()
    (tmp_path / an.A3_SUMMARY_NAME).write_bytes(frozen)
    assert an.a3_summary(tmp_path / "absent", tmp_path)["gemma4"]["vision_dependent"] is True
    (tmp_path / an.A3_SUMMARY_NAME).write_bytes(frozen + b" ")
    assert an.a3_summary(tmp_path, tmp_path) == {}


def test_outside_gates_read_the_files_and_missing_means_not_passed(tmp_path):
    source = ROOT / "research/reproduction"
    assert an.outside_gates(source, "molmo2", "single_face")["g2_g4"] is False
    gemma = an.outside_gates(source, "gemma4", "single_face")
    assert gemma["g2_g4"] is True and gemma["delivery_file"] is True
    assert an.outside_gates(tmp_path, "gemma4", "single_face", tmp_path) == {
        "rq0": False, "a1": None, "a3": None, "g2_g4": False, "delivery_file": False,
    }


DELIVERED = {
    "gate_2_every_frame_changes_output": True,
    "gate_3_logged_indices_are_the_frames_sent": True,
    "gate_4_ids_unchanged_within_clip": True,
    "pixels_change_under_shuffle": True,
}


def test_delivery_is_read_from_every_shard_and_gates_order():
    good = {"delivery": [{"video_face": DELIVERED}]}
    assert an.delivery_passed([good, good], "video_face")
    assert not an.delivery_passed([], "video_face")
    assert not an.delivery_passed([good, {"delivery": [{"video_face": {"failed": "x"}}]}], "video_face")
    wrong_order = {"delivery": [{"video_face": {**DELIVERED, "pixels_arrive_in_sent_order": False}}]}
    assert not an.delivery_passed([good, wrong_order], "video_face")
    stuck = {"delivery": [{"video_face": {**DELIVERED, "pixels_change_under_shuffle": False}}]}
    assert not an.delivery_passed([stuck], "video_face")
    unchecked = {k: v for k, v in DELIVERED.items() if k != "pixels_change_under_shuffle"}
    assert not an.delivery_passed([{"delivery": [{"video_face": unchecked}]}], "video_face")


def test_sham_must_send_the_same_pixels():
    scores = {"Happy": 0.0}
    same = [blind_row("t", "A", "Happy", c, scores, media_sha256=["x"]) for c in ("original", "sham_self_fill")]
    assert an.sham_passed(an.Cells(same), "single_face")
    differ = [same[0], {**same[1], "media_sha256": ["y"]}]
    assert not an.sham_passed(an.Cells(differ), "single_face")
    assert not an.sham_passed(an.Cells(same[:1]), "single_face")
    assert an.sham_passed(an.Cells(same[:1]), "video_face")


def test_descriptive_tables_flip_rates_and_failures(data):
    rows = rec.synthetic_rows("human_like", data, 5)
    rows[0] = {**rows[0], "status": "failed", "reason": "eye_closed"}
    cells = an.Cells(rows)
    actors = actors_of(data)
    table = an.descriptive_changes(cells, "single_face", actors)
    assert {"eye_crop", "eye_window"} <= set(table) and "original" not in table
    assert set(an.BASIC_LABELS) <= set(table["eye_crop"]) and {"basic", "complex"} <= set(table["eye_crop"])
    assert {"reverse", "static_apex", "shuffle_a"} <= set(an.descriptive_changes(cells, "video_face", actors))
    assert 0 <= an.flip_rates(cells, "video_face")["reverse"]["flip_rate"] <= 1
    assert 0 <= an.shuffle_nuisance(cells) <= 1
    assert sum(an.failure_table(cells).values()) == 1


def shard_files(tmp_path, data, hash_value="h", n_shards=3, drop=None, duplicate=False):
    rows = [dict(r, design_sha256=hash_value) for r in rec.synthetic_rows("human_like", data, 6) if r["input_name"] == "single_face"]
    trials = sorted({r["trial_id"] for r in rows})
    cells, reports = [], []
    for shard in range(n_shards):
        part = [r for r in rows if trials.index(r["trial_id"]) % n_shards == shard]
        if drop is not None and shard == drop:
            continue
        if duplicate and shard == 0:
            part = part + part[:1]
        path = tmp_path / f"cells_{shard}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in part))
        report = tmp_path / f"run_{shard}.json"
        report.write_text(json.dumps({"model": "gemma4", "smoke": False, "design_sha256": hash_value, "shard": shard, "n_shards": n_shards}))
        cells.append(path)
        reports.append(report)
    return cells, reports


def test_all_shards_load_and_any_gap_is_refused(tmp_path, data):
    cells, reports = shard_files(tmp_path, data)
    rows, loaded = an.load_shards(cells, reports, "gemma4", "h")
    assert len({r["trial_id"] for r in rows}) == 243 and len(loaded) == 3
    with pytest.raises(SystemExit, match="hashes"):
        an.load_shards(cells, reports, "gemma4", "other")
    with pytest.raises(SystemExit, match="full run"):
        an.load_shards(cells, reports, "qwen3vl", "h")
    with pytest.raises(SystemExit, match="Expected 99"):
        an.load_shards(cells, reports, "gemma4", "h", n_items=99)
    gap, gap_reports = shard_files(tmp_path, data, drop=1)
    with pytest.raises(SystemExit, match="complete set"):
        an.load_shards(gap, gap_reports, "gemma4", "h")
    twice, twice_reports = shard_files(tmp_path, data, duplicate=True)
    with pytest.raises(SystemExit, match="Duplicate"):
        an.load_shards(twice, twice_reports, "gemma4", "h")


def test_a_lost_cell_is_refused(tmp_path, data):
    cells, reports = shard_files(tmp_path, data)
    lines = cells[0].read_text().splitlines()
    cells[0].write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(SystemExit, match="missing"):
        an.load_shards(cells, reports, "gemma4", "h")


def test_analysis_runs_end_to_end_on_three_shards(tmp_path, data, monkeypatch):
    """Every output section, from shard files, on made-up scores."""
    rows = rec.synthetic_rows("human_like", data, 7)
    rows = [r for r in rows if r["input_name"] in ("single_face", "video_face")]
    words = run.lexicon_words()
    design = run.design_hash()
    trials = sorted({r["trial_id"] for r in rows})
    for row in rows:
        row.update({
            "design_sha256": design,
            "mcq": [],
            "media_sha256": ["same"],
            "edit": {"mask_fraction": 0.05},
            "position": 0,
        })
        row["blind"]["word_logprob"] = [row["blind"]["family_logprob"][f] for f, _ in words]
    delivery = {r: {**DELIVERED, "repeat_forward_max_abs": 0.0, "tokens_across_view": 48.0} for r in ("single_face", "video_face")}
    free = dict(zip(rec.FAMILIES, np.zeros(21).tolist()))
    baselines = [{"input_name": r, "condition": "image_free", "family_logprob": free} for r in delivery]
    cells, reports = [], []
    for shard in range(3):
        cell = tmp_path / f"cells_{shard}.jsonl"
        cell.write_text("".join(json.dumps(r) + "\n" for r in rows if trials.index(r["trial_id"]) % 3 == shard))
        report = tmp_path / f"run_{shard}.json"
        report.write_text(json.dumps({"model": "gemma4", "smoke": False, "design_sha256": design, "shard": shard, "n_shards": 3, "delivery": [delivery], "baselines": baselines}))
        cells.append(str(cell))
        reports.append(str(report))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"items": {t: {"face_box": [0, 0, 400, 400], "face_width_px": 200.0, "width": 800, "height": 400} for t in trials}}))
    monkeypatch.setattr(run, "check_geometry", lambda path: None)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "x", "--model", "gemma4", "--cells", *cells, "--run_report", *reports, "--geometry", str(geometry),
        "--reproduction", str(tmp_path), "--output", str(out), "--a2_frozen",
    ])
    monkeypatch.setattr(an, "a3_summary", lambda *a, **k: {})
    an.main()
    result = json.loads(out.read_text())
    assert result["design_sha256"] == design
    face = result["single_face"]
    assert set(face["sensitivity"]) == {
        "without_smoke_items", "without_weak_labels", "without_mode_not_target", "clips_at_8_tokens_or_more",
        "without_align_flagged", "without_donor_is_sent_frame", "without_own_neutral_donor", "canonical_word_only",
    }
    assert face["gates"]["rq0"] is False  # no RQ0 files here, so every verdict is inconclusive
    assert all(r["verdict"] == "inconclusive" for r in face["results"]["primary_blind_acted"].values())


def run_partners(data):
    from paper_v2.option_score import vision_mismatch_partners

    return vision_mismatch_partners(
        [i["trial_id"] for i in data], [i["label"] for i in data], [i["actor"] for i in data], seed=42
    )


def test_analysis_refuses_until_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--model", "gemma4", "--cells", "a", "--run_report", "b", "--output", str(tmp_path / "o")])
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
