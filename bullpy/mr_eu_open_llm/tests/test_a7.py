"""Amendment A7: design hash, Stage 0 checks, region test, guards, estimands and verdicts. CPU only.

The draft's "End to end" list (research/PREREG_A7_DRAFT.md, Recovery) is covered here: the
identity phi_A + phi_B = TE, a verdict from the real pinned gate paths, Gate B4 against the
block terms, no block reported as complete, and the audit's tests (a) to (e).
"""

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import a7_analyse as a7  # noqa: E402
from scripts import a7_design  # noqa: E402
from scripts import a7_recovery as rc  # noqa: E402
from scripts import a7_stage0 as s0  # noqa: E402
from scripts import analyse_stimulus_interventions as an  # noqa: E402
from scripts.a7_cuts import SiteRefused, check_site  # noqa: E402

D = a7.D
N = s0.N_CLIPS
W = s0.N_WORDS
TOL = s0.TOL


# ---------------------------------------------------------------------------
# Design block and pins
# ---------------------------------------------------------------------------


def test_design_hash_is_pinned_stated_and_post_audit():
    assert a7_design.block_hash() == a7_design.DESIGN_SHA256
    assert a7_design.stated_hash() == a7_design.DESIGN_SHA256
    assert a7_design.DESIGN_SHA256.startswith("9e40cfcf")
    assert a7_design.load_design()["amendment"] == "A7"


def test_design_refuses_an_edited_block(tmp_path):
    text = a7_design.DRAFT.read_text(encoding="utf-8")
    edited = tmp_path / "draft.md"
    edited.write_text(text.replace('"leak_limit": 0.05', '"leak_limit": 0.06'), encoding="utf-8")
    with pytest.raises(SystemExit):
        a7_design.load_design(edited)


def test_design_refuses_a_restated_hash_that_differs(tmp_path):
    text = a7_design.DRAFT.read_text(encoding="utf-8")
    other = "0" * 64
    edited = tmp_path / "draft.md"
    edited.write_text(text.replace(f"Current `design_sha256` of the block below: `{a7_design.DESIGN_SHA256}`",
                                   f"Current `design_sha256` of the block below: `{other}`"), encoding="utf-8")
    with pytest.raises(SystemExit):
        a7_design.load_design(edited)


def test_design_refuses_a_draft_with_no_current_hash(tmp_path):
    text = a7_design.DRAFT.read_text(encoding="utf-8")
    edited = tmp_path / "draft.md"
    edited.write_text(text.replace("Current `design_sha256` of the block below:", "Hash:"), encoding="utf-8")
    with pytest.raises(SystemExit):
        a7_design.load_design(edited)


def test_pins_match_files_on_disk_and_missing_files_refuse(tmp_path):
    a7_design.check_pins(D)
    with pytest.raises(SystemExit):
        a7_design.check_pins(D, root=tmp_path)


def test_code_pins_required_means_not_frozen_before_the_freeze():
    with pytest.raises(SystemExit):
        a7_design.check_code_pins(require=True)
    assert a7_design.check_code_pins(require=False) == {}


# ---------------------------------------------------------------------------
# Complete cut: C1 to C4, the 1e-3 rule with no partial credit (audit test a)
# ---------------------------------------------------------------------------


def base_cells(seed=0):
    rng = np.random.default_rng(seed)
    native = rng.normal(-6, 1, (N, W))
    free = rng.normal(-6, 1, (1, W)).repeat(N, 0)
    return {"native": native, "repeat": native.copy(), "image_free": free, "exit_cut": free.copy(),
            "all_held": native.copy(), "hold_A": native[[s0.partner(i) for i in range(N)]] * 0.5 + native * 0.5,
            "hold_B": native[[s0.partner(i) for i in range(N)]] * 0.5 + native * 0.5}


def test_all_checks_pass_on_exact_equality():
    table = s0.stage0_table(base_cells())
    assert table["status"] == "complete" and table["exit_complete"] and table["complete_cuts"] == ["exit"]
    assert all(table["checks"][k]["worst_abs_nats"] == 0.0 for k in ("C1", "C2", "C3", "C4"))
    assert s0.arms(table) == {"route": True, "token": False}


@pytest.mark.parametrize("key,check", [("repeat", "C1"), ("exit_cut", "C2"), ("all_held", "C4")])
def test_one_word_on_one_clip_over_the_limit_fails_the_site(key, check):
    cells = base_cells()
    cells[key] = cells[key].copy()
    cells[key][13, 77] += TOL * 1.01
    table = s0.stage0_table(cells)
    assert table["checks"][check]["pass"] is False
    assert table["checks"][check]["n_values_over_tolerance"] == (1 if check != "C2" else 1)
    assert (table["checks"][check]["worst_clip_index"], table["checks"][check]["worst_word_index"]) == (13, 77)
    if check == "C1":
        assert table["status"] == "stopped_c1" and "C2" not in table["checks"]
    if check in ("C1", "C2"):
        assert table["complete_cuts"] == [] and s0.arms(table) == {"route": False, "token": False}
    if check == "C4":
        assert table["exit_complete"] and not table["blocks_exhaust_exit"] and s0.arms(table)["route"] is False


def test_c3_compares_each_clip_with_its_partner_including_last_with_first():
    cells = base_cells()
    cells["exit_cut"] = cells["exit_cut"].copy()
    cells["exit_cut"][0, 5] += TOL * 2  # also breaks C2, but C3 must see it through the pair (23, 0)
    result = s0.c3(cells["exit_cut"])
    assert result["pass"] is False and result["worst_clip_index"] in (0, N - 1)


def test_the_limit_itself_passes_and_nothing_partial_is_credited():
    cells = base_cells()
    cells["repeat"] = cells["native"] + TOL * 0.999
    assert s0.c1(cells["native"], cells["repeat"])["pass"] is True
    cells["repeat"] = cells["native"].copy()
    cells["repeat"][:, :] += TOL * 0.5
    cells["repeat"][0, 0] = cells["native"][0, 0] + 1.0
    assert s0.c1(cells["native"], cells["repeat"])["pass"] is False


@pytest.mark.parametrize("bad", [None, "missing_clip", "nan", "wrong_words"])
def test_missing_or_invalid_logprobs_fail_safe(bad):
    cells = base_cells()
    if bad is None:
        cells["exit_cut"] = None
    elif bad == "missing_clip":
        cells["exit_cut"] = cells["exit_cut"][:-1]
    elif bad == "nan":
        cells["exit_cut"] = cells["exit_cut"].copy()
        cells["exit_cut"][3, 3] = np.nan
    else:
        cells["exit_cut"] = cells["exit_cut"][:, :-1]
    table = s0.stage0_table(cells)
    assert table["checks"]["C2"]["pass"] is False and table["checks"]["C2"]["worst_abs_nats"] is None
    assert table["complete_cuts"] == [] and s0.arms(table) == {"route": False, "token": False}


def test_a_stored_flag_cannot_admit_an_arm():
    table = s0.stage0_table(base_cells())
    forged = copy.deepcopy(table)
    forged["checks"]["C2"]["pass"] = False
    forged["exit_complete"] = True
    forged["arms"] = {"route": True, "token": True}
    assert s0.arms(forged) == {"route": False, "token": False}
    worse = copy.deepcopy(table)
    worse["checks"]["C3"]["worst_abs_nats"] = TOL * 1.01  # a pass flag beside a worst value over the limit
    assert s0.arms(worse) == {"route": False, "token": False} and not s0.exit_complete(worse)
    assert s0.arms(None) == {"route": False, "token": False}
    assert s0.arms({}) == {"route": False, "token": False}


def test_a_block_is_never_reported_as_complete():
    table = s0.stage0_table(base_cells())
    assert table["complete_cuts"] == ["exit"]
    text = json.dumps(table)
    assert '"complete_cuts": ["exit"]' in text
    assert "blocks_exhaust_exit" in table and "route_complete" not in table


def test_load_table_refuses_a_missing_file_or_another_design(tmp_path):
    assert s0.load_table(tmp_path / "none.json") is None
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**s0.stage0_table(base_cells()), "design_sha256": "0" * 64}))
    assert s0.load_table(other) is None


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------


def test_c5_is_a_ratio_of_means_and_missing_means_dead():
    cells = base_cells()
    out = s0.c5(cells["native"], {"A": cells["hold_A"], "B": cells["native"]})
    assert out["blocks"]["A"]["live"] is True and out["blocks"]["B"]["ratio"] == 0.0 and out["blocks"]["B"]["live"] is False
    clr = s0.family_clr(cells["native"])
    hold = s0.family_clr(cells["hold_A"])
    j = [s0.partner(i) for i in range(N)]
    expected = s0.mean_rms(hold, clr) / s0.mean_rms(clr[j], clr)
    assert out["blocks"]["A"]["ratio"] == pytest.approx(expected)
    missing = s0.c5(cells["native"], {"A": None, "B": cells["hold_B"]})
    assert missing["blocks"]["A"]["live"] is False
    table = s0.stage0_table({**cells, "hold_B": cells["native"]})
    assert s0.dead_blocks(table) == ["B"] and s0.arms(table)["route"] is True  # C5 never decides an arm


def test_dead_blocks_without_stage0_evidence_are_dead():
    assert s0.dead_blocks(None) == ["A", "B"]


# ---------------------------------------------------------------------------
# Region test
# ---------------------------------------------------------------------------


def region_inputs(rho=0.0, seed=1):
    rng = np.random.default_rng(seed)
    native = rng.normal(-6, 1, (N, W))
    edited = native + rng.normal(0, 0.5, (N, W))
    holds = {b: native + rho * (edited - native) for b in D["token_buffers"]}
    side = 16
    masks = [[r * side + c for r in range(6, 10) for c in range(6, 10)] for _ in range(N)]
    sets = s0.token_sets(masks, side)
    token_map = {"pass": True, "hit_rate": 1.0}
    return token_map, sets, side * side, native, edited, holds, s0.c5_denominator(native)


def test_region_test_passes_with_no_leak_and_picks_the_smallest_buffer():
    out = s0.region_test(*region_inputs(0.0))
    assert out["pass"] is True and out["b"] == 0


def test_region_test_fails_on_leak_over_the_limit():
    out = s0.region_test(*region_inputs(0.3))
    assert out["pass"] is False and out["reason"] == "leak_over_limit"
    assert out["leaks"]["0"]["upper"] > D["leak_limit"]


def test_region_test_failures_are_fail_safe():
    token_map, sets, n_tokens, native, edited, holds, denom = region_inputs(0.0)
    assert s0.region_test({"pass": False}, sets, n_tokens, native, edited, holds, denom)["reason"] == "token_map_failed"
    assert s0.region_test(None, sets, n_tokens, native, edited, holds, denom)["reason"] == "token_map_failed"
    assert s0.region_test(token_map, sets, n_tokens, native, None, holds, denom)["reason"] == "edit_failed"
    assert s0.region_test(token_map, sets, n_tokens, native, edited, holds, denom, edit_failed=True)["reason"] == "edit_failed"
    weak = native + 1e-6
    assert s0.region_test(token_map, sets, n_tokens, native, weak, holds, denom)["reason"] == "edit_too_weak"
    assert s0.region_test(token_map, sets, n_tokens, native, edited, holds, None)["reason"] == "edit_too_weak"
    assert s0.region_test(token_map, sets, n_tokens, native, edited, {}, denom)["reason"] == "leak_over_limit"
    table = s0.stage0_table(base_cells(), region={"pass": False, "b": None})
    assert s0.arms(table)["token"] is False


def test_share_cap_applies_on_every_clip():
    side = 16
    masks = [[r * side + c for r in range(6, 10) for c in range(6, 10)] for _ in range(N)]
    masks[5] = list(range(side * side // 2 + 1))  # one clip over half the tokens at b = 0
    sets = s0.token_sets(masks, side)
    assert s0.admissible(sets, side * side) == []


def test_token_map_needs_all_96_trials_and_the_hit_rate():
    side = 16
    expected = [s0.expected_token(c, side) for c in D["token_map_centres_px"]] * N
    assert s0.token_map_check(expected, expected, side)["pass"] is True
    near = [e + 1 for e in expected]
    assert s0.token_map_check(near, expected, side)["pass"] is True
    far = [e + 2 for e in expected]
    assert s0.token_map_check(far, expected, side)["pass"] is False
    assert s0.token_map_check(expected[:-1], expected[:-1], side)["pass"] is False
    some = list(expected)
    for k in range(10):
        some[k] = None
    assert s0.token_map_check(some, expected, side)["pass"] is False


# ---------------------------------------------------------------------------
# Guards (audit test d and the site guard)
# ---------------------------------------------------------------------------


def test_stage0_refuses_any_clip_outside_the_calibration_list():
    ids = s0.calibration_ids()
    assert len(ids) == N and s0.check_clip_ids(list(reversed(ids))) == ids
    with pytest.raises(SystemExit):
        s0.check_clip_ids(ids[:-1] + ["emotions_38/EDITED/Hurt/OF10_cut.mp4"])
    with pytest.raises(SystemExit):
        s0.check_clip_ids(ids[:-1])


def test_stage0_refuses_a_changed_calibration_list(tmp_path):
    path = tmp_path / "clips.csv"
    shutil.copy(s0.CLIPS_CSV, path)
    path.write_text(path.read_text() + "9999_XXX_ANG_XX\n")
    with pytest.raises(SystemExit):
        s0.calibration_ids(path)


@pytest.mark.parametrize("model,site", [("gemma4", "hidden_L1"), ("gemma4", "image_positions_from_L24"),
                                        ("gemma4", "image_positions_window_L6"), ("qwen3vl", "late_cut_image_positions_from_L3"),
                                        ("qwen3vl", "hidden_L1"), ("molmo2", "exit"), ("qwen3vl", "hidden_L0_text")])
def test_sites_past_the_entry_are_refused(model, site):
    with pytest.raises(SiteRefused):
        check_site(model, site)


def test_entry_sites_are_allowed():
    for model, sites in (("gemma4", ("exit", "hidden_L0", "per_layer_all", "tokens")), ("qwen3vl", ("exit", "main", "deepstack", "tokens"))):
        for site in sites:
            assert check_site(model, site) == site


def test_entry_points_refuse_before_the_freeze(monkeypatch):
    from scripts import a7_run

    for module, argv in ((s0, ["a7_stage0", "--model", "gemma4", "--records", "x.json", "--output", "x.json"]),
                         (a7, ["a7_analyse", "--inputs", "x.json", "--output", "x.json"]),
                         (a7_run, ["a7_run", "--model", "gemma4", "--arms", "route", "--stage0", "x.json", "--decision_date", "x"])):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit, match="not frozen"):
            module.main()


def test_placebo_set_is_outside_s_same_size_and_fixed_by_clip():
    from scripts.a7_run import placebo_set

    s = list(range(40))
    q1, q2 = placebo_set(s, 256, "clip"), placebo_set(s, 256, "clip")
    assert q1 == q2 and len(q1) == 40 and not set(q1) & set(s)
    assert placebo_set(s, 256, "other") != q1


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def test_phi_a_plus_phi_b_equals_te_for_every_item():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 5, (4, 1000))
    t = a7.terms(*y)
    assert np.max(np.abs(t["phi_A"] + t["phi_B"] - t["TE"])) < 1e-9
    assert np.max(np.abs(t["I"] - (t["TE"] - t["PE_A"] - t["PE_B"]))) < 1e-9


# ---------------------------------------------------------------------------
# Verdicts through analyse(), on the recovery's synthetic observers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    items = rc.real_items()
    rc._init(items)
    return rc._WORLD


def observer(name, **overrides):
    spec = dict(next(o for o in rc.observers() if o["name"] == name))
    spec.update(overrides)
    return spec


def run(world, spec, effect=2.0, k=0, mutate=None):
    rng = np.random.default_rng(k)
    inputs = {}
    for model in a7.MODELS:
        null = model != "gemma4"
        rows, fills = rc.model_rows(world, rng, effect, spec, null)
        inputs[model] = a7.ModelInputs(rows, rc.stage0_for(spec, null), rc.a2_output_for(spec, null), fills, rc.decisions_for(spec, null))
    if mutate:
        mutate(inputs)
    return a7.analyse(rc.world_items(world), inputs)


def verdict(out, arm="route", model="gemma4"):
    row = out["models"][model]["arms"][arm]
    return row["verdict"], row["reason"]


def test_a_real_pass_is_reachable(world):
    out = run(world, observer("a_only"))
    assert verdict(out) == ("Carried by A", None)
    assert out["models"]["gemma4"]["effect_class"] == "Present"
    assert len(out["family"]["columns"]) == 16


def test_missing_stage0_file_is_not_testable_whatever_else_holds(world):
    def drop(inputs):
        g = inputs["gemma4"]
        inputs["gemma4"] = a7.ModelInputs(g.stage1_rows, None, {a7.ROUTE: {"gates": {}, "margin": None}}, g.a2_rows, None)

    out = run(world, observer("a_only"), mutate=drop)
    assert verdict(out) == ("Not testable", "stage0_not_shown")
    assert verdict(out, "token") == ("Not testable", "stage0_not_shown")
    assert out["rq3"]["gemma4"]["rq3"] == "not testable" and out["rq3"]["gemma4"]["scope"] == "at every level"


def test_precedence_sentinel_failure_beats_a_failed_gate(world):
    out = run(world, observer("a_only", sentinel_fail=True, gate_fail=True, no_decision=True))
    assert verdict(out) == ("Not testable", "sentinel_failed")


def test_precedence_not_run_then_not_tested(world):
    out = run(world, observer("a_only", no_decision=True, gate_fail=True))
    assert verdict(out) == ("Not run", "no_recorded_decision")
    out = run(world, observer("a_only", gate_fail=True))
    assert verdict(out) == ("Not tested", "precondition_failed")


def test_a_missing_gate_has_not_passed(world):
    def drop_gate(inputs):
        g = inputs["gemma4"]
        gates = {k: True for k in ("rq0", "g2_g4", "delivery", "sham")}  # gate_b absent
        inputs["gemma4"] = a7.ModelInputs(g.stage1_rows, g.stage0, {a7.ROUTE: {"gates": gates, "margin": 0.2}}, g.a2_rows, g.decisions)

    assert verdict(run(world, observer("a_only"), mutate=drop_gate)) == ("Not tested", "precondition_failed")


def test_precedence_items_missing_beats_the_effect_class(world):
    out = run(world, observer("a_only", drop_items=30, dead_c5="B"))
    assert verdict(out) == ("Inconclusive", "items_missing")


def test_missing_margin_is_never_absent(world):
    out = run(world, observer("prior_only", no_margin=True))
    assert out["models"]["gemma4"]["effect_class"] == "Unresolved"
    assert verdict(out) == ("Inconclusive", "te_unresolved")
    assert verdict(run(world, observer("prior_only"))) == ("No effect to carry", None)


def test_missing_fill_is_never_present(world):
    out = run(world, observer("a_only", no_fills=True))
    assert out["models"]["gemma4"]["effect_class"] != "Present"
    assert verdict(out) == ("Inconclusive", "te_unresolved")


def test_fill_of_the_other_sign_is_never_present(world):
    def flip(inputs):
        g = inputs["gemma4"]
        rows = copy.deepcopy(g.a2_rows)
        for r in rows:
            if r["condition"] == a7.FILLS[1]:
                base = next(x for x in rows if x["trial_id"] == r["trial_id"] and x["condition"] == a7.ORIGINAL)
                fam = r["label"]
                r["blind"]["family_logprob"][fam] = 2 * base["blind"]["family_logprob"][fam] - r["blind"]["family_logprob"][fam]
        inputs["gemma4"] = a7.ModelInputs(g.stage1_rows, g.stage0, g.a2_output, rows, g.decisions)

    out = run(world, observer("a_only"), mutate=flip)
    assert out["models"]["gemma4"]["fills"][a7.FILLS[1]] < 0 and out["models"]["gemma4"]["effect_class"] != "Present"


def test_block_dead_comes_before_the_carrying_class(world):
    out = run(world, observer("dead_block_effect_via_both"))
    row = out["models"]["gemma4"]["arms"]["route"]
    assert row["carrying_class"] == "Both" and (row["verdict"], row["reason"]) == ("Inconclusive", "block_dead")


def test_block_liveness_does_not_bear_on_a_null_edit(world):
    out = run(world, observer("prior_only", dead_c5="B"))
    assert verdict(out) == ("No effect to carry", None)


def test_zero_or_missing_block_change_on_one_item_is_block_dead(world):
    def zero(inputs):
        g = inputs["gemma4"]
        rows = copy.deepcopy(g.stage1_rows)
        first = next(r for r in rows if r["condition"] == a7.EDIT)
        first["block_change"]["route"]["B"] = 0.0
        inputs["gemma4"] = a7.ModelInputs(rows, g.stage0, g.a2_output, g.a2_rows, g.decisions)

    assert verdict(run(world, observer("a_only"), mutate=zero)) == ("Inconclusive", "block_dead")

    def missing(inputs):
        g = inputs["gemma4"]
        rows = copy.deepcopy(g.stage1_rows)
        next(r for r in rows if r["condition"] == a7.EDIT).pop("block_change")
        inputs["gemma4"] = a7.ModelInputs(rows, g.stage0, g.a2_output, g.a2_rows, g.decisions)

    assert verdict(run(world, observer("a_only"), mutate=missing)) == ("Inconclusive", "block_dead")


def test_one_sentinel_word_over_the_limit_or_a_missing_sentinel_fails_the_arm(world):
    def nudge(inputs):
        g = inputs["gemma4"]
        rows = copy.deepcopy(g.stage1_rows)
        row = next(r for r in rows if r["condition"] == a7.SENTINEL["route"])
        row["blind"]["word_logprob"][50] += TOL * 1.01
        inputs["gemma4"] = a7.ModelInputs(rows, g.stage0, g.a2_output, g.a2_rows, g.decisions)

    out = run(world, observer("a_only"), mutate=nudge)
    assert verdict(out) == ("Not testable", "sentinel_failed")
    assert verdict(out, "token") == ("Carried by A", None)  # the other arm's sentinels are its own

    def drop(inputs):
        g = inputs["gemma4"]
        rows = [r for r in g.stage1_rows if r["condition"] != a7.SENTINEL["route"] or r is None]
        rows = rows[:]
        first = next(i for i, r in enumerate(g.stage1_rows) if r["condition"] == a7.SENTINEL["route"])
        rows = [r for i, r in enumerate(g.stage1_rows) if i != first]
        inputs["gemma4"] = a7.ModelInputs(rows, g.stage0, g.a2_output, g.a2_rows, g.decisions)

    assert verdict(run(world, observer("a_only"), mutate=drop)) == ("Not testable", "sentinel_failed")


def test_every_block_row_is_labelled_decomposition_never_cut(world):
    out = run(world, observer("token_local"))
    for model, body in out["models"].items():
        for arm, row in body["arms"].items():
            assert row["label"] == "decomposition" and "cut" not in row["label"]
            assert row["kind"] == "block"
    for row in out["rq3"].values():
        assert row["rq3"] == "not testable"
        assert row["complete_cuts"] in (["exit"], [])


def test_claim_labels_follow_the_recovery_ruling(world):
    """Ruling 2026-11-08: the interaction rider is descriptive; verdicts, including No effect to
    carry and the carrying verdicts, are confirmatory, with recovery reach beside the null."""
    out = run(world, observer("a_only"))
    assert out["claim_scope"]["confirmatory"][0] == "verdict"
    row = out["models"]["gemma4"]["arms"]["route"]
    assert row["verdict"] == "Carried by A" and row["verdict_confirmatory"] is True
    assert row["interaction_rider_confirmatory"] is False and row["confirmatory"]["verdict"] == "Carried by A"
    assert row["descriptive"]["interaction_rider"] == [r for r in row["riders"] if r in a7.RIDERS]
    assert all(r not in row["confirmatory"].values() for r in a7.RIDERS)
    null = run(world, observer("prior_only"))["models"]["gemma4"]["arms"]["route"]
    assert null["verdict"] == "No effect to carry" and null["verdict_confirmatory"] is True
    assert null["confirmatory"]["recovery_reach_observers"] == ["prior_only", "edit_does_nothing", "edit_null_dead_block"]
    spread = run(world, observer("token_not_local"))["models"]["gemma4"]["arms"]["token"]
    assert spread["confirmatory"]["not_local"] is True and "not_local" not in spread["descriptive"]["interaction_rider"]
    for model, body in out["models"].items():
        for arm, r in body["arms"].items():
            assert r["verdict_confirmatory"] is True and r["interaction_rider_confirmatory"] is False, (model, arm)


def test_unlabelled_row_stops_the_analysis():
    row = {"verdict": "Carried by A", "reason": None, "kind": "block", "label": "decomposition", "riders": []}
    with pytest.raises(SystemExit):
        a7.check_rows({"models": {"gemma4": {"arms": {"route": dict(row)}}}})
    a7.check_rows({"models": {"gemma4": {"arms": {"route": a7.label_claims(dict(row))}}}})


def test_items_missing_lists_failures_by_label_and_reason(world):
    out = run(world, observer("a_only", drop_items=30))
    row = out["models"]["gemma4"]["arms"]["route"]
    assert (row["verdict"], row["reason"]) == ("Inconclusive", "items_missing")
    assert sum(row["failures_by_label_reason"].values()) == 30
    assert all(k.endswith("/cells_missing") for k in row["failures_by_label_reason"])


def test_recovery_reach_reads_the_recorded_rates():
    reach = a7.recovery_reach()
    assert reach is not None
    assert reach["observers"]["edit_does_nothing"]["expected"][0] == "No effect to carry"
    assert a7.recovery_reach(ROOT / "absent.json") is None


def test_a2_crosscheck_is_descriptive_and_pins_refuse_other_designs(world):
    rng = np.random.default_rng(0)
    rows, _fills = rc.model_rows(world, rng, 2.0, observer("a_only"), False)
    words = list(np.round(rng.normal(-5, 1, W), 6))
    for r in rows:
        r["blind"]["word_logprob"] = words
        r["a7_design_sha256"] = a7_design.DESIGN_SHA256
    a2 = [{**r, "blind": {**r["blind"], "word_logprob": [round(w, 4) for w in words]}, "design_sha256": D["a2_design_sha256"]}
          for r in rows if r["condition"] in (a7.ORIGINAL, a7.EDIT)]
    check = a7.a2_crosscheck(rows, a2)
    assert check["descriptive"] is True and check[a7.ORIGINAL]["n_items"] == len(world.items)
    assert check[a7.ORIGINAL]["max_abs_word_logprob"] <= 5e-5
    assert a7.a2_crosscheck(rows, None)["available"] is False
    a7.check_a2_pins({"design_sha256": D["a2_design_sha256"]}, a2, rows)
    with pytest.raises(SystemExit):
        a7.check_a2_pins({"design_sha256": "0" * 64}, a2, rows)
    with pytest.raises(SystemExit):
        a7.check_a2_pins(None, [{**a2[0], "design_sha256": "0" * 64}], rows)
    with pytest.raises(SystemExit):
        a7.check_a2_pins(None, a2, [{**rows[0], "a7_design_sha256": "0" * 64}])


def test_token_arm_reading_and_not_local(world):
    local = run(world, observer("token_local"))["models"]["gemma4"]["arms"]["token"]
    assert local["verdict"] == "Carried by A" and local["reading"] == "carried by cue tokens" and "not_local" not in local["riders"]
    spread = run(world, observer("token_not_local"))["models"]["gemma4"]["arms"]["token"]
    assert spread["verdict"] == "Carried by A" and "not_local" in spread["riders"]


def test_a_row_with_no_verdict_stops_the_analysis():
    with pytest.raises(SystemExit):
        a7.check_rows({"models": {"gemma4": {"arms": {"route": {"verdict": None, "reason": None, "kind": "block", "label": "decomposition"}}}}})
    with pytest.raises(SystemExit):
        a7.check_rows({"models": {"gemma4": {"arms": {"route": {"verdict": "Inconclusive", "reason": "other", "kind": "block", "label": "decomposition"}}}}})
    with pytest.raises(SystemExit):
        a7.check_rows({"models": {"gemma4": {"arms": {"route": {"verdict": "Carried by A", "reason": None, "kind": "block", "label": "cut"}}}}})


def test_classes_on_columns():
    rng = np.random.default_rng(0)
    big = a7.Column(rng.normal(2, 0.3, 17), 0.0001)
    zero = a7.Column(rng.normal(0, 0.05, 17), 0.9)
    neg = a7.Column(rng.normal(-1, 0.3, 17), 0.0001)
    assert a7.carrying_class(big, zero, 2.0, 0.4) == "A only"
    assert a7.carrying_class(zero, big, 2.0, 0.4) == "B only"
    assert a7.carrying_class(big, big, 2.0, 0.4) == "Both"
    assert a7.carrying_class(big, neg, 2.0, 0.4) == "Opposed"
    assert a7.carrying_class(zero, zero, 2.0, 0.4) == "Unresolved"
    assert a7.interaction_rider(zero, 2.0, 0.4) == "Additive"
    assert a7.interaction_rider(neg, 2.0, 0.4) == "Redundant"
    assert a7.interaction_rider(big, 2.0, 0.4) == "Synergic"
    assert a7.effect_class(big, 0.2, {f: 1.0 for f in a7.FILLS}) == "Present"
    assert a7.effect_class(big, 0.2, {a7.FILLS[0]: 1.0, a7.FILLS[1]: None}) == "Unresolved"
    assert a7.effect_class(zero, 0.2, {f: 1.0 for f in a7.FILLS}) == "Absent"
    assert a7.effect_class(zero, None, {f: 1.0 for f in a7.FILLS}) == "Unresolved"
    assert not zero.inside(0.0) and not zero.inside(float("nan"))


def test_a2_margin_reader_fails_safe():
    assert a7.a2_margin(None) is None
    assert a7.a2_margin({"single_face": {"margin": float("nan")}}) is None
    assert a7.a2_margin({"single_face": {"margin": 0.0}}) is None
    assert a7.a2_margin({"single_face": {"margin": "x"}}) is None
    assert a7.a2_margin({"single_face": {"margin": 0.3}}) == 0.3


# ---------------------------------------------------------------------------
# Real pinned gate paths (End to end) and Gate B4 against the block terms
# ---------------------------------------------------------------------------


def test_preconditions_from_the_real_pinned_gate_paths_return_true_for_a_real_pass(tmp_path, world):
    repro = ROOT / "research" / "reproduction"
    for model in a7.MODELS:
        for name in (f"vision_gates_{model}.json", f"input_delivery_{model}.json"):
            shutil.copy(repro / name, tmp_path / name)
    shutil.copy(repro / an.A3_SUMMARY_NAME, tmp_path / an.A3_SUMMARY_NAME)
    (tmp_path / "rotation_summary.json").write_text(json.dumps({m: {"vision_dependent": True} for m in a7.MODELS}))
    gates = {}
    for model in a7.MODELS:
        outside = an.outside_gates(tmp_path, model, a7.ROUTE, a3_output_dir=tmp_path / "absent")
        gates[model] = {"rq0": outside["rq0"], "g2_g4": outside["g2_g4"], "delivery": outside["delivery_file"],
                        "gate_b": True, "sham": True}
        assert a7.preconditions_met(gates[model]) is True, (model, outside)

    def real_gates(inputs):
        for model in a7.MODELS:
            g = inputs[model]
            inputs[model] = a7.ModelInputs(g.stage1_rows, g.stage0, {a7.ROUTE: {"gates": gates[model], "margin": 0.2}}, g.a2_rows, g.decisions)

    assert verdict(run(world, observer("a_only"), mutate=real_gates)) == ("Carried by A", None)
    # A1 failed and A3 passed: single_face stays admitted by the frozen table. A1 missing: not passed.
    (tmp_path / "rotation_summary.json").write_text(json.dumps({m: {"vision_dependent": False} for m in a7.MODELS}))
    assert an.outside_gates(tmp_path, "gemma4", a7.ROUTE, a3_output_dir=tmp_path / "absent")["rq0"] is True
    (tmp_path / "rotation_summary.json").unlink()
    outside = an.outside_gates(tmp_path, "gemma4", a7.ROUTE, a3_output_dir=tmp_path / "absent")
    assert outside["rq0"] is False
    assert a7.preconditions_met({**gates["gemma4"], "rq0": outside["rq0"]}) is False


def _b4_cells(world, face_signal, block_terms):
    from paper_v2.option_score import vision_mismatch_partners

    rng = np.random.default_rng(11)
    rows = []
    fams = rc.FAMILIES
    for k, item in enumerate(world.items):
        others = rng.normal(0, 0.3, len(fams))
        acted = world.acted[k]
        y0 = face_signal + rng.normal(0, 0.3)
        meta = {"trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"],
                "human_modal_label": item["human_modal_label"], "weak_label": item["weak_label"],
                "input_name": a7.ROUTE, "status": "ok"}
        values = {a7.ORIGINAL: y0, a7.EDIT: y0 + block_terms, "hold_route_A_edit": y0 + block_terms, "hold_route_B_edit": y0}
        for condition, y in values.items():
            blind = {"family_logprob": rc.scores(others, acted, y), "lexicon_mass": 0.5}
            rows.append({**meta, "condition": condition, "blind": blind})
    cells = an.Cells(rows)
    ids = sorted(cells.items)
    partners = vision_mismatch_partners(ids, [cells.items[i]["label"] for i in ids], [cells.items[i]["actor"] for i in ids], seed=42)
    b = an.gate_b(cells, a7.ROUTE, partners)
    trials = [i["trial_id"] for i in world.items]
    _kept, values = a7.item_table(cells, trials, "route")
    return b, values


def test_gate_b4_and_the_block_terms_are_different_quantities(world):
    b, values = _b4_cells(world, face_signal=2.0, block_terms=0.0)
    assert b["B4"] is True
    assert all(np.allclose(values[k], 0.0) for k in ("phi_A", "phi_B", "I", "TE"))
    b, values = _b4_cells(world, face_signal=0.0, block_terms=1.0)
    assert b["B4"] is False
    assert np.allclose(values["TE"], 1.0) and np.allclose(values["phi_A"], 1.0) and np.allclose(values["phi_B"], 0.0)


# ---------------------------------------------------------------------------
# The hooks on toys (Level 1 in miniature) and on the tiny real Gemma 4
# ---------------------------------------------------------------------------


def _toy_table(n_clips=6, region=False, **kwargs):
    from scripts import a7_toy as toy

    model = toy.ToyVLM(kwargs.pop("style", "gemma4"), **kwargs).eval()
    images = toy.toy_images(n_clips)
    return s0.run_stage0(rc._toy_session(model, model.style), images, expected_tokens=toy.SIDE * toy.SIDE,
                         region=rc._toy_region(images) if region else None)


def test_toy_without_a_third_route_is_complete_and_both_blocks_live():
    table = _toy_table()
    assert table["complete_cuts"] == ["exit"] and table["blocks_exhaust_exit"] and s0.dead_blocks(table) == []
    assert all(table["checks"][k]["worst_abs_nats"] == 0.0 for k in ("C1", "C2", "C3", "C4"))


def test_toy_third_route_over_the_limit_is_not_testable():
    table = _toy_table(leak_nats=0.002, style="qwen3vl")
    assert not any(table["checks"][k]["pass"] for k in ("C2", "C3", "C4"))
    assert s0.arms(table) == {"route": False, "token": False}


def test_toy_non_deterministic_forward_stops_at_c1():
    table = _toy_table(noise=0.01)
    assert table["status"] == "stopped_c1" and "C2" not in table["checks"]


def test_toy_layout_fault_stops_stage0():
    from scripts import a7_toy as toy

    model = toy.ToyVLM("gemma4").eval()
    with pytest.raises(s0.Fault):
        s0.run_stage0(rc._toy_session(model, "gemma4"), toy.toy_images(4), expected_tokens=576)


def test_toy_dry_run_writes_a_table_in_the_final_format(tmp_path):
    table = _toy_table(region=True, n_clips=N)
    table.update({"model": "toy", "clip_ids": [f"toy_{i}" for i in range(N)]})
    path = tmp_path / "a7_stage0_toy.json"
    path.write_text(json.dumps(table))
    loaded = s0.load_table(path)
    assert loaded is not None and s0.arms(loaded) == {"route": True, "token": True}
    assert loaded["region_test"]["b"] == 0


def _has_gemma4() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("transformers.models.gemma4") is not None
    except (ImportError, ValueError):
        return False


@pytest.mark.skipif(not _has_gemma4(), reason="needs transformers with gemma4 (the project env has 5.9.0)")
def test_gemma_l0_and_per_layer_hold_on_the_tiny_real_gemma4():
    from scripts import a7_toy as toy
    from scripts.a7_cuts import Session
    from scripts.vision_gates import adapter_for

    model = toy.tiny_gemma4()
    adapter = adapter_for("gemma4", model)
    session = Session(adapter, "gemma4", lambda clip, _c: toy.tiny_gemma4_inputs(clip),
                      toy.readout_for(model, toy.TINY_WORD_IDS), D["route_blocks"]["gemma4"])
    xi, xj = toy.tiny_gemma4_inputs(1), toy.tiny_gemma4_inputs(2)
    native, state = session.native(xi)
    assert sorted(state.per_layer) == list(range(len(adapter.lm_layers)))
    both = session.hold(xj, ("A", "B"), state)
    only_a = session.hold(xj, ("A",), state)
    only_b = session.hold(xj, ("B",), state)
    assert np.array_equal(both, native)
    assert np.abs(only_a - native).max() > TOL and np.abs(only_b - native).max() > TOL
    free, placeholder = session.image_free(xi)
    assert np.array_equal(session.exit_cut(xi, placeholder), free)
    assert np.array_equal(session.exit_cut(xj, placeholder), free)


def test_recovery_judge_applies_the_draft_rule():
    spec = observer("a_only")
    good = {"expected_rate": 0.9, "other_named_rate": 0.05, "inconclusive_rate": 0.05}
    assert rc.judge(spec, good)["pass"] is True
    assert rc.judge(spec, {**good, "expected_rate": 0.79})["pass"] is False
    assert rc.judge(spec, {**good, "other_named_rate": 0.11})["pass"] is False
    assert rc.judge(spec, {**good, "inconclusive_rate": 0.21})["pass"] is False
    assert rc.judge(observer("false_carry_0.25"), {"carried_by_a_rate": 0.071})["pass"] is False
    assert rc.judge(observer("all_null_family"), {"any_column_significant_rate": 0.07})["pass"] is True


def test_recovery_fast_signflip_equals_a2s_signflip():
    from scripts import stimulus_interventions as si

    original = rc._ORIGINAL_SIGNFLIP
    rng = np.random.default_rng(5)
    for k in range(6):
        m = rng.normal(0, 1, (17, 16)) + rng.normal(0, 1, 16) * (k % 3)
        a, b = original(m, alternative=[0] * 16), rc.fast_signflip_maxt(m, alternative=[0] * 16)
        assert np.allclose(a["p_maxt"], b["p_maxt"], atol=1e-9) and np.allclose(a["p_raw"], b["p_raw"], atol=1e-9)
        assert np.allclose(a["t"], b["t"])
    zero = np.zeros((17, 3))
    assert rc.fast_signflip_maxt(zero)["p_maxt"] == original(zero)["p_maxt"]
    assert si.signflip_maxt is original or si.signflip_maxt is rc.fast_signflip_maxt
