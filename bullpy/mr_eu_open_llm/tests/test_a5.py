import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from analyse_rotation import actor_bootstrap_ci, actor_sign_flip_p  # noqa: E402
from scripts import a5_analyse as a5  # noqa: E402
from scripts import a5_design as design  # noqa: E402
from scripts import a5_recovery as rec  # noqa: E402
from scripts import analyse_stimulus_interventions as an  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_design as sd  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402

D = a5.D
REPRO = ROOT / "research" / "reproduction"
PASSED = {"rq0": True, "g2_g4": True, "delivery": True}
FULL_GATE = {"B1": True, "B2": True, "B3": True, "B4": True, "B5": True, "pass": True}
SHUFFLES = 60


@pytest.fixture(scope="module")
def items():
    return rec.real_items(ROOT / "data/eu_emotions_full_manifest.json", ROOT / "data/eu_emotions_human_entropy.json")


@pytest.fixture(scope="module")
def world(items):
    return rec.World(items)


@pytest.fixture(scope="module")
def human():
    data = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    return {k: v["human_response_distribution"] for k, v in data.items()}


def judged(world, spec, seed=0):
    rng = np.random.default_rng(seed)
    s, w, g = world.scores(spec, rng)
    return world.judgement(s), a5.softmax_of_log(w), a5.softmax_of_log(g), s


def built(world, spec, seed=0, gates=None):
    rng = np.random.default_rng(seed)
    s, w, g = world.scores(spec, rng)
    return a5.build_cell("synthetic", rec.ROUTE, an.Cells(world.rows(s)), world.actors, world.partners,
                         a5.baselines([world.report(w, g)], rec.ROUTE), dict(gates or PASSED), 0.0, SHUFFLES)


# ---------------------------------------------------------------------------
# Design block
# ---------------------------------------------------------------------------


def test_block_parses_and_hash_is_pinned_and_stated():
    assert design.block_hash() == design.DESIGN_SHA256 == design.stated_hash()
    assert design.load_design()["amendment"] == "A5"
    raw = design.block_bytes()
    assert raw.endswith(b"}\n") and not raw.endswith(b"\n\n")


def test_hash_mismatch_is_refused(tmp_path):
    text = design.DRAFT.read_text(encoding="utf-8")
    edited = tmp_path / "draft.md"
    edited.write_text(text.replace('"seed": 42', '"seed": 43'), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the pinned"):
        design.load_design(edited)
    # A block that matches the pin but differs from what the draft states is refused too.
    moved = tmp_path / "moved.md"
    moved.write_text(text.replace(design.DESIGN_SHA256, "0" * 64, 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="draft states"):
        design.load_design(moved)
    bare = tmp_path / "bare.md"
    bare.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        design.load_design(bare)


def test_a2_code_is_the_frozen_code_and_a_change_is_refused(tmp_path):
    design.check_a2_code(D)
    name = "scripts/stimulus_design.py"
    (tmp_path / "scripts").mkdir()
    for key in D["a2_code_sha256"]:
        (tmp_path / key).write_bytes((ROOT / key).read_bytes())
    (tmp_path / name).write_bytes((ROOT / name).read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="not the frozen"):
        design.check_a2_code(D, tmp_path)


def test_block_agrees_with_a2():
    assert run.design_hash() == D["a2_design_sha256"]
    assert D["a3_summary_sha256"] == sd.A3_SUMMARY_SHA256
    assert D["n_items"] == sd.N_ITEMS and D["seed"] == si.SEED and D["alpha"] == sd.ALPHA
    assert D["bootstrap_draws"] == sd.BOOTSTRAP_DRAWS and D["token_floor"] == sd.TOKEN_FLOOR
    assert D["lexicon_words"] == len(run.lexicon_words()) and D["n_families"] == len(run.LEXICON)
    assert tuple(D["weak_labels"]) == run.WEAK_LABELS


def test_real_design_counts(items, world):
    assert len(items) == D["n_items"] and len(world.actors) == D["n_actors"]
    assert sum(i["weak_label"] for i in items) == 26
    ids = {i["trial_id"] for i in items}
    assert set(sd.SMOKE_ITEMS) <= ids


def test_committed_a3_file_is_the_pinned_one_by_hash_only():
    digest = hashlib.sha256((REPRO / an.A3_SUMMARY_NAME).read_bytes()).hexdigest()
    assert digest == D["a3_summary_sha256"]


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["acted", "wrong", "spread"])
def test_identity_holds_for_every_clip_on_real_labels(world, mode):
    j, w, g, _ = judged(world, rec.base_spec(prior=1.0, signal=1.5, mode=mode, identity=0.5))
    p = a5.decompose(j, w, g)
    assert np.abs(p["T"] - (p["W"] + p["G"] + p["F"] + p["E"] + p["R"])).max() < 1e-9
    assert np.abs(p["P"] - (p["W"] + p["G"] + p["F"])).max() < 1e-9
    assert np.abs(p["D"] - (p["E"] + p["R"])).max() < 1e-9
    assert np.abs(p["T"] - (p["P"] + p["D"])).max() < 1e-9
    assert p["D"].min() >= -1e-12  # a KL divergence


def test_identity_holds_under_the_size_reference(world):
    j, w, g, _ = judged(world, rec.base_spec(prior=1.0, signal=1.0))
    u = a5.size_reference()
    p = a5.decompose(j, w, g, u)
    assert abs(u.sum() - 1) < 1e-12
    assert np.abs(p["T"] - (p["W"] + p["G"] + p["F"] + p["E"] + p["R"])).max() < 1e-9


def test_default_and_template_never_touch_the_clips_own_actor(world):
    j, *_ = judged(world, rec.base_spec(prior=1.0, signal=1.0))
    n = len(j.actors)
    default = a5.default_for_clips(j.q, j.actor, n)
    tpl = a5.templates(j.q, j.actor, j.label, n, len(j.labels))
    changed = j.q.copy()
    mine = j.actor == 3
    changed[mine] = np.roll(changed[mine], 5, axis=1)
    assert np.abs(a5.default_for_clips(changed, j.actor, n)[mine] - default[mine]).max() < 1e-12
    held = a5.templates(changed, j.actor, j.label, n, len(j.labels))[:, 3]
    assert np.nanmax(np.abs(held - tpl[:, 3])) < 1e-12


def test_default_is_a_mean_of_actor_means_not_of_clips():
    q = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    actor = np.array([0, 0, 0, 1, 2])
    assert a5.default_for_clips(q, actor, 3)[4] == pytest.approx([0.7, 0.3])
    assert a5.default_for_clips(q, actor, 3, leave_out=False)[0] == pytest.approx([0.5, 0.5])


def test_no_template_is_nan_and_clip_stays_in_t_and_d():
    q = np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]])
    j = a5.Judgement(tuple("abcd"), q, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 2]), np.array([0, 1, 0, 1]),
                     ("x", "y"), ("p", "q", "r"), np.ones(4))
    p = a5.decompose(j)
    assert np.isnan(p["E"][1]) and np.isfinite(p["E"][0]) and np.isfinite(p["D"]).all()


def test_shuffles_stay_within_actor_and_keep_each_label_set(world):
    j, *_ = judged(world, rec.base_spec(signal=1.0))
    index = a5.shuffle_index(j.actor, 25, D["seed"])
    for row in index:
        assert np.array_equal(j.actor[row], j.actor)
        for a in range(len(j.actors)):
            assert sorted(j.label[row][j.actor == a]) == sorted(j.label[j.actor == a])
    assert (index != np.arange(j.actor.size)).any(1).all()


def test_fast_shuffled_e_equals_a_direct_recomputation(world):
    j, *_ = judged(world, rec.base_spec(prior=0.5, signal=1.0))
    fast = a5.shuffled_e_rows(j, 12, 7)
    index = a5.shuffle_index(j.actor, 12, 7)
    n = len(j.actors)
    direct = np.mean([a5.actor_means(a5.decompose(a5.Judgement(j.trials, j.q, j.actor, j.label[row], j.target, j.actors, j.labels, j.mass))["E"],
                                     j.actor, n) for row in index], 0)
    assert np.abs(fast - direct).max() < 1e-9


def test_e_star_is_centred_without_label_signal(world):
    values = [a5.estimate(judged(world, rec.base_spec(prior=0.3, identity=1.0), seed)[0], shuffles=40).E_star_row.mean() for seed in range(10)]
    T = a5.estimate(judged(world, rec.base_spec(prior=0.3, identity=1.0))[0], shuffles=5).T
    assert abs(np.mean(values)) < 0.02 * T


def test_even_is_the_direct_test_that_neither_part_is_twice_the_other():
    rng = np.random.default_rng(0)

    def est(ratio):  # D / P per actor near ratio, with small actor noise
        P = 1.0 + 0.05 * rng.normal(size=17)
        Dv = ratio * P + 0.05 * rng.normal(size=17)
        rows = {"P": P, "D": Dv, "T": P + Dv}
        return a5.Estimates({}, rows, Dv - P, np.zeros(17), float((P + Dv).mean()), 0.1)

    assert min(a5.balance_stats(est(1.5))["neither_dominant_lower_bounds"]) > 0
    assert min(a5.balance_stats(est(0.7))["neither_dominant_lower_bounds"]) > 0
    assert min(a5.balance_stats(est(2.3))["neither_dominant_lower_bounds"]) < 0
    assert min(a5.balance_stats(est(0.4))["neither_dominant_lower_bounds"]) < 0
    assert D["dominance_ratio"] == 2.0


def test_even_needs_b_not_told_from_zero_by_its_interval():
    excluded = {**ZERO, "ci95": [0.01, 0.05]}
    assert decide(ZERO, excluded, failing("B4"), p_b=0.2)["reasons"] == ["balance_unresolved"]


def test_e_star_margin_is_not_widened_by_the_balance_rule():
    mid = {"mean": 0.0, "ci95": [-0.5, 0.5], "ci90": [-0.4, 0.4], "neither_dominant_lower_bounds": [0.1, 0.1]}
    assert decide(ZERO, mid, failing("B4"))["balance"] == "Even"
    row = decide(mid, ZERO, failing("B4"))
    assert row["evidence"] == "Unresolved" and row["reasons"] == ["evidence_unresolved"]


def test_bootstrap_t_detects_a_shift_and_step_down_is_not_below_raw():
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.normal(2.0, 1, 17), rng.normal(0, 1, 17), rng.normal(0, 1, 17)])
    out = a5.bootstrap_t_maxt(x, [0, 0, 1], draws=4000)
    assert out["p_maxt"][0] < 0.01
    assert all(m >= r - 1e-12 for m, r in zip(out["p_maxt"], out["p_raw"]))


def test_bootstrap_t_is_close_to_nominal_on_symmetric_null_rows():
    rng = np.random.default_rng(4)
    hits = [a5.bootstrap_t_maxt(rng.normal(0, 1, (17, 1)), [0], draws=999, seed=k)["p_raw"][0] < 0.05 for k in range(300)]
    assert 0.01 < np.mean(hits) < 0.10


def test_bootstrap_t_interval_brackets_the_mean():
    x = np.random.default_rng(5).lognormal(0, 1, 17)
    bounds = a5.bootstrap_t_bounds(x, [0.95, 0.90], draws=2000)
    assert bounds[0.95][0] < bounds[0.90][0] < x.mean() < bounds[0.90][1] < bounds[0.95][1]


def test_phi_interval_brackets_phi(world):
    est = a5.estimate(judged(world, rec.base_spec(prior=0.7, signal=1.0))[0], shuffles=5)
    low, high = a5.phi_interval(est, draws=2000)
    assert low < np.nanmean(est.rows["D"]) / est.T < high


def test_phi_above_one_comes_from_leaving_the_actor_out(world):
    """Stated in the draft: with no prior, P is below 0 only because the actor is left out."""
    j, *_ = judged(world, rec.base_spec(signal=2.0, w="flat"))
    n = len(j.actors)

    def share(leave_out):
        parts = a5.decompose(j, leave_out=leave_out)
        return np.nanmean(a5.actor_means(parts["P"], j.actor, n)), np.nanmean(a5.actor_means(parts["D"], j.actor, n))

    assert share(True)[0] < 0 < share(False)[0]


def test_temperature_keeps_a_distribution(world):
    j, *_ = judged(world, rec.base_spec(signal=1.0))
    assert np.abs(j.tempered(2.0).q.sum(1) - 1).max() < 1e-12
    assert j.tempered(0.5).q.max() < j.q.max() < j.tempered(2.0).q.max()


# ---------------------------------------------------------------------------
# Riders
# ---------------------------------------------------------------------------


def test_word_share_reads_each_prior_correctly(world):
    word = a5.estimate(*judged(world, rec.base_spec(prior=1.5, actor_sd=0.1, item_sd=0.1))[:3], shuffles=5)
    face = a5.estimate(*judged(world, rec.base_spec(prior=1.5, w="flat", actor_sd=0.1, item_sd=0.1))[:3], shuffles=5)
    assert a5.word_share(word)["reading"] == "word_prior"
    assert a5.word_share(face)["reading"] == "image_or_face"


def test_word_share_is_mixed_when_p_is_not_above_zero():
    rows = {"W": np.array([0.1, 0.2, 0.1]), "P": np.array([-0.1, -0.2, -0.1])}
    est = a5.Estimates({}, rows, np.zeros(3), np.zeros(3), 1.0, 0.1)
    out = a5.word_share(est)
    assert out["reading"] == "mixed" and out["omega"] is None


def test_survival_by_hand():
    q = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1], [0.2, 0.2, 0.6]])
    q = np.vstack([q] * 5)
    actor = np.repeat(np.arange(5), 4)
    target = np.tile([0, 1, 0, 2], 5)
    trials = tuple(f"t{i}" for i in range(20))
    j = a5.Judgement(trials, q, actor, target, target, tuple("abcde"), tuple("xyz"), np.ones(20))
    partners = {t: trials[(i + 1) % 20] for i, t in enumerate(trials)}
    out = a5.survival(j, partners, draws=200)
    # Clips 0, 1 and 3 of each actor are right. Own: .8, .8, .6. On the next clip: .1, .5, .1.
    assert out["n_correct_items"] == 15 and out["n_correct_actors"] == 5
    assert out["S"] == pytest.approx(0.7 / 2.2, abs=1e-9)


def test_survival_not_estimated_below_its_floors():
    q = np.tile([0.8, 0.2], (6, 1))
    j = a5.Judgement(tuple("abcdef"), q, np.array([0, 0, 0, 1, 1, 1]), np.zeros(6, int), np.zeros(6, int), ("x", "y"), ("l",), np.ones(6))
    partners = dict(zip("abcdef", "bcdefa"))
    assert a5.survival(j, partners)["S"] is None


def test_chain_merges_g_and_f_at_low_grey_mass(world):
    cell = built(world, rec.base_spec(prior=0.5, signal=1.0))
    assert "G" in a5.chain(cell.est, cell.j, cell.base)
    low = {**cell.base, "mass_grey": D["grey_mass_floor"] / 2}
    out = a5.chain(cell.est, cell.j, low)
    assert "G" not in out and "G_plus_F" in out and out["G_F_merged"] is True


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def report(free, grey=None, route="single_face"):
    rows = [{"input_name": route, "condition": "image_free", "status": "ok", "family_logprob": free, "lexicon_mass": 0.5},
            {"input_name": route, "condition": "grey", "status": "ok", "family_logprob": grey or free, "lexicon_mass": 0.1}]
    return {"baselines": rows}


def test_baselines_agree_across_shards_within_tolerance():
    free = {f: -1.0 for f in a5.FAMILIES}
    moved = {**free, a5.FAMILIES[0]: -1.0 + 0.5 * D["baseline_agreement_nats"]}
    assert a5.baselines([report(free), report(moved)], "single_face")["ok"] is True


def test_disagreeing_or_missing_baselines_fail_the_check():
    free = {f: -1.0 for f in a5.FAMILIES}
    moved = {**free, a5.FAMILIES[0]: -1.0 + 2 * D["baseline_agreement_nats"]}
    assert a5.baselines([report(free), report(moved)], "single_face")["ok"] is False
    assert a5.baselines([report(free), {"baselines": None}], "single_face")["ok"] is False
    assert a5.baselines([report(free)], "video_face")["ok"] is False
    failed = {"baselines": [{"input_name": "single_face", "condition": "image_free", "status": "failed"}]}
    assert a5.baselines([failed], "single_face")["ok"] is False


def test_b_and_e_star_need_no_baseline(world):
    j, w, g, _ = judged(world, rec.base_spec(prior=0.5, signal=1.0))
    with_base, without = a5.estimate(j, w, g, shuffles=10), a5.estimate(j, shuffles=10)
    assert np.array_equal(with_base.B_row, without.B_row)
    assert np.array_equal(with_base.E_star_row, without.E_star_row)
    assert without.parts["W"] is None


# ---------------------------------------------------------------------------
# Classes and verdicts
# ---------------------------------------------------------------------------

# neither_dominant_lower_bounds: the two one-sided bounds for 2P - D and 2D - P. Used for B only.
HIGH = {"mean": 1.0, "ci95": [0.5, 1.5], "ci90": [0.6, 1.4], "neither_dominant_lower_bounds": [-1.0, 0.5]}
LOW = {"mean": -1.0, "ci95": [-1.5, -0.5], "ci90": [-1.4, -0.6], "neither_dominant_lower_bounds": [0.5, -1.0]}
ZERO = {"mean": 0.0, "ci95": [-0.05, 0.05], "ci90": [-0.04, 0.04], "neither_dominant_lower_bounds": [0.3, 0.3]}
WIDE = {"mean": 0.0, "ci95": [-1.0, 1.0], "ci90": [-0.9, 0.9], "neither_dominant_lower_bounds": [-0.2, -0.2]}
T, MARGIN = 2.0, 0.2


def decide(e, b, gate=FULL_GATE, gates=PASSED, p_e=None, p_b=None, base=True, t=T, cells=None):
    p_e = (0.001 if e is HIGH else 0.5) if p_e is None else p_e
    p_b = (0.001 if b in (HIGH, LOW) else 0.5) if p_b is None else p_b
    cells = D["min_balance_cells"] if cells is None else cells
    return a5.decide(gates, gate, t, MARGIN, b, e, p_b, p_e, base, family_cells=cells)


def failing(*names):
    gate = {**FULL_GATE, "pass": False}
    gate.update({n: False for n in names})
    return gate


NAMED_ROWS = [
    (HIGH, HIGH, FULL_GATE, "Evidence-led"),
    (HIGH, LOW, FULL_GATE, "Default-led with evidence"),
    (HIGH, ZERO, FULL_GATE, "Evidence, neither part dominant"),
    (HIGH, HIGH, failing("B4"), "Face-led, unmapped"),
    (HIGH, LOW, failing("B5"), "Default-led, unmapped"),
    (HIGH, ZERO, failing("B4", "B5"), "Unmapped, neither part dominant"),
    (ZERO, LOW, failing("B4"), "No face evidence"),
    (ZERO, ZERO, failing("B4"), "No face evidence, neither part dominant"),
    (ZERO, HIGH, failing("B4"), "Face-dependent, no expression"),
]


@pytest.mark.parametrize("e,b,gate,expected", NAMED_ROWS)
def test_decision_table(e, b, gate, expected):
    assert decide(e, b, gate)["verdict"] == expected


@pytest.mark.parametrize("cells", [1, 2, 3, 4, 5])
def test_every_named_row_obeys_the_family_size_rule(cells):
    assert len({row[3] for row in NAMED_ROWS}) == 9
    for e, b, gate, expected in NAMED_ROWS:
        row = decide(e, b, gate, cells=cells)
        if cells >= D["min_balance_cells"]:
            assert row["verdict"] == expected and row["balance_withheld"] is None
        else:
            assert row["verdict"] == a5.INCONCLUSIVE and "balance_small_family" in row["reasons"]
            assert row["balance"] == "Unresolved" and row["balance_withheld"] in ("Face", "Default", "Even")


def test_decide_requires_family_cells():
    with pytest.raises(TypeError):
        a5.decide(PASSED, FULL_GATE, T, MARGIN, HIGH, HIGH, 0.001, 0.001, True)


def test_failed_gate_b_cannot_give_a_shown_verdict():
    for name in ("B3", "B4", "B5"):
        for balance in (HIGH, LOW, ZERO):
            assert decide(HIGH, balance, failing(name))["verdict"] not in rec.EVIDENCE_VERDICTS


def test_b3_alone_failing_is_not_inconclusive_when_e_star_passes():
    """The gap the audit closed: B4 and B5 pass, B3 does not, and no row covered it."""
    row = decide(HIGH, HIGH, failing("B3"))
    assert row["verdict"] == "Face-led, unmapped" and "b3_failed" in row["flags"]


def test_b4_passing_with_e_star_null_is_evidence_unresolved():
    row = decide(ZERO, HIGH)
    assert row["verdict"] == a5.INCONCLUSIVE and row["reasons"] == ["evidence_unresolved"]


def test_wide_intervals_give_their_reason_codes():
    assert decide(WIDE, HIGH, failing("B4"))["reasons"] == ["evidence_unresolved"]
    assert decide(HIGH, WIDE)["reasons"] == ["balance_unresolved"]
    assert decide(WIDE, WIDE, failing("B4"))["reasons"] == ["evidence_unresolved", "balance_unresolved"]


def test_significant_with_an_interval_across_zero_is_unresolved():
    leaning = {"mean": 0.5, "ci95": [-0.1, 1.1], "ci90": [0.0, 1.0], "neither_dominant_lower_bounds": [-0.1, 0.2]}
    assert decide(HIGH, leaning, p_b=0.01)["reasons"] == ["balance_unresolved"]
    assert decide(leaning, HIGH, failing("B4"), p_e=0.01)["reasons"] == ["evidence_unresolved"]


def test_structure_floor():
    row = decide(HIGH, HIGH, t=D["min_total_nats"] / 2)
    assert row["verdict"] == a5.INCONCLUSIVE and row["reasons"] == ["no_structure"]


@pytest.mark.parametrize("gone", ["rq0", "g2_g4", "delivery"])
def test_missing_outside_gate_is_not_tested_not_inconclusive(gone):
    gates = {k: v for k, v in PASSED.items() if k != gone}
    assert decide(HIGH, HIGH, gates=gates)["verdict"] == a5.NOT_TESTED
    assert decide(HIGH, HIGH, gates={**PASSED, gone: False})["verdict"] == a5.NOT_TESTED


@pytest.mark.parametrize("name", ["B1", "B2"])
def test_b1_or_b2_is_not_tested(name):
    assert decide(HIGH, HIGH, failing(name))["verdict"] == a5.NOT_TESTED


def test_no_baseline_keeps_the_verdict_and_flags_it():
    row = decide(HIGH, HIGH, base=False)
    assert row["verdict"] == "Evidence-led" and "no_baseline" in row["flags"]


def test_a_failed_gate_alone_never_causes_inconclusive():
    for names in (("B3",), ("B4",), ("B5",), ("B3", "B4"), ("B4", "B5"), ("B3", "B4", "B5")):
        for e in (HIGH, ZERO):
            for b in (HIGH, LOW, ZERO):
                row = decide(e, b, failing(*names))
                if row["verdict"] == a5.INCONCLUSIVE:
                    assert e is ZERO and "B4" not in names and row["reasons"] == ["evidence_unresolved"]


def test_row_without_verdict_or_code_stops_the_analysis():
    with pytest.raises(RuntimeError):
        a5.check_row({"verdict": "Something", "reasons": []})
    with pytest.raises(RuntimeError):
        a5.check_row({"verdict": a5.INCONCLUSIVE, "reasons": []})
    with pytest.raises(RuntimeError):
        a5.check_row({"verdict": a5.INCONCLUSIVE, "reasons": ["other"]})


def test_b_and_e_star_are_separate_families(world):
    cells = [built(world, rec.base_spec(prior=4.0, signal=2.0), seed=s).est for s in range(2)]
    tests = a5.family_tests(cells)
    assert cells[0].B_row.mean() < 0 and tests[0]["p_B"] < D["alpha"]  # two-sided finds a negative B
    e_only = si.signflip_maxt(np.column_stack([c.E_star_row for c in cells]), alternative=[1, 1])
    assert [t["p_E"] for t in tests] == e_only["p_maxt"]
    rows = np.column_stack([c.B_row for c in cells])
    boot = a5.bootstrap_t_maxt(rows, [0, 0])["p_maxt"]
    flip = si.signflip_maxt(rows, alternative=[0, 0])["p_maxt"]
    assert [t["p_B"] for t in tests] == [max(a, b) for a, b in zip(boot, flip)]


def test_step_down_covers_only_eligible_cells(world):
    good = built(world, rec.base_spec(signal=2.0, w="flat"))
    bad = built(world, rec.base_spec(signal=2.0, w="flat"), seed=1, gates={**PASSED, "rq0": False})
    rows = a5.analyse_family([good, bad])
    assert rows[1]["verdict"] == a5.NOT_TESTED and rows[1]["p"] is None and rows[1]["T"] > 0
    assert rows[0]["verdict"] != a5.NOT_TESTED


def test_survival_conflict_is_flagged_only_beside_evidence_led(world, monkeypatch):
    monkeypatch.setitem(a5.D, "min_balance_cells", 1)  # this test is about the flag, not the family size
    cell = built(world, rec.base_spec(signal=2.0, w="flat"))
    rows = a5.analyse_family([cell])
    assert rows[0]["verdict"] == "Evidence-led"
    original = a5.survival
    try:
        a5.survival = lambda *a, **k: {"S": 0.9, "ci95": [0.8, 1.0], "n_correct_items": 50, "n_correct_actors": 10}
        assert "survival_conflict" in a5.analyse_family([cell])[0]["flags"]
    finally:
        a5.survival = original


# ---------------------------------------------------------------------------
# E* and B are not B4 or B5
# ---------------------------------------------------------------------------


def test_observer_passes_e_star_and_fails_b4_and_b5(world):
    cell = built(world, rec.base_spec(signal=2.0, mode="wrong", w="flat"))
    row = a5.analyse_family([cell])[0]
    assert row["evidence"] == "Unmapped"
    assert row["gate_b"]["B4"] is False and row["gate_b"]["B5"] is False


def test_observer_passes_b4_and_b5_with_b_below_zero(world):
    cell = built(world, rec.base_spec(prior=4.0, signal=2.0))
    row = a5.analyse_family([cell])[0]
    assert row["gate_b"]["B4"] is True and row["gate_b"]["B5"] is True
    # One cell is below min_balance_cells, so the class is withheld and printed beside it.
    assert row["B"]["ci95"][1] < 0 and row["balance"] == "Unresolved" and row["balance_withheld"] == "Default"


# ---------------------------------------------------------------------------
# End to end on the pinned paths
# ---------------------------------------------------------------------------

DELIVERED = {
    "gate_2_every_frame_changes_output": True,
    "gate_3_logged_indices_are_the_frames_sent": True,
    "gate_4_ids_unchanged_within_clip": True,
    "ids_unchanged_under_shuffle": True,
    "pixels_change_under_shuffle": True,
    "pixels_arrive_in_sent_order": True,
    "repeat_forward_max_abs": 0.0,
    "tokens_across_view": 16.0,
}


def write_shards(folder, world, a2_hash, model="gemma4", n_shards=2):
    spec = rec.base_spec(signal=2.0, w="flat")
    rng = np.random.default_rng(3)
    folder.mkdir(exist_ok=True)
    bases, all_rows = [], {}
    for route in a5.ROUTES:
        s, w, g = world.scores(spec, rng)
        bases.append(world.report(w, g)["baselines"])
        for row in world.rows(s):
            all_rows[(row["trial_id"], route)] = row
    keep = {"trial_id", "actor", "label", "human_modal_label", "weak_label"}
    trials = sorted({k[0] for k in all_rows})
    words = run.lexicon_words()
    for shard in range(n_shards):
        lines = []
        for trial in trials[shard::n_shards]:
            for route in a5.ROUTES:
                source = all_rows[(trial, route)]
                base = {k: source[k] for k in keep}
                for condition in run.INPUTS[route]["conditions"]:
                    row = {**base, "input_name": route, "condition": condition, "design_sha256": a2_hash}
                    if condition == "original":
                        scores = source["blind"]["family_logprob"]
                        word = [scores[f] for f, _ in words]
                        row.update({"status": "ok", "blind": {"family_logprob": scores, "lexicon_mass": 0.5, "word_logprob": word},
                                    "media_sha256": "x"})
                    else:
                        row.update({"status": "failed", "reason": "not_in_fixture"})
                    lines.append(json.dumps(row))
        (folder / f"cells_{model}_shard{shard}.jsonl").write_text("\n".join(lines) + "\n")
        rows = []
        for route, base in zip(a5.ROUTES, bases):
            rows += [{**b, "input_name": route} for b in base]
        report = {"model": model, "smoke": False, "design_sha256": a2_hash, "shard": shard, "n_shards": n_shards,
                  "baselines": rows, "delivery": [{r: dict(DELIVERED) for r in a5.ROUTES}]}
        (folder / f"run_{model}_shard{shard}.json").write_text(json.dumps(report))


def pinned_paths(folder, monkeypatch, model="gemma4"):
    """Fixture files under A2's real names. Their content is synthetic, not any result."""
    summary = json.dumps({model: {"vision_dependent": True}}).encode()
    monkeypatch.setattr(an, "A3_SUMMARY_SHA256", hashlib.sha256(summary).hexdigest())
    (folder / an.A3_SUMMARY_NAME).write_bytes(summary)
    (folder / "rotation_summary.json").write_text(json.dumps({model: {"vision_dependent": True}}))
    (folder / f"vision_gates_{model}.json").write_text(json.dumps(
        {"smoke": False, "verdict": {g: {"pass": True} for g in ("G2", "G3", "G4")}}))
    (folder / f"input_delivery_{model}.json").write_text(json.dumps(
        {"verdicts": {r: {"pass": True} for r in a5.ROUTES}}))


def test_gate_loader_returns_true_on_the_pinned_paths(tmp_path, monkeypatch):
    pinned_paths(tmp_path, monkeypatch)
    delivered = [{"delivery": [{r: dict(DELIVERED) for r in a5.ROUTES}]}]
    for route in a5.ROUTES:
        gates = a5.route_gates(delivered, tmp_path, "gemma4", route, tmp_path / "nowhere")
        assert gates["rq0"] is True and gates["g2_g4"] is True and gates["delivery"] is True


def test_a_failed_gate_gives_not_tested_never_inconclusive(tmp_path, world):
    delivered = [{"delivery": [{r: dict(DELIVERED) for r in a5.ROUTES}]}]
    gates = a5.route_gates(delivered, tmp_path, "gemma4", "single_face", tmp_path)
    cell = built(world, rec.base_spec(signal=2.0, w="flat"), gates=gates)
    assert a5.analyse_family([cell])[0]["verdict"] == a5.NOT_TESTED


def test_missing_gate_files_stop_the_analysis(tmp_path, world, human):
    a2_hash = D["a2_design_sha256"]
    folder = tmp_path / "cells"
    write_shards(folder, world, a2_hash)
    runs = {"gemma4": a5.load_model("gemma4", folder, a2_hash), "qwen3vl": None}
    with pytest.raises(SystemExit, match="Gate files missing"):
        a5.analyse_runs(runs, tmp_path, human, None, tmp_path, shuffles=5)


def test_default_paths_resolve_against_the_repo():
    assert a5.A3_DIR.is_absolute() and a5.A3_DIR == ROOT / an.A3_OUTPUT_DIR


def test_first_shard_baseline_row_is_used_whatever_its_status():
    free = {f: -1.0 for f in a5.FAMILIES}
    failed = {"baselines": [{"input_name": "single_face", "condition": "image_free", "status": "failed"},
                            {"input_name": "single_face", "condition": "grey", "status": "ok", "family_logprob": free, "lexicon_mass": 0.1}]}
    out = a5.baselines([failed, report(free)], "single_face")
    assert out["image_free_family_logprob"] is None and out["ok"] is False
    assert a5.baselines([report(free), failed], "single_face")["image_free_family_logprob"] == free


def test_code_pins_hold_and_a_change_is_refused(tmp_path):
    design.check_code_pins()
    pins = design.code_pins()
    assert {"scripts/a5_analyse.py", "scripts/a5_design.py", "research/scripts/analyse_rotation.py",
            "src/paper_v2/option_score.py"} <= set(pins)
    for name in pins:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    (tmp_path / "scripts/a5_analyse.py").write_bytes(b"# changed\n")
    with pytest.raises(SystemExit, match="not the pinned"):
        design.check_code_pins(tmp_path)


def test_end_to_end_from_shards_to_a_withheld_verdict(tmp_path, world, human, monkeypatch):
    a2_hash = D["a2_design_sha256"]
    pinned_paths(tmp_path, monkeypatch)
    folder = tmp_path / "cells"
    write_shards(folder, world, a2_hash)
    (folder / "cells_gemma4_smoke.jsonl").write_text("not a real shard\n")
    runs = {"gemma4": a5.load_model("gemma4", folder, a2_hash), "qwen3vl": a5.load_model("qwen3vl", folder, a2_hash)}
    assert runs["qwen3vl"] is None
    out = a5.analyse_runs(runs, tmp_path, human, None, tmp_path / "nowhere", shuffles=SHUFFLES)
    rows = {(r["model"], r["route"]): r for r in out["cells"]}
    assert len(rows) == 4
    assert rows[("qwen3vl", "single_face")]["verdict"] == a5.NOT_TESTED
    for route in a5.ROUTES:
        row = rows[("gemma4", route)]
        # Two eligible cells is below min_balance_cells: evidence is named, balance is withheld.
        assert row["verdict"] == a5.INCONCLUSIVE and row["reasons"] == ["balance_small_family"], row["reasons"]
        assert row["evidence"] == "Shown" and row["balance_withheld"] == "Face"
        assert row["baseline_ok"] is True and row["n_clips"] == 243
    assert len(out["h1"]) == 2 and out["h3"]["status"] == "not tested"


def test_load_refuses_wrong_design_hash(tmp_path, world):
    write_shards(tmp_path / "cells", world, "wrong")
    with pytest.raises(SystemExit, match="Design hashes"):
        a5.load_model("gemma4", tmp_path / "cells", D["a2_design_sha256"])


def test_main_refuses_until_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["a5_analyse", "--output", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="not frozen"):
        a5.main()


# ---------------------------------------------------------------------------
# Humans
# ---------------------------------------------------------------------------


def test_h1_sign_and_template_form(world, human):
    """An observer that copies the human shares must move towards people, beyond the default."""
    j, *_ = judged(world, rec.base_spec(prior=1.0))
    q = j.q.copy()
    for i, trial in enumerate(j.trials):
        named = {a5.OPTION_TO_FAMILY[k]: v for k, v in human[trial].items() if k != a5.ABSTAIN}
        q[i] = 0.001
        for family, value in named.items():
            q[i, a5.FAMILIES.index(family)] = max(value, 0.001)
    q /= q.sum(1, keepdims=True)
    j = a5.Judgement(j.trials, q, j.actor, j.label, j.target, j.actors, j.labels, j.mass)
    rows = a5.human_divergence_rows(j, human)
    means = a5.actor_means(rows["default"], j.actor, len(j.actors))
    assert np.nanmean(means) > 0 and np.isfinite(rows["template"]).all()


def test_h1_runs_its_own_two_sided_family(world, human):
    good = built(world, rec.base_spec(signal=2.0, w="flat"))
    out = a5.h1([good], human)["synthetic|single_face"]
    assert 0 <= out["p_maxt_two_sided"] <= 1 and out["delta_J"]["ci95"] is not None


def test_h2_bootstrap_matches_a_direct_within_label_r():
    rng = np.random.default_rng(1)
    group, actor = np.repeat(np.arange(4), 25), rng.integers(0, 10, 100)
    x, y = rng.normal(size=100), rng.normal(size=100)
    y = x + y
    out = a5.within_label_r(x, y, group, actor, 10, draws=500)
    dx, dy = x.copy(), y.copy()
    for g in range(4):
        dx[group == g] -= dx[group == g].mean()
        dy[group == g] -= dy[group == g].mean()
    assert out["r"] == pytest.approx(np.corrcoef(dx, dy)[0, 1], abs=1e-9)
    assert out["ci95"][0] < out["r"] < out["ci95"][1]


def test_h2_within_label_ignores_label_means():
    x = np.array([0.1, 0.2, 5.1, 5.2])
    y = np.array([0.2, 0.1, 9.2, 9.1])
    out = a5.within_label_r(x, y, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), 2, draws=50)
    assert out["r"] == pytest.approx(-1.0)


def test_h2_and_repeats_on_real_labels(world, human):
    cell = built(world, rec.base_spec(signal=2.0, w="flat"))
    items = {i["trial_id"]: i for i in world.items}
    acted = {t: i["label"] for t, i in items.items()}
    modal = {t: i["human_modal_label"] for t, i in items.items()}
    weak = [t for t, i in items.items() if i["weak_label"]]
    out = a5.human_repeats(cell, human, modal, acted, weak)
    assert set(out) == {"delta_J_without_flagged", "delta_J_against_modal_point_mass", "h2_acted", "h2_without_flagged", "h2_against_human_mode"}
    assert out["h2_without_flagged"]["n_items"] == 243 - 26
    assert a5.leakage_table(cell.j, human, acted, weak)["n_items"] == 26


def votes():
    import csv

    rows = list(csv.DictReader((REPRO / "crema_d_selection.csv").open()))
    actors = sorted({r["actor_id"] for r in rows})
    options = "ADFHNS"
    table = np.array([[float(r["votes_" + c]) for c in options] for r in rows])
    return table / table.sum(1, keepdims=True), np.array([actors.index(r["actor_id"]) for r in rows]), \
        np.array([options.index(r["emotion"]) for r in rows]), len(actors)


def test_h3_needs_no_smoothing_on_the_real_votes():
    q, actor, label, n_actors = votes()
    assert q.shape == (819, 6) and n_actors == 91
    used = a5.templates(q, actor, label, n_actors, 6)[label, actor]
    assert (used[q > 0] > 0).all()
    row = a5.lambda_share(q, actor, label, n_actors, 6)
    assert np.isfinite(row["E"]).all() and np.isfinite(row["P"]).all()


def test_h3_difference_is_zero_for_the_same_data_and_detects_a_prior():
    q, actor, label, n_actors = votes()
    same = a5.h3_difference(q, q, actor, label, n_actors, 6, draws=200)
    assert same["difference"] == 0 and same["within_lambda_margin"] is True
    prior = np.array([0.05, 0.05, 0.05, 0.05, 0.75, 0.05])
    leaning = 0.4 * q + 0.6 * prior
    other = a5.h3_difference(q, leaning, actor, label, n_actors, 6, draws=200)
    assert other["lambda_model"] < other["lambda_human"]


# ---------------------------------------------------------------------------
# Sensitivities
# ---------------------------------------------------------------------------


def test_every_sensitivity_runs_and_none_changes_a_verdict(world):
    spec = rec.base_spec(prior=0.5, signal=2.0)
    rng = np.random.default_rng(0)
    s, w, g = world.scores(spec, rng)
    rows = world.rows(s)
    words = run.lexicon_words()
    for row in rows:
        row["blind"]["word_logprob"] = [row["blind"]["family_logprob"][f] for f, _ in words]
    cells = an.Cells(rows)
    cell = a5.build_cell("synthetic", rec.ROUTE, cells, world.actors, world.partners,
                         a5.baselines([world.report(w, g)], rec.ROUTE), dict(PASSED), 0.0, SHUFFLES)
    before = a5.analyse_family([cell])[0]["verdict"]
    canonical = a5.judgement(an.Cells(an.canonical_rows(rows)), rec.ROUTE, world.actors)
    items = {i["trial_id"]: i for i in world.items}
    modal = {t: i["human_modal_label"] for t, i in items.items()}
    weak = [t for t, i in items.items() if i["weak_label"]]
    small = sorted(items)[:30]
    out = a5.sensitivities(cell, canonical, modal, small, weak, shuffles=20)
    assert set(out) == {
        "without_smoke_items", "without_flagged_items", "templates_by_human_modal_label", "clips_at_token_floor_or_more",
        "canonical_word_only", "flat_reference_by_family_size", "default_with_all_actors", "temperature",
        "basic_labels_only_descriptive", "complex_labels_only_descriptive", "tables_by_label"}
    assert set(out["temperature"]) == {str(t) for t in D["temperature_sensitivity"]}
    assert out["templates_by_human_modal_label"]["clips_without_template"] >= 0
    assert all(row["B"]["mean"] is not None for key, row in out.items() if key in ("without_smoke_items", "without_flagged_items"))
    assert out["tables_by_label"]["Happy"]["n_clips"] == 15 and "W" in out["tables_by_label"]["Happy"]
    assert a5.analyse_family([cell])[0]["verdict"] == before


def test_size_reference_changes_p_but_not_d_or_e(world):
    j, w, g, _ = judged(world, rec.base_spec(prior=0.5, signal=1.0))
    a, b = a5.decompose(j, w, g), a5.decompose(j, w, g, a5.size_reference())
    assert np.array_equal(a["D"], b["D"]) and np.array_equal(a["E"], b["E"])
    assert not np.allclose(a["P"], b["P"])


def test_token_floor_sensitivity_with_no_clip_left_reports_none(world):
    cell = built(world, rec.base_spec(signal=2.0, w="flat"))
    items = {i["trial_id"]: i for i in world.items}
    modal = {t: i["human_modal_label"] for t, i in items.items()}
    out = a5.sensitivities(cell, cell.j, modal, sorted(items), [], shuffles=5)
    assert out["clips_at_token_floor_or_more"] is None


# ---------------------------------------------------------------------------
# Recovery machinery
# ---------------------------------------------------------------------------


def test_fast_statistics_equal_the_frozen_ones():
    rng = np.random.default_rng(0)
    actors = np.array(list("abcdefghij"))[rng.integers(0, 10, 80)]
    delta = rng.normal(0.2, 1, 80)
    assert rec.fast_sign_flip_p(delta, actors) == actor_sign_flip_p(delta, actors)
    assert rec.fast_bootstrap_ci(delta, actors) == pytest.approx(actor_bootstrap_ci(delta, actors), abs=1e-12)


def test_recovery_rows_are_a2_format_and_no_model_output(world):
    cells = an.Cells(world.rows(world.scores(rec.base_spec(), np.random.default_rng(0))[0]))
    assert len(cells.ok) == 243 and not cells.failed
    assert set(next(iter(cells.ok.values()))["blind"]) == {"family_logprob", "lexicon_mass"}
    source = (ROOT / "scripts/a5_recovery.py").read_text()
    assert "eval_v2" not in source and "same_clip_family_model" not in source


def test_noise_has_actor_family_correlation(world):
    noise = world.family_noise(np.random.default_rng(0), 20000)
    corr = np.corrcoef(noise.T)
    assert abs(noise.var(0).mean() - 1) < 0.05
    assert np.abs(corr[np.triu_indices(rec.F, 1)]).max() > 0.2


def test_prior_calibration_hits_phi(world):
    spec = rec.prior_for_phi(world, 0.5, rec.base_spec(signal=1.0))
    assert rec.shares(world, spec)[0] == pytest.approx(0.5, abs=0.005)


def test_balance_zero_calibration_puts_b_at_zero(world, monkeypatch):
    monkeypatch.setitem(rec.R, "balance_zero_calibration_draws", 300)
    spec = rec.prior_for_balance_zero(world, rec.base_spec(signal=2.0, gain=0.5))
    assert rec.balance_over_total(world, spec) == pytest.approx(0.0, abs=0.002)
    assert rec.balance_over_total(world, {**spec, "prior": spec["prior"] * 0.9}) > 0


def test_every_observer_in_the_draft_exists(world, monkeypatch):
    # The recovery uses the full balance_zero_calibration_draws; names do not depend on it.
    monkeypatch.setitem(rec.R, "balance_zero_calibration_draws", 100)
    names = {o["name"] for o in rec.observers(world)}
    assert {"prior_only_word", "prior_only_face_default", "evidence_only", "identity", "noise", "spread"} <= names
    assert any(n.startswith("unmapped_face") for n in names) and any(n.startswith("unmapped_default") for n in names)
    assert any(n.startswith("boundary") for n in names)
    assert "all_null_family" in names and any(n.startswith("false_even") for n in names)
    assert {"all_null_family|companions0", "gain0.5_phi0.5|companions0", "identity|companions1"} <= names
    assert any(n.startswith("inside_even") for n in names)
    assert {f"mixed_phi{p}_signal{s}" for p in D["recovery"]["phi_grid"] for s in D["recovery"]["signal_grid"]} <= names


def test_pass_rule_reads_the_block():
    observer = {"expected": "Even", "judged": True, "spec": {"prior": 0.0, "signal": 1.0}}
    row = dict(verdict="Even", reasons=[], share="mixed", evidence="Shown", balance="Even", phi=0.5, T=1.0, b_over_t=0.0,
               e_over_t=0.1, b_half_over_t=0.05, e_half_over_t=0.05, b4=True, b5=True, e_sig=True, e_raw_sig=True, e_passes=True,
               b_sig=False, b_raw_sig=False, b_named=False, balance_withheld=None, any_b_sig=False, any_e_sig=True,
               gain_log_sd=None, family_gain_log_sd=None, **{"would_fire_at_0.4": False},
               conflict=False, b3_failed=False)
    good = rec.summarise(observer, [row] * 8 + [{**row, "verdict": a5.INCONCLUSIVE, "reasons": ["balance_unresolved"]}] * 2)
    assert good["pass"] is True
    bad = rec.summarise(observer, [row] * 7 + [{**row, "verdict": a5.INCONCLUSIVE, "reasons": ["balance_unresolved"]}] * 3)
    assert bad["pass"] is False


def test_design_block_constants_are_the_only_ones_in_a5_analysis():
    source = (ROOT / "scripts/a5_analyse.py").read_text()
    for literal in ("10000", "0.05", "0.95", "0.90", "1000,"):
        lines = [l for l in source.splitlines() if literal in l and not l.strip().startswith(("#", '"""'))]
        assert not lines, lines


def test_false_even_check_covers_every_phi_outside_the_range():
    row = {"rates": {"e_passes": 0.0, "b4": 0.0, "b_sig": 0.0, "any_b_sig": 0.0, "any_e_sig": 0.0}, "verdicts": {},
           "balance_classes": {"Even": 0.2}}
    table = {"outside": row, "inside": row, "edge": row}
    meta = {"outside": {"phi": 0.32}, "inside": {"phi": 0.4}, "edge": {"phi": 0.3334}}
    out, below = rec.false_positive_checks(table, meta)
    assert set(out) == {"outside|false_even"} and out["outside|false_even"]["pass"] is False and not below


def test_balance_is_withheld_below_min_balance_cells():
    b = {"mean": 0.5, "ci95": [0.2, 0.8], "ci90": [0.3, 0.7], "neither_dominant_lower_bounds": [-1.0, 1.0]}
    e = {"mean": 0.5, "ci95": [0.2, 0.8], "ci90": [0.3, 0.7]}
    small = a5.decide(PASSED, FULL_GATE, T, MARGIN, b, e, 0.001, 0.001, True, family_cells=D["min_balance_cells"] - 1)
    assert small["verdict"] == a5.INCONCLUSIVE and small["reasons"] == ["balance_small_family"]
    assert small["balance"] == "Unresolved" and small["balance_withheld"] == "Face"
    full = a5.decide(PASSED, FULL_GATE, T, MARGIN, b, e, 0.001, 0.001, True, family_cells=D["min_balance_cells"])
    assert full["balance"] == "Face" and full["balance_withheld"] is None


def test_small_family_judges_the_named_claim_not_the_raw_test():
    row = {"rates": {"e_passes": 0.0, "b4": 0.0, "b_sig": 0.12, "b_named": 0.0, "any_b_sig": 0.0, "any_e_sig": 0.0,
                     "any_b_claim": 0.0, "b_sig_open": 0.12, "guard_fires": 0.0},
           "verdicts": {}, "balance_classes": {}}
    out, below = rec.false_positive_checks({"one": row, "four": row}, {"one": {"true_balance_zero": True, "companions": 0},
                                                                        "four": {"true_balance_zero": True}})
    assert "one|B_dominance_named" not in out and "one|B_significant" not in out
    assert below["one|B_significant"]["judged"] is False and below["one|B_dominance_named"]["why"].startswith("vacuous")
    assert out["four|B_significant"]["pass"] is False and out["four|B_dominance_named"]["pass"] is True


def test_false_even_covers_observers_placed_without_a_stated_phi():
    row = {"rates": {"e_passes": 0.0, "b4": 0.0, "b_sig": 0.0, "b_named": 0.0, "any_b_sig": 0.0, "any_e_sig": 0.0,
                     "any_b_claim": 0.0, "b_sig_open": 0.0, "guard_fires": 0.0},
           "verdicts": {}, "balance_classes": {"Even": 0.2}}
    out, _ = rec.false_positive_checks({"evidence_only": row, "null": row}, {"evidence_only": {"true_phi": 0.97}, "null": {"true_phi": 0.5}})
    assert set(out) == {"evidence_only|false_even"}


def test_shared_family_is_judged_familywise():
    row = {"rates": {"e_passes": 0.0, "b4": 0.0, "b_sig": 0.02, "b_named": 0.02, "any_b_sig": 0.09, "any_e_sig": 0.0},
           "verdicts": {}, "balance_classes": {}}
    out, _ = rec.false_positive_checks({"s": row}, {"s": {"true_balance_zero": True, "shared_family": True, "companions": 3}})
    assert out["s|any_B_column"]["pass"] is False and out["s|B_significant"]["pass"] is True


def test_shared_family_columns_share_gains(world):
    spec = rec.base_spec(signal=2.0, gain=0.5, item_sd=0.0, actor_sd=0.0)
    gain = np.random.default_rng(1).normal(0, 0.5, len(world.actors))
    a = world.scores(spec, np.random.default_rng(2), gain)[0]
    b = world.scores(spec, np.random.default_rng(3), gain)[0]
    assert np.allclose(a, b)


def test_family_diagnostic_interval():
    from scripts import a5_family_diagnostic as diag

    low, high = diag.wilson(152, 2000)
    assert low < 0.076 < high and high - low < 0.03


def test_power_table_ignores_companion_sweep_copies():
    row = {"power": 0.9, "verdicts": {}, "rates": {"e_passes": 0.0, "b4": 0.0, "b_sig": 0.0, "b_named": 0.0,
           "any_b_sig": 0.0, "any_e_sig": 0.0, "would_fire_at_0.4": 0.0},
           "balance_classes": {}, "judged": False, "word_share": {}, "gain_log_sd": {}, "family_gain_log_sd": {}}
    table = {"m": row, "m|companions0": {**row, "power": 0.0}}
    spec = {"gain": 0.3, "signal": 2.0}
    meta = {"m": {"grid": True, "phi": 0.5, "signal": 2.0, "spec": spec},
            "m|companions0": {"grid": True, "phi": 0.5, "signal": 2.0, "companions": 0, "spec": spec}}
    assert rec.build_report(table, meta, 10)["power_by_phi_and_signal"] == {"0.5": {"2.0": 0.9}}


def test_gain_heterogeneity_tracks_the_true_spread(world):
    spec = rec.base_spec(signal=2.0, item_sd=0.2, actor_sd=0.0)
    estimates = {}
    for sd in (0.0, 0.5):
        values = [a5.gain_heterogeneity(world.judgement(world.scores({**spec, "gain": sd}, np.random.default_rng(k))[0]))
                  for k in range(20)]
        assert all(v["estimable"] for v in values)
        estimates[sd] = np.mean([v["log_sd"] for v in values])
    assert estimates[0.0] < 0.2 and 0.3 < estimates[0.5] < 0.7


def test_no_push_is_not_estimable(world):
    j = world.judgement(world.scores(rec.base_spec(prior=0.5), np.random.default_rng(0))[0])
    h = a5.gain_heterogeneity(j)
    assert h["estimable"] is False and h["log_sd"] is None


def test_gain_estimate_changes_no_verdict():
    assert "balance_gain_heterogeneity" not in a5.REASONS
    assert "gain_log_sd" not in a5.decide.__code__.co_varnames
