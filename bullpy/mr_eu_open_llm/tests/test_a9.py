"""Tests for draft amendment A9 (Liu et al. 2022 Experiment II). No model, render or model output."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import a9_human_target as ht  # noqa: E402
import a9_manifest as man  # noqa: E402
import human_au_effects as hae  # noqa: E402
import render_au_faces as raf  # noqa: E402

from scripts import a4_analyse as an  # noqa: E402
from scripts import a4_design as a4  # noqa: E402
from scripts import a9_analyse as ana9  # noqa: E402
from scripts import a9_design as a9  # noqa: E402
from scripts import a9_recovery as rec9  # noqa: E402
from scripts import a9_run as run9  # noqa: E402

LIU = ROOT / a9.LIU_DIR
HAVE_LIU = all((LIU / n).exists() for n in a9.LIU_FILES)
HAVE_SNOEK = ht.SNOEK_ZIP.exists()

# A4's frozen code, sha256 as listed in research/PREREG_A4_DRAFT.md. A9 must not change any of it.
A4_FROZEN = {
    "scripts/a4_design.py": "a0bf7bb1c69ec7af1f11fabb5a9f31a6c694a30b92e624f1e1c241f161d062a8",
    "scripts/a4_analyse.py": "30f8f529d128b555ccfbf4d31ac3de70d91f90b1f5bfd52360ae9f39305fe56d",
    "scripts/a4_recovery.py": "12b15e89a366d772078034fd56b105532b347f1b73afe0b4efd5548ad3302526",
    "research/scripts/a4_stimuli.py": "d0dff6930ede7f908e8d7faee2bc647adeb02f366601d04bea47353833715875",
    "research/scripts/render_au_faces.py": "6bdfa4e947b06aa7c2213dc31fbd297df4f8b97ca045a40905c6de4800c001d4",
    "research/scripts/human_au_effects.py": "c02a945ba578156cc4394e7d3a49761ad6e74495585034fbb059a4ce40199a2b",
    "research/reproduction/human_au_effects.json": "58c0f082cbf6aa4827054188ac2fc47da58a254c38da2c7ac202e8f52399310c",
    "research/reproduction/a4_human_ceiling.json": "21fe4f8441532992ecb22b60fba4afc59c38e70aed5407b2d0e8ab48b76cd2eb",
    "research/reproduction/a4_recovery.json": "1ebfc8ea00da4261e3a05e6de09b3fbe7bd259ce0a5d840fe4c7ab03894bedca",
}


@pytest.mark.parametrize("path,digest", sorted(A4_FROZEN.items()))
def test_a4_frozen_files_unchanged(path, digest):
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


# ---------------------------------------------------------------- design and map


def test_columns_and_grid_match_a4():
    assert list(a9.AU_COLS) == list(hae.AU_COLS) == list(raf.AU_COLS)
    assert tuple(a9.LAMBDA_GRID) == tuple(hae.LAMBDA_GRID)


def test_shared_constants_are_a4s():
    assert a9.STATISTIC == a4.STATISTIC == "residual"
    assert a9.SESOI_CFA == a4.SESOI_CFA
    assert a9.OPTIONS == a4.OPTIONS["basic"] and a9.HUMAN_LABELS == a4.HUMAN_LABELS["basic"]
    assert a9.RECOVERY is a4.RECOVERY and a9.MODELS == a4.MODELS
    assert a9.RECOVERY_WITHDRAWN == ()
    assert not any(k.startswith(a9.DATASET) for k in a4.RECOVERY["pair_error_withdrawn"])


def test_response_codes_follow_released_code_order():
    # PMI_exp2.m: {'happy','surprise','fear','disgust','anger','sad','other'}
    assert [a9.RESPONSE_CODES[k] for k in range(1, 8)] == [
        "happy", "surprise", "fear", "disgust", "anger", "sadness", "other"]
    assert set(a9.RESPONSE_CODES.values()) == set(a9.HUMAN_LABELS)


def test_au_map_examples():
    m = a9.AU_MAP
    assert len(m) == 42
    assert m["AU1"] == ("AU01",)
    assert m["AU1-2"] == ("AU01", "AU02L", "AU02R")
    assert m["AU2"] == ("AU02L", "AU02R") and m["AU2L"] == ("AU02L",)
    assert m["AU7"] == ("AU07L", "AU07R") and m["AU7R"] == ("AU07R",)
    assert m["AU10Open"] == ("AU10L", "AU10R") and m["AU10LOpen"] == ("AU10L",)
    assert m["AU16Open"] == ("AU16",) and m["AU27i"] == ("AU27",)
    assert m["AU25-12"] == ("AU25", "AU12L", "AU12R")
    assert m["AU12-6"] == ("AU12L", "AU12R", "AU06L", "AU06R")
    assert m["AU11L"] == ("AU11L",)


def test_au_map_covers_every_column_and_family():
    covered = {c for cols in a9.AU_MAP.values() for c in cols}
    assert covered == set(a9.AU_COLS)
    rows = a9.family_table()
    assert len(rows) == 25 and all(r["liu_labels"] for r in rows)
    assert a9.UNCOVERED == ()


def test_every_column_has_a_render_target():
    for column in a9.AU_COLS:
        status, _note = raf.COVERAGE[column]
        assert status in ("direct", "approximated")
    assert {c for c, (s, _n) in raf.COVERAGE.items() if s == "approximated"} == {"AU22", "AU27"}


def test_liu_to_columns_max_rule_and_rounding():
    out = a9.liu_to_columns({"AU12": 0.3, "AU25-12": 0.712345, "AU1": 0.0, "AU2L": 0.00001})
    assert out == {"AU12L": 0.7123, "AU12R": 0.7123, "AU25": 0.7123}
    raf.check_config(out)


def test_block_hash_is_pinned_and_stated():
    assert a9.block_hash() == a9.DESIGN_SHA256 == a9.stated_hash() == a9.design_hash()


def test_tampered_block_is_refused(tmp_path):
    text = a9.DRAFT.read_text(encoding="utf-8")
    bad = tmp_path / "draft.md"
    bad.write_text(text.replace('"replicate_configs": 200', '"replicate_configs": 201'), encoding="utf-8")
    with pytest.raises(SystemExit):
        a9.load_design(bad)
    stated = tmp_path / "stated.md"
    stated.write_text(text.replace(f"block below: `{a9.DESIGN_SHA256}`", "block below: `" + "0" * 64 + "`"),
                      encoding="utf-8")
    with pytest.raises(SystemExit):
        a9.load_design(stated)


def test_block_pins_hold():
    a9.check_pins()
    for name, digest in A4_FROZEN.items():
        if name in a9.DESIGN["a4_pins"]:
            assert a9.DESIGN["a4_pins"][name] == digest


def test_code_pins_required_means_not_frozen_before_the_freeze(tmp_path):
    unpinned = tmp_path / "draft.md"
    unpinned.write_text(a9.DRAFT.read_text(encoding="utf-8").split("## Code hashes")[0], encoding="utf-8")
    with pytest.raises(SystemExit):
        a9.check_code_pins(path=unpinned, require=True)
    assert a9.check_code_pins(path=unpinned, require=False) == {}


def test_code_matches_pins_when_pinned():
    pins = a9.code_pins()
    if not pins:
        pytest.skip("A9 code not pinned yet")
    assert set(pins) == set(a9.A9_FILES)
    assert a9.check_code_pins(require=True) == pins


def test_runner_refuses_before_freeze(tmp_path):
    with pytest.raises(SystemExit):
        run9.preflight(tmp_path / "manifest.csv", frozen=False)


# ---------------------------------------------------------------- estimation on synthetic data


def synthetic(n=600, subjects=6, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 33)) < 0.1) * rng.random((n, 33))
    labels = list(a9.HUMAN_LABELS)
    eta = np.zeros((n, 7))
    eta[:, labels.index("happy")] += 4 * X[:, hae.AU_COLS.index("AU12L")] + 4 * X[:, hae.AU_COLS.index("AU12R")]
    eta[:, labels.index("surprise")] += 5 * X[:, hae.AU_COLS.index("AU27")]
    Y = np.empty((n, subjects), dtype=object)
    for s in range(subjects):
        p = np.exp(eta + rng.normal(0, 0.3, eta.shape))
        p /= p.sum(axis=1, keepdims=True)
        Y[:, s] = [labels[rng.choice(7, p=row)] for row in p]
    return X, Y, labels


def test_fit_recovers_planted_effects_and_sums_to_zero():
    X, Y, labels = synthetic()
    lam, losses = ht.choose_lambda(Y, X, labels, seed=1)
    assert lam in a9.LAMBDA_GRID and "intercept_only" in losses
    betas, dos, never = ht.fit_subject_effects(Y, X, labels, lam)
    names, _ = hae.effect_index()
    do = dos.mean(axis=0)
    assert np.abs(do[:, :33].sum(axis=0)).max() < 1e-8
    assert do[labels.index("happy"), names.index("AU12_bilateral")] > 0.2
    assert do[labels.index("surprise"), names.index("AU27")] > 0.2


def test_fit_matches_hae_on_one_subject_with_constant_gender_removed():
    """With no gender column, a9's per-subject fit is hae's fit on [1, X]."""
    X, Y, labels = synthetic(subjects=1)
    _betas, dos, _ = ht.fit_subject_effects(Y, X, labels, 1.0)
    Xf = np.column_stack([np.ones(len(X)), X])
    pen = np.r_[0.0, np.full(33, 1.0)]
    fitted = np.vstack([hae.fit_ridge_logit(Xf, (Y[:, 0] == l).astype(float), pen=pen) for l in labels])
    _, members = hae.effect_index()
    np.testing.assert_allclose(dos[0], hae.do_effects(X, np.ones((len(X), 1)), fitted, members))


def test_ceilings_run_and_are_high_for_clean_effects():
    X, Y, labels = synthetic(n=800, subjects=8)
    out = ht.do_ceiling(Y, X, labels, splits=4, seed=0)
    assert set(out) >= {"ceiling", "ceiling_residual", "split_half_r"}
    assert out["split_half_r"] > 0.3
    _b, dos, _ = ht.fit_subject_effects(Y, X, labels, 1.0)
    assert "ceiling" in ht.subjects_only_ceiling(dos, splits=4, seed=0)


# ---------------------------------------------------------------- manifest


def fake_p(n=40, seed=0):
    rng = np.random.default_rng(seed)
    p = np.zeros((42, n))
    for i in range(n):
        p[rng.choice(42, 3, replace=False), i] = rng.random(3)
    pseq = np.repeat(p[:, None, :], a9.FRAME_COUNT, axis=1) * np.linspace(0, 1, a9.FRAME_COUNT)[None, :, None]
    return p, pseq


def test_manifest_rows_render_ready(monkeypatch):
    monkeypatch.setattr(a9, "REPLICATE_CONFIGS", 5)
    p, pseq = fake_p()
    rows = man.build(p, pseq, "A")
    main = [r for r in rows if r["set"] == "main"]
    assert len(main) == 40 and len([r for r in rows if r["set"] == "replicate"]) == 5
    assert sum(r["gender"] == "F" for r in main) == 20
    for r in rows:
        assert set(r) == set(man.COLUMNS)
        raf.check_config(json.loads(r["au_json"]))
        assert (r["identity"], r["seed"]) in a9.IDENTITIES[r["gender"]]
    # frame_max at the last frame equals the peak here, so the image id repeats
    fm = {r["cfg"]: r["stim_id"] for r in rows if r["set"] == "frame_max"}
    assert all(fm[r["cfg"]] == r["stim_id"] for r in main)
    rep = [r for r in rows if r["set"] == "replicate"]
    main_face = {r["cfg"]: (r["identity"], r["seed"]) for r in main}
    assert all((r["identity"], r["seed"]) != main_face[r["cfg"]] for r in rep)


def test_manifest_design_b_has_all_four_faces():
    p, pseq = fake_p(n=10)
    rows = man.build(p, pseq, "B")
    for cfg in {r["cfg"] for r in rows}:
        faces = {(r["identity"], r["seed"]) for r in rows if r["cfg"] == cfg and r["set"] in ("main", "all_identities")}
        assert len(faces) == 4


def test_manifest_is_deterministic():
    p, pseq = fake_p()
    assert man.build(p, pseq) == man.build(p, pseq)


# ---------------------------------------------------------------- recovery wrapper


def small_store(X, Y, labels):
    lam = 1.0
    betas, dos, _ = ht.fit_subject_effects(Y, X, labels, lam)
    counts = {l: int((Y == l).sum()) for l in labels}
    return {"datasets": {a9.DATASET: {"labels": labels, "structure": {"label_counts": counts},
                                      "beta_log_odds_per_unit_amplitude": betas.mean(0).tolist(),
                                      "do_effect_probability_au_full_vs_none": dos.mean(0).tolist(),
                                      "ridge_lambda_selected": lam}}}


@pytest.mark.parametrize("design", a9.RECOVERY_RUNS)
def test_recovery_replicate_runs(design):
    X, Y, labels = synthetic(n=300, subjects=4)
    store = small_store(X, Y, labels)
    gender = np.where(np.arange(300) % 2, 0.5, -0.5)
    W, base = rec9.rec.human_weights(store, a9.DATASET)
    human, lam = an.human_matrix(store, a9.DATASET)
    task = ("human_like", a9.DATASET, 1.0, 0, X, gender, W, base, human, lam, 3, 19, 9, 2, design)
    out = rec9.one_replicate(task)
    assert out["run"] == design
    assert out["residual"]["decision"] in an.OUTCOMES
    assert -1 <= out["residual"]["truth"] <= 1


def test_correlated_noise_is_correlated():
    rng = np.random.default_rng(0)
    target, base, X = np.zeros((7, 33)), np.zeros(7), np.zeros((4000, 33))
    assert rec9.run_parts("B_rho0.25") == ("B", 0.25) and rec9.run_parts("A") == ("A", 0.0)
    spread = {rho: rec9.respond_b(target, base, X, rng, 1.0, rho).std() for rho in (0.0, 0.5)}
    assert spread[0.5] > spread[0.0] * 1.2  # shared noise survives averaging over faces


def test_r_star():
    ok, bad = {"residual": True}, {"residual": False}
    assert rec9.r_star({"B": ok, "B_rho0.25": ok, "B_rho0.5": ok}) == 0.5
    assert rec9.r_star({"B": ok, "B_rho0.25": ok, "B_rho0.5": bad}) == 0.25
    assert rec9.r_star({"B": ok, "B_rho0.25": bad, "B_rho0.5": ok}) == 0.0
    assert rec9.r_star({"B": bad}) is None
    assert not hasattr(ana9, "scope")  # lead decision 2026-11-09: no label on decisions


def test_face_correlation_never_reaches_the_decision():
    """The decision depends on the gates and the statistic only, whatever rho_hat is."""
    base = {"p_signal": 0.001, "residual": {"cfa": 0.6, "ci": (0.4, 0.8), "r_s": 0.95, "p_cue": 0.001},
            "lost_ok": True}
    decisions = []
    for rho_hat in (0.0, 0.3, 0.9, None):
        entry = {**base, "residual": dict(base["residual"]), "face_shared_residual_correlation": rho_hat}
        results = {"gemma4": entry}
        an.apply_rule(results, {"gemma4": {"all": True}}, a9.STATISTIC)
        decisions.append(results["gemma4"]["decision"])
        assert set(results["gemma4"]) >= {"face_shared_residual_correlation", "decision"}
    assert decisions == ["Aligned"] * 4
    assert a9.DESIGN["recovery"]["withdrawn_under_correlated_face_noise"] == ["liu_WC|0.25|pair_error"]


def test_choice_rule():
    assert rec9.choose({"A": {"residual": True}, "B": {"residual": True}}) == {"design": "A", "statistic": "residual"}
    assert rec9.choose({"A": {"residual": False}, "B": {"residual": True}}) == {"design": "B", "statistic": "residual"}
    assert rec9.choose({"A": {"residual": False, "raw": True}, "B": {"residual": False}}) == {
        "design": None, "statistic": None}


# ---------------------------------------------------------------- real data, when present


@pytest.mark.skipif(not HAVE_LIU, reason="Liu files not present")
def test_liu_files_structure():
    a9.check_liu_files()
    data = ht.load_liu()
    assert tuple(data["labels"]) == a9.LIU_LABELS
    assert data["labels_same_for_all_stimuli"]
    assert data["codes"].shape == (1200, 20)
    X = ht.design_matrix(data["p"])
    assert X.shape == (1200, 33) and X.max() <= 1 and (X > 0).sum(axis=1).min() >= 1


@pytest.mark.skipif(not HAVE_SNOEK, reason="Snoek zip not present")
def test_map_reproduces_snoek_recoding():
    out = ht.snoek_token_check()
    assert out["checked"] and out["mismatched"] == 0 and out["token_sets_equal"]


# ---------------------------------------------------------------- runner and analysis on synthetic cells


def test_planned_cells_cover_every_image_once(monkeypatch):
    monkeypatch.setattr(a9, "REPLICATE_CONFIGS", 5)
    p, pseq = fake_p(n=12)
    rows = man.build(p, pseq, "B")
    cells = [c for s in range(3) for c in run9.planned_cells(rows, s, 3)]
    images = {r["stim_id"] for r in rows}
    mcq = [c for c in cells if c["readout"] == "mcq"]
    assert len(mcq) == len(images) * a9.ROTATIONS
    assert len({run9.cell_key(c) for c in cells}) == len(cells)
    assert {c["stim_id"] for c in cells if c["readout"] == "image_free"} <= images


def synthetic_cells(rows, X_by_stim, store, rng):
    """Fake model cells: a human-like observer answering each image, every rotation."""
    W, base = rec9.rec.human_weights(store, a9.DATASET)
    out = []
    for stim, x in X_by_stim.items():
        eta = base + x @ W.T + rng.normal(0, 0.3, len(base))
        p = np.exp(eta - eta.max())
        p /= p.sum()
        for rot in range(a9.ROTATIONS):
            options = a9.rotated_options(rot)
            lp = [float(np.log(p[list(a9.OPTIONS).index(o)])) for o in options]
            out.append({"stim_id": stim, "task": "basic", "rotation": rot, "readout": "mcq", "options": options,
                        "joint_logprob": lp, "status": "ok"})
    return out


@pytest.mark.parametrize("design", a9.DESIGNS)
def test_analyse_model_end_to_end(monkeypatch, design):
    monkeypatch.setattr(a9, "REPLICATE_CONFIGS", 20)
    rng = np.random.default_rng(3)
    X, Y, labels = synthetic(n=300, subjects=4)
    store = small_store(X, Y, labels)
    store["datasets"][a9.DATASET]["presence_sensitivity"] = {
        "ridge_lambda_selected": 1.0, "do_family_25": np.array(store["datasets"][a9.DATASET][
            "do_effect_probability_au_full_vs_none"])[:, an.FAM].tolist()}
    p, pseq = fake_p(n=150, seed=4)
    rows = man.build(p, pseq, design)
    X_by_stim = {}
    for r in rows:
        x = np.zeros(33)
        for c, v in json.loads(r["au_json"]).items():
            x[hae.AU_COLS.index(c)] = v
        X_by_stim[r["stim_id"]] = x
    cells = synthetic_cells(rows, X_by_stim, store, rng)
    ceiling = {a9.DATASET: {"ceiling": 0.97, "ceiling_residual": 0.96}}
    out, P = ana9.analyse_model(cells, rows, store, ceiling, None, design, draws=5, cue_n=19, signal_n=9, splits=2)
    assert out["share_lost"] == 0 and out["n_configurations"] == 150 and P.shape == (150, 7)
    assert out["residual"]["cfa"] > 0.3
    assert set(out["sensitivities"]) >= {"presence", "frame_max", "without_AU22_AU27"}
    if design == "B":
        assert out["face_shared_residual_correlation"] is not None
    results = {"gemma4": dict(out)}
    an.apply_rule(results, {"gemma4": {"all": True}}, a9.STATISTIC)
    assert results["gemma4"]["decision"] in an.OUTCOMES


def test_face_shared_correlation_reads_shared_noise():
    rng = np.random.default_rng(0)
    X = rng.random((400, 33)) * (rng.random((400, 33)) < 0.1)
    shared = rng.normal(size=(400, 7))
    faces = [0.7 * shared + 0.7 * rng.normal(size=(400, 7)) for _ in range(4)]
    alone = [rng.normal(size=(400, 7)) for _ in range(4)]
    assert ana9.face_shared_correlation(faces, X) > 0.35
    assert abs(ana9.face_shared_correlation(alone, X)) < 0.1


# ---------------------------------------------------------------- audit fixes (2026-11-09)


def a4_summary_with(model, decision, gates=None):
    return {"datasets": {"basic_WC": {model: {"decision": decision, "gates": gates or {"all": True}, "p_signal": 0.001}}},
            "models": {model: {"headline": {"basic": "Inconclusive"}}}}


@pytest.mark.parametrize("a4_dec,a9_dec,expected", [
    ("Aligned", "Aligned", "replicated on a second human sample (same lab and generator)"),
    ("Not aligned", "Not aligned", "the Not aligned result replicated"),
    ("Aligned", "Not aligned", "did not replicate"),
    ("Not aligned", "Aligned", "did not replicate"),
    ("Inconclusive", "Aligned", "replication not testable"),
    ("Aligned", "Inconclusive", "replication not testable"),
    ("Inconclusive", "Inconclusive", "replication not testable"),
])
def test_replication_wording_is_the_blocks(a4_dec, a9_dec, expected):
    entry = {"decision": a9_dec, "gates": {"all": True}, "p_signal": 0.001}
    out = ana9.replication_reading("gemma4", entry, a4_summary_with("gemma4", a4_dec))
    assert out["reading"] == expected
    assert out["a4_basic_headline_beside"] == "Inconclusive"  # reported beside, never the comparator
    if expected == "replication not testable":
        assert out["reason"]
    if a9_dec == "Inconclusive":
        assert "failed to replicate" not in out["reading"] and out["reading"] != "did not replicate"


def test_replication_not_testable_names_the_gate_or_missing_a4():
    entry = {"decision": "Inconclusive", "gates": {"v3_config_carries": False, "rq0": True}, "p_signal": 0.001}
    out = ana9.replication_reading("qwen3vl", entry, a4_summary_with("qwen3vl", "Aligned"))
    assert "v3_config_carries" in out["reason"]
    out = ana9.replication_reading("qwen3vl", {"decision": "Aligned"}, None)
    assert out["reading"] == "replication not testable" and "not run" in out["reason"]
    entry = {"decision": "Inconclusive", "gates": {"all": True}, "p_signal": 0.4}
    assert "signal" in ana9.why_inconclusive(entry)


def test_earlier_rule_reading_is_descriptive():
    assert ana9.earlier_rule_reading(0.3, 0.0) == "outside the recovery-validated conditions"
    assert ana9.earlier_rule_reading(0.0, 0.0) == "within the recovery-validated conditions"
    assert ana9.earlier_rule_reading(None, 0.0) == "face-shared correlation not estimable"
    assert "first operationalisation" in a9.DESIGN["recovery"]["correlation_use"]
    assert "applying the rule stated before" not in a9.DESIGN["recovery"]["correlation_use"]
    assert a9.DESIGN["recovery"]["withdrawn_under_independent_face_noise"] == []


def test_pinned_inputs_refuse_another_file(tmp_path):
    good = ROOT / "research/reproduction/a9_human_target.json"
    pin = a9.DESIGN["input_pins"]["research/reproduction/a9_human_target.json"]
    assert ana9.pinned_input(good, pin, "human") == pin
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        ana9.pinned_input(other, pin, "human")


def test_behaviour_unavailable_is_recorded(monkeypatch):
    def broken():
        raise SystemExit("pin mismatch")
    monkeypatch.setattr(ht, "load_liu", broken)
    table, status = ana9.human_configuration_table()
    assert table is None and status.startswith("unavailable") and "pin mismatch" in status
