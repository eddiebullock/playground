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
from scripts import a9_design as a9  # noqa: E402
from scripts import a9_recovery as rec9  # noqa: E402

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


def test_design_hash_tracks_constants(monkeypatch):
    first = a9.design_hash()
    assert first == a9.design_hash()
    monkeypatch.setattr(a9, "SEED", a9.SEED + 1)
    assert a9.design_hash() != first


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


@pytest.mark.parametrize("design", a9.DESIGNS)
def test_recovery_replicate_runs(design):
    X, Y, labels = synthetic(n=300, subjects=4)
    store = small_store(X, Y, labels)
    gender = np.where(np.arange(300) % 2, 0.5, -0.5)
    W, base = rec9.rec.human_weights(store, a9.DATASET)
    human, lam = an.human_matrix(store, a9.DATASET)
    task = ("human_like", a9.DATASET, 1.0, 0, X, gender, W, base, human, lam, 3, 19, 9, 2, design)
    out = rec9.one_replicate(task)
    assert out["design"] == design
    assert out["residual"]["decision"] in an.OUTCOMES
    assert -1 <= out["residual"]["truth"] <= 1


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
