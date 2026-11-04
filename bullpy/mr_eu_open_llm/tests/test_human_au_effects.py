"""Human AU effect estimation, on synthetic data with known effects."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "research/scripts/human_au_effects.py"
spec = importlib.util.spec_from_file_location("human_au_effects", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

LABELS = ["target", "rival", "other"]


def make_design(rng, n_stim, active_rate=0.15):
    """Randomised amplitudes, one row per stimulus, in the study's amplitude steps."""
    active = rng.random((n_stim, len(mod.AU_COLS))) < active_rate
    amps = rng.choice([1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0], size=active.shape)
    return active * amps


def simulate(
    rng,
    n_subjects=12,
    n_stim=1200,
    target_effects=None,
    subject_sd=0.0,
    stimulus_sd=0.0,
    stim_set="single",
):
    """Softmax choice among three options; AU effects act on the target logit.

    The same stimuli go to all subjects, as in the real data. stimulus_sd adds a
    per-stimulus shock that AUs do not explain.
    """
    target_effects = target_effects or {"AU12L": 2.0, "AU12R": 2.0, "AU04": -2.0}
    amps = make_design(rng, n_stim)
    weights = np.zeros(len(mod.AU_COLS))
    for au, value in target_effects.items():
        weights[mod.AU_COLS.index(au)] = value
    gender = rng.choice(["F", "M"], n_stim)
    shock = rng.normal(0, stimulus_sd, n_stim) if stimulus_sd else np.zeros(n_stim)
    rows = []
    for s in range(n_subjects):
        subject_w = weights + rng.normal(0, subject_sd, len(weights))
        target = -0.5 + amps @ subject_w + shock
        logits = np.column_stack([target, np.zeros(n_stim), np.full(n_stim, -0.3)])
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        draws = (probs.cumsum(axis=1) > rng.random((n_stim, 1))).argmax(axis=1)
        frame = pd.DataFrame(amps, columns=mod.AU_COLS)
        frame["cfg"] = [f"stim{i}" for i in range(n_stim)]
        frame["face_gender"] = gender
        frame["face_id"] = 1
        frame["sub"] = f"{s:02d}XX"
        frame["stim_set"] = stim_set
        frame["label"] = np.array(LABELS)[draws]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_ridge_logit_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 3))
    true = np.array([-0.5, 1.5, -1.0, 0.0])
    X = np.column_stack([np.ones(len(x)), x])
    y = (rng.random(len(x)) < mod.expit(X @ true)).astype(float)
    beta = mod.fit_ridge_logit(X, y)
    assert np.allclose(beta, true, atol=0.08)


def test_ridge_penalty_shrinks_slopes_but_not_the_intercept_term():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 2))
    X = np.column_stack([np.ones(len(x)), x])
    y = (rng.random(len(x)) < mod.expit(1.0 + 2 * x[:, 0])).astype(float)
    weak = mod.fit_ridge_logit(X, y, pen=np.array([0.0, 0.01, 0.01]))
    strong = mod.fit_ridge_logit(X, y, pen=np.array([0.0, 500.0, 500.0]))
    assert abs(strong[1]) < 0.5 * abs(weak[1])


def test_fractional_outcome_with_weights_matches_binary_fit():
    rng = np.random.default_rng(3)
    x = rng.choice([0.0, 1.0], size=4000)
    X = np.column_stack([np.ones(len(x)), x])
    y = (rng.random(len(x)) < mod.expit(-1 + 2 * x)).astype(float)
    binary = mod.fit_ridge_logit(X, y)
    cells = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["mean", "size"]).reset_index()
    Xc = np.column_stack([np.ones(len(cells)), cells["x"]])
    pooled = mod.fit_ridge_logit(Xc, cells["mean"].to_numpy(), w=cells["size"].to_numpy())
    assert np.allclose(binary, pooled, atol=1e-5)


def test_label_never_used_gives_zero_slopes_and_finite_output():
    X = np.column_stack([np.ones(200), np.random.default_rng(4).random(200)])
    beta = mod.fit_ridge_logit(X, np.zeros(200), pen=np.array([0.0, 1.0]))
    assert beta[1] == 0.0 and np.isfinite(beta[0]) and beta[0] < 0


def test_subject_means_recover_signs_and_nulls_of_known_effects():
    rng = np.random.default_rng(5)
    frame = simulate(rng)
    subs, betas, dos, never = mod.fit_subject_effects(frame, LABELS, "label", lam=0.1)
    names, _ = mod.effect_index()
    mean = betas.mean(axis=0)[0]
    assert len(subs) == 12 and not never.any()
    # One-vs-rest coefficients on target are close to softmax effects, attenuated a little.
    assert mean[names.index("AU12_bilateral")] == pytest.approx(4.0, abs=1.0)
    assert mean[names.index("AU04")] == pytest.approx(-2.0, abs=0.6)
    nulls = [names.index(a) for a in ("AU09", "AU15", "AU26", "AU43")]
    assert np.abs(mean[nulls]).max() < 0.45


def test_do_effect_matches_true_interventional_contrast():
    rng = np.random.default_rng(6)
    frame = simulate(rng, n_subjects=30, n_stim=1500)
    amps = frame[frame["sub"] == "00XX"][mod.AU_COLS].to_numpy()
    w = np.zeros(33)
    w[mod.AU_COLS.index("AU04")] = -2.0
    w[mod.AU_COLS.index("AU12L")] = w[mod.AU_COLS.index("AU12R")] = 2.0
    j = mod.AU_COLS.index("AU04")

    def softmax_target(a):
        t = np.exp(-0.5 + a @ w)
        return t / (t + 1 + np.exp(-0.3))

    on, off = amps.copy(), amps.copy()
    on[:, j], off[:, j] = 1.0, 0.0
    truth = float(np.mean(softmax_target(on) - softmax_target(off)))
    _, _, dos, _ = mod.fit_subject_effects(frame, LABELS, "label", lam=0.1)
    names, _ = mod.effect_index()
    assert dos.mean(axis=0)[0, names.index("AU04")] == pytest.approx(truth, abs=0.03)


def test_do_effects_sum_to_zero_across_labels():
    rng = np.random.default_rng(7)
    frame = simulate(rng, n_subjects=3)
    _, _, dos, _ = mod.fit_subject_effects(frame, LABELS, "label", lam=0.1)
    assert np.abs(dos.sum(axis=1)).max() < 1e-10


def test_bootstrap_ci_covers_truth_and_respects_strata():
    rng = np.random.default_rng(8)
    arr = rng.normal(1.0, 0.5, size=(40, 2, 3))
    lo, hi, se = mod.stratified_bootstrap_ci(arr, ["a"] * 10 + ["b"] * 30, 400, seed=1)
    assert np.all(lo < 1.0) and np.all(hi > 1.0)
    assert np.allclose(se, 0.5 / np.sqrt(40), atol=0.04)


def test_split_half_high_for_shared_signal_and_near_zero_for_noise():
    rng = np.random.default_rng(9)
    signal = rng.normal(size=(1, 3, 25))
    noisy = signal + rng.normal(0, 0.5, size=(40, 3, 25))
    overall, per_label = mod.split_half_reliability(noisy, ["s"] * 40, 100, seed=1)
    pure_noise = rng.normal(size=(40, 3, 25))
    null, _ = mod.split_half_reliability(pure_noise, ["s"] * 40, 100, seed=1)
    assert overall["split_half_r"] > 0.9 and len(per_label) == 3
    assert overall["spearman_brown"] > overall["split_half_r"]
    assert abs(null["split_half_r"]) < 0.1


def test_spearman_brown_known_value():
    assert mod.spearman_brown(0.5) == pytest.approx(2 / 3)


def test_shared_stimulus_shocks_inflate_same_stimulus_split_only():
    """With no AU effect, stimulus noise fakes reliability unless stimuli are split too."""
    rng = np.random.default_rng(10)
    frame = simulate(rng, n_subjects=20, n_stim=800, target_effects={"AU09": 0.0}, stimulus_sd=2.0)
    subs, betas, _, _ = mod.fit_subject_effects(frame, LABELS, "label", lam=0.1)
    fam = mod.family_positions(mod.effect_index()[0])
    same, _ = mod.split_half_reliability(betas[:, :, fam], ["s"] * 20, 50, seed=1)
    table = mod.StimulusTable(frame, LABELS, "label")
    cross = mod.cross_stimulus_reliability(table, 20, seed=1)
    assert same["split_half_r"] > 0.6
    assert cross["split_half_r"] < 0.3


def test_stimulus_table_pooled_fit_agrees_with_generating_signs():
    rng = np.random.default_rng(11)
    frame = simulate(rng, n_subjects=10, n_stim=1000)
    table = mod.StimulusTable(frame, LABELS, "label")
    assert table.N.sum() == len(frame)
    assert table.C.sum() == len(frame)
    pooled = table.fit(np.ones(10, dtype=bool), np.arange(table.X.shape[0]))
    names, _ = mod.effect_index()
    assert pooled[0, names.index("AU12_bilateral")] > 2.0
    assert pooled[0, names.index("AU04")] < -1.0


def test_effect_index_has_33_columns_plus_8_composites_and_25_families():
    names, members = mod.effect_index()
    assert len(names) == 41 and len(mod.FAMILIES) == 25
    assert members[names.index("AU12_bilateral")] == [
        mod.AU_COLS.index("AU12L"),
        mod.AU_COLS.index("AU12R"),
    ]
    assert len(mod.family_positions(names)) == 25


def test_lambda_selection_scores_on_unseen_stimuli_and_tracks_signal():
    # Three real effects among 33 inputs: moderate penalty beats both extremes.
    frame = simulate(np.random.default_rng(12), n_subjects=4, n_stim=2400)
    lam, losses = mod.choose_lambda(frame, LABELS, "label", grid=(0.01, 10.0, 300.0), seed=1)
    assert lam == 10.0
    assert losses["10.0"] < losses["intercept_only"]
    # No signal: the heaviest penalty wins.
    null = simulate(np.random.default_rng(13), n_subjects=4, n_stim=2400, target_effects={"AU09": 0.0})
    lam_null, _ = mod.choose_lambda(null, LABELS, "label", grid=(0.01, 10.0, 300.0), seed=1)
    assert lam_null == 300.0


def test_load_zip_rejects_unexpected_member_names(tmp_path):
    import zipfile

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("basic_emotions/WC/evil.tsv", "a\tb\n")
    with pytest.raises(ValueError):
        mod.load_zip(bad)
