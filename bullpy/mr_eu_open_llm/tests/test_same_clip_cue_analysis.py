"""Same-clip cue analysis for RQ1: estimation, guards and synthetic recovery."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "research/scripts/same_clip_cue_analysis.py"
spec = importlib.util.spec_from_file_location("same_clip_cue_analysis", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["same_clip_cue_analysis"] = mod
spec.loader.exec_module(mod)

ACTORS = list("BCDEFGHIJKLM")
CATEGORIES = {
    "Happy": ["Happy", "Sad", "Bored", "None of the above"],
    "Sad": ["Sad", "Happy", "Hurt", "None of the above"],
    "Bored": ["Bored", "Hurt", "Happy", "None of the above"],
}


def trial_id(category, actor, take):
    return f"emotions/EDITED/{category}/{actor}F{take:02d}_cut.mp4"


def lookup(tmp_path):
    """Target-first human lookup with an abstain option, as in the real file."""
    trials = {}
    for category, options in CATEGORIES.items():
        for actor in ACTORS:
            for take in (1, 2, 3):
                probs = np.array([0.55, 0.2, 0.2, 0.05])
                trials[trial_id(category, actor, take)] = {
                    "human_options": options,
                    "human_response_distribution": dict(zip(options, probs.tolist())),
                    "human_target_label": category,
                    "n_respondents": 80,
                }
    out = tmp_path / "human.json"
    out.write_text(json.dumps({"trials": trials}), encoding="utf-8")
    return out


def table(rng, n_cues=3, beta=1.0):
    """Long responses with a known effect of cue 0 on option Happy within category."""
    rows = []
    cue_rows = {}
    for category, options in CATEGORIES.items():
        for actor in ACTORS:
            for take in (1, 2, 3):
                tid = trial_id(category, actor, take)
                x = rng.normal(0, 1, n_cues)
                cue_rows[tid] = x
                eta = np.array([0.5 if o == category else -0.5 for o in options])
                eta[options.index("Happy")] += beta * x[0]
                p = np.exp(eta) / np.exp(eta).sum()
                for option, value in zip(options, rng.multinomial(400, p) / 400):
                    rows.append({"trial_id": tid, "actor": actor, "category": category,
                                 "option": option, "p": value, "n": 400})
    cues = pd.DataFrame.from_dict(cue_rows, orient="index",
                                  columns=[f"AU{i:02d}" for i in range(1, n_cues + 1)])
    return pd.DataFrame(rows), cues


def write_rotation(tmp_path, human_path, *, rotations=mod.ROTATIONS, rotate=True, suffix="real",
                   tweak=None):
    trials = json.loads(human_path.read_text())["trials"]
    paths = []
    for r in rotations:
        items = []
        for tid, rec in trials.items():
            options = mod.rotate_options(rec["human_options"], r) if rotate else rec["human_options"]
            dist = {o: 1.0 / len(options) for o in options}
            items.append({"trial_id": tid, "error": None, "vision_source_trial_id": None,
                          "stage2": {"options": options, "response_distribution": dist,
                                     "fc_n_samples": 20}})
        if tweak:
            tweak(items)
        out = tmp_path / f"eval_v2_x_seed42_rot{r}_{suffix}.json"
        out.write_text(json.dumps({"trials": items}), encoding="utf-8")
        paths.append(out)
    return paths


# Estimation ---------------------------------------------------------------


def test_within_design_cues_have_no_category_or_actor_mean():
    responses, cues = table(np.random.default_rng(0))
    design = mod.build_design(responses, cues + 5.0, spec="within")
    frame = pd.DataFrame(design.X)
    assert np.abs(frame.groupby(design.categories).mean().to_numpy()).max() < 1e-8
    assert np.abs(frame.groupby(design.actors).mean().to_numpy()).max() < 1e-8


def test_recovers_known_effect_and_nulls():
    responses, cues = table(np.random.default_rng(1), beta=1.0)
    design = mod.build_design(responses, cues)
    B = mod.fit_matrix(design, 0.3)
    happy, sad = design.options.index("Happy"), design.options.index("Sad")
    assert B[happy, 0] == pytest.approx(0.75, abs=0.2)  # softmax competition shrinks it
    assert B[sad, 0] < -0.2
    assert np.abs(B[:, 1:]).max() < 0.1


def test_option_only_uses_clips_that_offered_it():
    responses, cues = table(np.random.default_rng(2))
    design = mod.build_design(responses, cues)
    rows, block = design.base[design.options.index("Hurt")]
    assert set(design.categories[rows]) == {"Sad", "Bored"}
    assert block.shape == (len(rows), 2)


def test_label_alias_merges_excitement_with_excited(tmp_path):
    source = lookup(tmp_path)
    obj = json.loads(source.read_text())
    tid = trial_id("Happy", "B", 1)
    obj["trials"][tid]["human_options"][2] = "Excitement"
    obj["trials"][tid]["human_response_distribution"]["Excitement"] = 0.1
    source.write_text(json.dumps(obj))
    assert "Excited" in set(mod.load_human(source).option)
    assert "Excitement" not in set(mod.load_human(source).option)


def test_actor_is_the_resampling_unit():
    actors = np.repeat(list("abcd"), 3)
    draws = mod.actor_draws(actors, 50, seed=1)
    assert draws.shape == (50, 12)
    assert (draws.reshape(50, 4, 3).std(axis=2) == 0).all()
    assert draws.sum(axis=1).tolist() == [12.0] * 50
    assert (draws == mod.actor_draws(actors, 50, seed=1)).all()


def test_heldout_loss_ignores_heldout_responses():
    responses, cues = table(np.random.default_rng(3))
    design = mod.build_design(responses, cues)
    heldout = design.actors == "B"
    train = (~heldout).astype(float)
    happy = design.options.index("Happy")
    changed = design.Y.copy()
    changed[heldout & ~np.isnan(changed[:, happy]), happy] = 0.5
    other = design.with_responses(changed)
    a, _ = mod.fit_option(design, happy, 1.0, train)
    b, _ = mod.fit_option(other, happy, 1.0, train)
    assert np.allclose(a, b)


def test_loao_gain_positive_with_cues_and_near_zero_without():
    responses, cues = table(np.random.default_rng(4), beta=1.0)
    design = mod.build_design(responses, cues)
    assert mod.loao_gain(design, lam=1.0)["heldout_gain"] > 0.02
    responses, cues = table(np.random.default_rng(5), beta=0.0)
    design = mod.build_design(responses, cues)
    assert mod.loao_gain(design, lam=100.0)["heldout_gain"] < 0.005


def test_human_and_identical_model_give_identical_matrices():
    responses, cues = table(np.random.default_rng(6))
    a = mod.build_design(responses, cues)
    b = mod.build_design(responses.copy(), cues)
    res_a = mod.estimate(a, n_boot=30, with_gain=False, lam=1.0)
    res_b = mod.estimate(b, n_boot=30, with_gain=False, lam=1.0)
    assert (res_a["B"] == res_b["B"]).all() and (res_a["boots"] == res_b["boots"]).all()
    aligned = mod.compare(a, b, res_a, res_b, n_splits=10)
    assert aligned["pearson"] == 1.0
    assert aligned["pearson_ci_paired_actor_bootstrap"][0] == 1.0


def test_compare_refuses_different_ridge_strength():
    responses, cues = table(np.random.default_rng(7))
    a = mod.build_design(responses, cues)
    with pytest.raises(ValueError, match="ridge"):
        mod.compare(a, a, mod.estimate(a, lam=1.0, n_boot=5, with_gain=False),
                    mod.estimate(a, lam=3.0, n_boot=5, with_gain=False), n_splits=2)


# Detectors ----------------------------------------------------------------


def test_load_cues_filters_and_builds_composite(tmp_path):
    rows = []
    for tid, a, b in (("x_BF1", 1.0, 10.0), ("x_CF1", 2.0, 30.0), ("x_DF1", 3.0, 20.0)):
        for frame_set, shift in (("all", 0.0), ("model_1fps", 100.0)):
            rows += [
                {"trial_id": tid, "detector": "libreface", "kind": "intensity", "cue": "AU12",
                 "frame_set": frame_set, "mean": a + shift, "p90": 0.0},
                {"trial_id": tid, "detector": "libreface", "kind": "occurrence", "cue": "AU12",
                 "frame_set": frame_set, "mean": -9.0, "p90": 0.0},
                {"trial_id": tid, "detector": "libreface", "kind": "occurrence", "cue": "AU07",
                 "frame_set": frame_set, "mean": 7.0, "p90": 0.0},
                {"trial_id": tid, "detector": "openface3", "kind": "intensity", "cue": "AU12",
                 "frame_set": frame_set, "mean": b + shift, "p90": 0.0},
            ]
    out = tmp_path / "au.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    lf = mod.load_cues(out, detector="libreface")
    assert list(lf.columns) == ["AU12"] and lf.AU12.tolist() == [1.0, 2.0, 3.0]
    mixed = mod.load_cues(out, detector="libreface", cues=["AU12", "AU07"])
    assert mixed.AU12.tolist() == [1.0, 2.0, 3.0] and mixed.AU07.tolist() == [7.0] * 3
    assert mod.load_cues(out, detector="libreface", frame_set="model_1fps").AU12.iloc[0] == 101.0
    both = mod.load_cues(out, detector="composite")
    assert both.AU12.mean() == pytest.approx(0.0)
    assert both.AU12.loc["x_BF1"] < both.AU12.loc["x_DF1"]
    agreement = mod.detector_agreement(out)
    assert agreement["AU12"]["pearson"] == pytest.approx(0.5)


def test_rule8_verdicts():
    def frame(beta, lo, hi, excl):
        return pd.DataFrame([{
            "observer": "human", "frame_set": "all", "spec": "within", "option": o, "cue": "AU12",
            "beta": b, "ci_lo": l, "ci_hi": h, "ci_excludes_zero": e, "loao_sign_stability": 1.0,
        } for o, b, l, h, e in zip("abcd", beta, lo, hi, excl)])
    a = frame([1, 1, 1, 1], [0.5, 0.5, 0.5, -1], [1.5, 1.5, 1.5, 2], [True, True, True, False])
    b = frame([1, -1, 0.2, 0.1], [0.2, -2, -1, -1], [2, -0.5, 1, 1], [True, True, False, False])
    out = mod.combine_detectors(a, b).set_index("option")
    assert out.verdict.to_dict() == {"a": "robust", "b": "opposed", "c": "one_detector", "d": "neither"}
    assert out.loc["a", "range_lo"] == 0.2 and out.loc["a", "range_hi"] == 2


# Rotation guards ----------------------------------------------------------


def test_rotation_run_averages_and_uses_every_rotation(tmp_path):
    source = lookup(tmp_path)
    paths = write_rotation(tmp_path, source)
    responses = mod.load_rotation_run(paths, source)
    assert responses.attrs["rotations"] == [0, 1, 2, 3, 4]
    assert responses.p.round(6).eq(0.25).all()
    assert responses.n.eq(100).all()
    assert list(responses.columns) == list(mod.load_human(source).columns)


def test_rotation_run_drops_unparsed_answers_and_renormalises(tmp_path):
    source = lookup(tmp_path)

    def tweak(items):
        for item in items:
            options = item["stage2"]["options"]
            item["stage2"]["response_distribution"] = {
                options[0]: 0.5, options[1]: 0.25, "__unparsed__": 0.25,
            }

    responses = mod.load_rotation_run(write_rotation(tmp_path, source, tweak=tweak), source)
    assert responses.groupby("trial_id").p.sum().round(6).eq(1.0).all()
    assert responses.n.eq(75).all()
    assert responses.attrs["unparsed_share"] == 0.25


def test_refuses_old_target_first_eval_by_name(tmp_path):
    source = lookup(tmp_path)
    old = tmp_path / "eval_v2_eu_emotions_gemma4_video_only_seed42.json"
    old.write_text("{}")
    with pytest.raises(ValueError, match="clean rotation"):
        mod.load_rotation_run([old], source)


def test_refuses_target_first_options_under_a_rotation_name(tmp_path):
    source = lookup(tmp_path)
    paths = write_rotation(tmp_path, source, rotate=False)
    with pytest.raises(ValueError, match="option order"):
        mod.load_rotation_run(paths, source)


def test_refuses_mismatch_smoke_and_partial(tmp_path):
    source = lookup(tmp_path)
    with pytest.raises(ValueError, match="clean rotation"):
        mod.load_rotation_run(write_rotation(tmp_path, source, suffix="mismatch"), source)
    with pytest.raises(ValueError, match="clean rotation"):
        mod.load_rotation_run([tmp_path / "eval_rot1_mismatch_smoke.json"], source)
    with pytest.raises(ValueError, match="Missing rotations"):
        mod.load_rotation_run(write_rotation(tmp_path, source, rotations=(0, 1)), source)


def test_refuses_trial_showing_another_clip(tmp_path):
    source = lookup(tmp_path)

    def tweak(items):
        items[0]["vision_source_trial_id"] = items[1]["trial_id"]

    paths = write_rotation(tmp_path, source, tweak=tweak)
    with pytest.raises(ValueError, match="another clip"):
        mod.load_rotation_run(paths, source)


# Recovery -----------------------------------------------------------------


def template():
    responses, cues = table(np.random.default_rng(8), n_cues=4, beta=0.0)
    cues.columns = ["AU01", "AU04", "AU12", "AU25"]
    return responses, cues, mod.build_design(responses, cues)


def test_noise_matches_published_reliability():
    rng = np.random.default_rng(9)
    n = 40000
    T = rng.normal(0, 1, (n, 2))
    cats = np.repeat(["a", "b"], n // 2)
    actors = np.tile(["p", "q"], n // 2)
    measured = mod.add_detector_noise(T, (cats, actors), np.array([0.34, 0.94]), rng)
    assert np.corrcoef(T[:, 0], measured[:, 0])[0, 1] == pytest.approx(0.34, abs=0.02)
    assert np.corrcoef(T[:, 1], measured[:, 1])[0, 1] == pytest.approx(0.94, abs=0.01)


def test_shared_detector_error_correlates_two_detectors():
    rng = np.random.default_rng(10)
    T = np.zeros((20000, 1)) + rng.normal(0, 1, (20000, 1))
    cats, actors = np.repeat(["a"], 20000), np.tile(["p", "q"], 10000)
    common = rng.normal(0, 1, T.shape)
    a = mod.add_detector_noise(T, (cats, actors), np.array([0.5]), rng, 0.5, common) - T
    b = mod.add_detector_noise(T, (cats, actors), np.array([0.5]), rng, 0.5, common) - T
    assert np.corrcoef(a[:, 0], b[:, 0])[0, 1] == pytest.approx(0.5, abs=0.03)


def test_prior_only_observer_has_zero_weights_and_tracks_category():
    _, _, design = template()
    rng = np.random.default_rng(11)
    W = mod.planted_weights("prior_only", design.options, 4, 1.0, rng)
    assert (W == 0).all()
    T = mod.synthetic_cues(design.categories, design.actors, design.cues, rng).to_numpy()
    n = np.full(len(design.categories), 10**6)
    Y, clean = mod.simulate_observer(design, T, W, kind="prior_only", rng=rng, n_resp=n,
                                     item_sd=0.0, actor_sd=0.0)
    expected = pd.DataFrame(design.Y).groupby(design.categories).transform("mean").to_numpy()
    offered = ~np.isnan(expected)
    assert np.nanmax(np.abs(clean[offered] - expected[offered])) < 0.02


def recovery(name, spec="within", observer="prior_only", scale=0.0, noise=True, n_sims=12):
    responses, cues, _ = template()
    cond = {"name": name, "observer": observer, "scale": scale, "noise": noise, "spec": spec,
            "shared_error": 0.0, "reliability_scale": 1.0, "n_resp": 0, "with_gain": False,
            "lam": None, "seed": 7}
    report = mod.run_recovery(responses, cues_table=None, cue_names=["AU04", "AU12", "AU25"],
                              n_sims=n_sims, n_boot=60, workers=1, conditions=[cond])
    return report["conditions"][0]


def test_recovery_finds_strong_planted_weights():
    out = recovery("strong", observer="sparse", scale=1.0)
    assert out["pearson_with_oracle"] > 0.8
    assert out["sign_accuracy_strong"] > 0.9
    assert out["attenuation_slope"] < 1.0  # detector noise and ridge both attenuate
    assert out["recovered"]


def test_prior_only_stays_null_within_but_not_pooled():
    within = recovery("within", spec="within")
    pooled = recovery("pooled", spec="pooled")
    assert within["median_sd_beta"] < 0.05
    assert pooled["median_sd_beta"] > 5 * within["median_sd_beta"]
    assert pooled["pointwise_fpr"] > 2 * within["pointwise_fpr"]
    assert not pooled["recovered"]


def test_noise_lowers_recovery_of_weak_weights():
    perfect = recovery("perfect", observer="sparse", scale=0.5, noise=False)
    noisy = recovery("noisy", observer="sparse", scale=0.5, noise=True)
    assert noisy["pearson_with_oracle"] < perfect["pearson_with_oracle"]


# Extraction aggregates ----------------------------------------------------

_spec = importlib.util.spec_from_file_location(
    "extract_au_measures", ROOT / "research/scripts/extract_au_measures.py"
)
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)


def write_eval(path, frames):
    path.write_text(json.dumps({"trials": [
        {"trial_id": t, "error": None, "frame_indices": f} for t, f in frames.items()
    ]}))


def test_model_frames_skip_mismatch_runs_and_must_agree(tmp_path):
    write_eval(tmp_path / "eval_a_seed42.json", {"t": [0, 2]})
    write_eval(tmp_path / "eval_a_seed42_rot0_real.json", {"t": [0, 2]})
    write_eval(tmp_path / "eval_a_seed42_rot1_mismatch.json", {"t": [9]})
    write_eval(tmp_path / "eval_a_seed42_rot1_real_smoke.json", {"t": [8]})
    frames, sources = extract.model_frame_indices(str(tmp_path / "eval_*.json"))
    assert frames == {"t": [0, 2]} and len(sources) == 2
    write_eval(tmp_path / "eval_b_seed42.json", {"t": [1, 2]})
    with pytest.raises(ValueError, match="disagree"):
        extract.model_frame_indices(str(tmp_path / "eval_*.json"))


def test_long_measures_splits_frame_sets_and_handles_missing_faces():
    columns = extract.frame_columns()
    frames = pd.DataFrame(np.nan, index=range(4), columns=columns)
    frames.insert(0, "frame_index", range(4))
    frames.insert(0, "trial_id", "x_BF1")
    frames["lf_face"] = [1.0, 1.0, 1.0, 0.0]
    frames["of_face"] = 1.0
    frames["lf_AU12"] = [1.0, 2.0, 3.0, np.nan]
    frames["of_AU12"] = [0.0, 0.0, 4.0, 4.0]
    out = extract.long_measures(frames, {"x_BF1": [0, 2]})
    assert out.trial_id.nunique() == 1 and out.actor.iloc[0] == "B"
    pick = out[(out.cue == "AU12") & (out.detector == "libreface") & (out.kind == "intensity")]
    every = pick[pick.frame_set == "all"].iloc[0]
    seen = pick[pick.frame_set == "model_1fps"].iloc[0]
    assert every["mean"] == 2.0 and every.face_rate == 0.75
    assert seen["mean"] == 2.0 and seen.n_frames == 2 and seen.face_rate == 1.0
    assert set(out.kind) == {"intensity", "occurrence", "pose", "gaze"}
