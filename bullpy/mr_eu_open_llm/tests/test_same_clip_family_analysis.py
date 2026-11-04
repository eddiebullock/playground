"""Family-level same-clip analysis for RQ1: criteria, cues, permutation tests and pattern score."""

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
name = "same_clip_family_analysis"
spec = importlib.util.spec_from_file_location(name, ROOT / f"research/scripts/{name}.py")
fam = importlib.util.module_from_spec(spec)
sys.modules[name] = fam
spec.loader.exec_module(fam)
scc = fam.scc

ACTORS = list("BCDEFGHIJKLM")
OPTIONS = {
    "Happy": ["Happy", "Sad", "Bored", "None of the above"],
    "Sad": ["Sad", "Happy", "Hurt", "None of the above"],
    "Bored": ["Bored", "Hurt", "Happy", "None of the above"],
}


def trial_id(category, actor, take):
    return f"emotions/EDITED/{category}/{actor}F{take:02d}_cut.mp4"


def build(rng, beta=0.0, n_cues=2, cue_noise=None):
    """A small within design. beta is the effect of cue 0 on option Happy."""
    rows, cue_rows = [], {}
    for category, options in OPTIONS.items():
        for actor in ACTORS:
            for take in (1, 2, 3):
                tid = trial_id(category, actor, take)
                x = rng.normal(0, 1, n_cues)
                cue_rows[tid] = x
                eta = np.array([0.5 if o == category else -0.5 for o in options])
                eta[options.index("Happy")] += beta * x[0]
                eta = eta + rng.normal(0, 0.2, len(options))
                p = np.exp(eta) / np.exp(eta).sum()
                for option, value in zip(options, rng.multinomial(200, p) / 200):
                    rows.append({"trial_id": tid, "actor": actor, "category": category,
                                 "option": option, "p": value, "n": 200})
    cues = pd.DataFrame.from_dict(cue_rows, orient="index",
                                  columns=[f"c{i}" for i in range(n_cues)])
    return pd.DataFrame(rows), cues


def test_thresholds_match_the_earlier_recovery():
    crit = fam.FAMILY_RECOVERY_CRITERIA
    assert crit["cue_observer"]["median_pearson_min"] == 0.70
    assert crit["cue_observer"]["median_sign_accuracy_min"] == 0.90
    assert crit["null_observer"]["pointwise_fpr_max"] == 0.05
    assert crit["null_observer"]["two_detector_fpr_max"] == 0.01
    assert crit["null_observer"]["abs_pearson_with_any_fixed_matrix_max"] == 0.10
    old = scc.RECOVERY_CRITERIA
    assert crit["cue_observer"] == old["cue_observer"]
    assert {k: crit["null_observer"][k] for k in old["null_observer"]} == old["null_observer"]


def test_families_are_the_decided_groupings():
    assert fam.FAMILIES == {"brow_raise": ("AU01", "AU02"), "brow_lower": ("AU04", "AU09"),
                            "smile": ("AU06", "AU12"), "mouth_open": ("AU25", "AU26")}
    assert sorted(fam.needed_aus(fam.CUE_SETS["families"])) == sorted(scc.SHARED_AUS)


def test_family_score_is_mean_of_z_scores_at_unit_variance():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(3, 2, (50, 2)), columns=["AU01", "AU02"])
    frame["AU07"] = rng.normal(0, 5, 50)
    out = fam.cue_matrix(frame.to_numpy(), list(frame.columns), ["brow_raise", "AU07"])
    z = (frame - frame.mean()) / frame.std(ddof=0)
    expected = z[["AU01", "AU02"]].mean(1)
    expected = (expected - expected.mean()) / expected.std(ddof=0)
    assert np.allclose(out[:, 0], expected)
    assert np.allclose(out[:, 1], z.AU07)
    assert np.allclose(out.std(0), 1)


def test_fast_fitter_reaches_the_same_optimum_as_the_reference_fit():
    rng = np.random.default_rng(1)
    responses, cues = build(rng, beta=0.7)
    design = scc.build_design(responses, cues)
    for lam in (0.3, 10.0, 300.0):
        assert np.allclose(fam.Fitter(design, lam).matrix(), scc.fit_matrix(design, lam), atol=2e-4)


def test_shuffles_stay_within_category_and_are_reproducible():
    categories = np.array(list("aaabbbbcc"))
    one = fam.permutation_indices(categories, 20, 5)
    two = fam.permutation_indices(categories, 20, 5)
    assert np.array_equal(one, two)
    for row in one:
        assert np.array_equal(categories[row], categories)
        assert sorted(row) == list(range(9))


def test_shuffled_cues_keep_controls_out():
    rng = np.random.default_rng(2)
    responses, cues = build(rng)
    design = scc.build_design(responses, cues)
    indices = fam.permutation_indices(design.categories, 3, 0)
    out = fam.shuffled_cues(design.X, indices[0], fam.control_basis(design))
    frame = pd.DataFrame(out)
    assert frame.groupby(design.actors).mean().abs().to_numpy().max() < 1e-9
    assert frame.groupby(design.categories).mean().abs().to_numpy().max() < 1e-9
    assert np.allclose(out.std(0), 1)


def test_exceedance_p_counts_observed_as_one_of_the_draws():
    assert fam.exceedance_p(np.array(5.0), np.array([1.0, 2.0, 3.0, 6.0])) == pytest.approx(2 / 5)


def test_max_t_is_never_below_pointwise():
    rng = np.random.default_rng(3)
    T = rng.normal(0, 1, (5, 3))
    null = rng.normal(0, 1, (99, 5, 3))
    pointwise, familywise = fam.cell_tests(T, null)
    assert (familywise >= pointwise).all()


def test_two_detector_needs_sign_agreement():
    Ta, Tb = np.array([[5.0, 5.0]]), np.array([[5.0, -5.0]])
    null = np.random.default_rng(4).normal(0, 1, (199, 1, 2))
    p = fam.two_detector_p(Ta, Tb, null, null.copy())
    assert p[0, 0] < 0.01
    assert p[0, 1] == 1.0


def test_two_detector_null_is_at_nominal_size_with_correlated_detectors():
    rng = np.random.default_rng(5)
    shared, a, b = rng.normal(size=(3, 3000, 1, 200))
    Ta = np.sqrt(0.8) * shared + np.sqrt(0.2) * a
    Tb = np.sqrt(0.8) * shared + np.sqrt(0.2) * b
    p = fam.two_detector_p(Ta[:1], Tb[:1], Ta[1:], Tb[1:])
    assert np.mean(p < 0.01) < 0.03


def test_permutation_finds_a_planted_cell_and_not_the_null_cue():
    rng = np.random.default_rng(6)
    responses, cues = build(rng, beta=0.8)
    design = scc.build_design(responses, cues)
    indices = fam.permutation_indices(design.categories, 199, 0)
    res = fam.observer_inference(design, 1.0, indices)
    happy = design.options.index("Happy")
    assert res["p_familywise"][happy, 0] < 0.05
    assert res["p_pointwise"][happy, 1] > 0.05


def test_null_observers_keep_family_wise_size():
    flagged = []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        responses, cues = build(rng)
        design = scc.build_design(responses, cues)
        indices = fam.permutation_indices(design.categories, 99, seed)
        res = fam.observer_inference(design, 10.0, indices)
        flagged.append((res["p_familywise"] < 0.05).any())
    assert np.mean(flagged) <= 0.15


def test_pattern_test_high_for_shared_weights_null_for_unrelated():
    rng = np.random.default_rng(7)
    responses, cues = build(rng, beta=1.0)
    other, _ = build(np.random.default_rng(8), beta=1.0)
    other = other.assign(trial_id=other.trial_id)  # same clips, other response noise
    design = scc.build_design(responses, cues)
    reference = fam.Fitter(design, 1.0).matrix()
    indices = fam.permutation_indices(design.categories, 199, 0)
    fitter = fam.Fitter(design, 1.0)
    same = fam.pattern_test(fitter.matrix(), reference, fam.permutation_matrices(fitter, indices))
    assert same["score"] > 0.99 and same["p"] < 0.01

    null_resp, _ = build(np.random.default_rng(9), beta=0.0)
    unrelated = scc.build_design(null_resp, cues)
    fitter = fam.Fitter(unrelated, 1.0)
    test = fam.pattern_test(fitter.matrix(), reference, fam.permutation_matrices(fitter, indices))
    assert test["p"] > 0.01


def test_criteria_file_records_hash_and_condition_names(tmp_path):
    record = fam.write_criteria(tmp_path / "c.json")
    assert record["criteria_sha256"] == fam.criteria_hash()
    assert "prior_only_within" in record["conditions"]
    assert "human_like_published_noise" in record["conditions"]
    assert "actor_prior_within" in record["conditions"]
    assert len(record["pattern_conditions"]) == 3


def test_human_scale_is_chosen_only_from_gain():
    params = inspect.signature(fam.calibrate_human_scale).parameters
    assert "target_gain" in params
    assert "n_perm" not in params


def test_rotation_results_enter_only_through_model_route():
    source = (ROOT / "research/scripts/same_clip_family_analysis.py").read_text()
    code = source.split('"""', 2)[2]
    assert code.count("load_rotation_run(") == 1
    start, end = code.index("def model_route("), code.index("def model_route_summary(")
    assert "scc.load_rotation_run(" in code[start:end]
    assert "allow_partial=False" in code[start:end]
    assert "allow_partial=True" not in code
    assert code.count("model_route(") == 2  # the definition and the --model-route branch


def test_family_pipeline_runs_on_synthetic_cues():
    rng = np.random.default_rng(10)
    responses, _ = build(rng)
    ids = sorted(responses.trial_id.unique())
    names = fam.needed_aus(fam.CUE_SETS["families"])
    frame = pd.DataFrame(rng.normal(0, 1, (len(ids), 4)), index=ids,
                         columns=list(fam.CUE_SETS["families"]))
    template = scc.build_design(responses, frame)
    n_resp = np.full(len(ids), 100)
    cond = next(c for c in fam.family_conditions() if c["name"] == "sparse_1.0_published_noise")
    cond = {**cond, "_n_resp": n_resp, "_pcc_tables": {"published": fam.AU_PCC},
            "_W_ref": None, "with_gain": False, "lam": 1.0}
    out = fam.run_family_sim((template, None, names, cond, 0, 19, None))
    assert out["pearson_with_oracle"] > 0
    prior = next(c for c in fam.family_conditions() if c["name"] == "prior_only_within")
    prior = {**prior, "_n_resp": n_resp, "_pcc_tables": {"published": fam.AU_PCC},
             "_W_ref": None, "with_gain": False, "lam": 10.0}
    row = fam.run_family_sim((template, None, names, prior, 0, 19, None))
    assert row["pointwise_fpr"] is not None
    summary = fam.summarise_family_condition(prior, [row, row])
    assert summary["pass_pointwise"] in (True, False)
    assert "pass_familywise" in summary


def test_pattern_pipeline_runs_on_synthetic_cues():
    rng = np.random.default_rng(11)
    responses, _ = build(rng)
    ids = sorted(responses.trial_id.unique())
    names = fam.needed_aus(fam.CUE_SETS["families"])
    frame = pd.DataFrame(rng.normal(0, 1, (len(ids), 4)), index=ids,
                         columns=list(fam.CUE_SETS["families"]))
    template = scc.build_design(responses, frame)
    W_ref = rng.normal(0, 0.3, (len(template.options), 4))
    cond = {**fam.pattern_conditions(1.0)[1], "_n_resp": np.full(len(ids), 100),
            "_pcc_tables": {"published": fam.AU_PCC}, "_W_ref": W_ref}
    row = fam.run_pattern_sim((template, None, names, cond, 0, 19))
    summary = fam.summarise_pattern_condition(cond, [row, row])
    assert "power_human_like" in summary
    assert "auc_human_like_vs_prior_only" in summary


# --------------------------------------------------------------------------
# Added detector. Nothing here needs torch, cv2 or any eval file.
# --------------------------------------------------------------------------

extract_spec = importlib.util.spec_from_file_location(
    "extract_au_measures", ROOT / "research/scripts/extract_au_measures.py")
ext = importlib.util.module_from_spec(extract_spec)
sys.modules["extract_au_measures"] = ext
extract_spec.loader.exec_module(ext)


def sampler(frame_count, duration_seconds=None, fps=1.0, max_frames=16):
    """Copy of frame_indices_for_video, so this test runs without cv2."""
    n = min(max_frames, max(1, int(np.floor(duration_seconds * fps))))
    if n <= 1:
        return [0]
    return sorted({int(round(i * (frame_count - 1) / (n - 1))) for i in range(n)})


def graphau_frames(trial, n):
    frame = pd.DataFrame({"trial_id": trial, "frame_index": range(n), "declared_frames": n,
                          "video_fps": 25.0, "ge_face": 1.0, "ge_conf": 0.99})
    for number in ext.GRAPHAU_AUS:
        frame[f"ge_{ext.au_name(number)}"] = np.linspace(0, 1, n)
    return frame


def test_policy_frames_need_no_eval_file_and_use_the_recorded_fps():
    meta = pd.DataFrame({"trial_id": ["a", "b"], "declared_frames": [100, 25],
                         "video_fps": [25.0, 25.0]})
    out = ext.policy_frame_indices(meta, sampler)
    assert out["a"] == [0, 33, 66, 99]
    assert out["b"] == [0]


def test_policy_frames_are_refused_when_they_differ_from_stored_aggregates():
    selected = {"a": [0, 2]}
    lf = pd.DataFrame({"trial_id": "a", "frame_index": range(4), "lf_AU12": [1.0, 2.0, 3.0, 9.0]})
    stored = pd.DataFrame([{"trial_id": "a", "detector": "libreface", "frame_set": "model_1fps",
                            "cue": "AU12", "kind": "intensity", "n_frames": 2, "mean": 2.0}])
    assert ext.check_selection_against_aggregates(selected, lf, stored) == 1
    stored["mean"] = 5.0
    with pytest.raises(ValueError):
        ext.check_selection_against_aggregates(selected, lf, stored)


def test_graphau_measures_have_the_layout_the_analysis_reads():
    trial = "emotions/EDITED/Happy/BF01_cut.mp4"
    table = ext.graphau_long_measures(graphau_frames(trial, 50), {trial: [0, 49]})
    assert set(table.frame_set) == {"all", "model_1fps"}
    assert set(table.detector) == {"graphau"}
    assert len(table) == 2 * len(ext.GRAPHAU_AUS)
    chosen = table[(table.frame_set == "model_1fps") & (table.cue == "AU17")].iloc[0]
    assert chosen["n_frames"] == 2 and chosen["mean"] == pytest.approx(0.5)


def test_graphau_covers_every_singly_measured_au():
    names = {ext.au_name(a) for a in ext.GRAPHAU_AUS}
    assert set(fam.SINGLES_TWO_DETECTORS) <= names


def test_five_point_template_keeps_the_face_inside_the_central_crop():
    target = ext.five_point_target()
    margin = (256 - 224) / 2
    assert target.min() > margin and target.max() < 256 - margin


def test_graphau_steps_read_no_eval_file():
    source = (ROOT / "research/scripts/extract_au_measures.py").read_text()
    start = source.index("def aggregate_graphau(args):")
    end = source.index("def long_measures(")
    assert "model_frame_indices" not in source[start:end]
    assert "args.evals" not in source[start:end]


# --------------------------------------------------------------------------
# Second attempt and the frozen route
# --------------------------------------------------------------------------


def test_attempt_1_draws_are_unchanged_by_the_scheme_argument():
    categories = np.array(list("aaabbbbcc"))
    actors = np.array(list("xyxyxyxyx"))
    plan = fam.null_plan(categories, actors, 20, 5)
    assert np.array_equal(plan.indices, fam.permutation_indices(categories, 20, 5))
    assert plan.signs is None and plan.centre is False


def test_within_actor_keeps_every_row_with_its_actor():
    categories = np.array(list("aaabbbbcc"))
    actors = np.array(list("xyxyxyxyx"))
    plan = fam.null_plan(categories, actors, 50, 1, scheme="within_actor")
    moved_category = False
    for row in plan.indices:
        assert np.array_equal(actors[row], actors)
        moved_category = moved_category or not np.array_equal(categories[row], categories)
    assert moved_category


def test_within_actor_holds_for_specs_without_actor_control():
    categories, actors = np.array(list("aabb")), np.array(list("xyxy"))
    plan = fam.null_plan(categories, actors, 10, 1, spec="pooled", scheme="within_actor")
    for row in plan.indices:
        assert np.array_equal(actors[row], actors)


def test_actor_sign_flips_each_actor_as_one_block():
    actors = np.array(list("xxyy"))
    plan = fam.null_plan(np.zeros(4), actors, 30, 1, scheme="actor_sign")
    assert np.array_equal(plan.signs[:, 0], plan.signs[:, 1])
    assert (plan.indices == np.arange(4)).all()


def test_attempt_2_keeps_the_thresholds_and_is_labelled_post_hoc():
    assert fam.ATTEMPT_2["scheme"] == "within_actor"
    assert "second attempt" in fam.ATTEMPT_2["label"]
    assert "Not a replacement" in fam.ATTEMPT_2["label"]
    assert fam.criteria_hash().startswith("4b0a8c3c")


def frozen_setup(beta=0.8):
    rng = np.random.default_rng(12)
    human, cues = build(rng, beta=beta)
    second = cues + rng.normal(0, 0.3, cues.shape)
    return human, {"composite": (cues + second) / 2, "first": cues, "second": second}


def test_frozen_route_takes_each_lambda_from_the_human_and_never_from_the_observer(monkeypatch):
    human, cues = frozen_setup()
    other, _ = build(np.random.default_rng(13), beta=0.0)
    seen = []
    real = scc.choose_lambda

    def spy(design, *a, **k):
        seen.append(float(design.Y[0, ~np.isnan(design.Y[0])][0]))
        return real(design, *a, **k)

    monkeypatch.setattr(scc, "choose_lambda", spy)
    out = fam.frozen_route(human, other, cues, n_perm=19, seed=1, scheme="within_actor")
    human_design = scc.build_design(human, cues["first"])
    assert set(seen) == {float(human_design.Y[0, ~np.isnan(human_design.Y[0])][0])}
    assert len(seen) == 3
    assert set(out) == {"composite", "first", "second"}


def test_frozen_route_gives_a_perfect_score_to_the_human_itself():
    human, cues = frozen_setup()
    out = fam.frozen_route(human, human, cues, n_perm=49, seed=1, scheme="within_actor")
    for part in out.values():
        assert part["pattern"]["score"] == pytest.approx(1.0)
        assert part["pattern"]["p"] < 0.05
        assert np.allclose(part["B"], part["reference"])


def test_frozen_route_does_not_flag_a_null_observer_as_human_like():
    human, cues = frozen_setup(beta=1.0)
    shared, _ = build(np.random.default_rng(14), beta=0.0)
    out = fam.frozen_route(human, shared, cues, n_perm=99, seed=2, scheme="within_actor")
    assert out["composite"]["pattern"]["p"] > 0.01


def test_frozen_route_refuses_different_clips():
    human, cues = frozen_setup()
    keep = sorted(human.trial_id.unique())[:-3]
    with pytest.raises(ValueError):
        fam.frozen_route(human, human[human.trial_id.isin(keep)], cues, n_perm=9, seed=1,
                         scheme="within_actor")


def test_recovery_honours_a_frozen_lambda_per_input(monkeypatch):
    rng = np.random.default_rng(15)
    responses, _ = build(rng)
    ids = sorted(responses.trial_id.unique())
    names = fam.needed_aus(fam.CUE_SETS["families"])
    frame = pd.DataFrame(rng.normal(0, 1, (len(ids), 4)), index=ids,
                         columns=list(fam.CUE_SETS["families"]))
    template = scc.build_design(responses, frame)
    cond = next(c for c in fam.family_conditions((3.0, 10.0, 30.0))
                if c["name"] == "prior_only_frozen_lambda")
    assert cond["lam"] == (3.0, 10.0, 30.0)
    cond = {**cond, "_n_resp": np.full(len(ids), 100), "with_gain": False, "_W_ref": None,
            "_pcc_tables": {"published": fam.AU_PCC}, "_scheme": "within_actor",
            "_seed_offset": 100000}
    used = []
    real = fam.observer_inference
    monkeypatch.setattr(fam, "observer_inference",
                        lambda d, lam, plan, **k: used.append(lam) or real(d, lam, plan, **k))

    def no_choice(*a, **k):
        raise AssertionError("Ridge strength must not be chosen on a frozen route")

    monkeypatch.setattr(scc, "choose_lambda", no_choice)
    fam.run_family_sim((template, None, names, cond, 0, 9, None))
    assert used == [3.0, 10.0, 30.0]


# --------------------------------------------------------------------------
# Route pattern recovery and the loader. Every eval file here is written by the test.
# --------------------------------------------------------------------------


def lookup_file(tmp_path, rng):
    trials = {}
    for category, options in OPTIONS.items():
        for actor in ACTORS:
            for take in (1, 2, 3):
                probs = rng.dirichlet(np.ones(len(options)) * 8)
                trials[trial_id(category, actor, take)] = {
                    "human_options": options,
                    "human_response_distribution": dict(zip(options, probs.tolist())),
                    "human_target_label": category,
                    "n_respondents": 80,
                }
    path = tmp_path / "human.json"
    path.write_text(json.dumps({"trials": trials}), encoding="utf-8")
    return path, trials


def rotation_files(tmp_path, trials, rng, rotations=scc.ROTATIONS, drop=(), drop_in=None):
    """Eval files. Clips in `drop` fail in every rotation, or in `drop_in` rotations if given."""
    paths = []
    for r in rotations:
        items = []
        for tid, rec in trials.items():
            options = scc.rotate_options(rec["human_options"], r)
            probs = rng.dirichlet(np.ones(len(options)) * 8)
            failed = tid in drop and (drop_in is None or r in drop_in)
            items.append({"trial_id": tid, "error": "failed" if failed else None,
                          "vision_source_trial_id": None,
                          "stage2": {"options": options, "fc_n_samples": 20,
                                     "response_distribution": dict(zip(options, probs.tolist()))}})
        path = tmp_path / f"eval_v2_x_seed42_rot{r}_real.json"
        path.write_text(json.dumps({"trials": items}), encoding="utf-8")
        paths.append(path)
    return paths


def cue_csv(tmp_path, trials, rng):
    rows = []
    for tid in trials:
        for detector in ("libreface", "openface3"):
            for cue in scc.SHARED_AUS:
                rows.append({"trial_id": tid, "detector": detector, "kind": "intensity", "cue": cue,
                             "frame_set": "model_1fps", "mean": rng.normal()})
    path = tmp_path / "au.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_model_route_reads_a_clean_run_on_model_frames(tmp_path):
    rng = np.random.default_rng(20)
    human, trials = lookup_file(tmp_path, rng)
    paths = rotation_files(tmp_path, trials, rng)
    au = cue_csv(tmp_path, trials, rng)
    result, attrs = fam.model_route(paths, human, au, n_perm=19)
    assert set(result) == {"composite", "first", "second"}
    assert result["composite"]["B"].shape[1] == 4
    assert attrs["rotations"] == list(scc.ROTATIONS)
    summary = fam.model_route_summary(result, attrs)
    assert summary["cells_are_exploratory"] is True
    json.dumps(summary)


def test_model_route_refuses_a_partial_run(tmp_path):
    rng = np.random.default_rng(21)
    human, trials = lookup_file(tmp_path, rng)
    paths = rotation_files(tmp_path, trials, rng, rotations=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="Missing rotations"):
        fam.model_route(paths, human, cue_csv(tmp_path, trials, rng), n_perm=9)


def test_failed_clips_leave_both_matrices_and_are_listed(tmp_path):
    rng = np.random.default_rng(22)
    human, trials = lookup_file(tmp_path, rng)
    lost = sorted(trials)[:2]
    paths = rotation_files(tmp_path, trials, rng, drop=lost)
    au = cue_csv(tmp_path, trials, rng)
    result, attrs = fam.model_route(paths, human, au, n_perm=19)
    coverage = attrs["coverage"]
    assert coverage["n_dropped"] == 2 and coverage["n_kept"] == len(trials) - 2
    assert [d["trial_id"] for d in coverage["dropped"]] == lost
    for entry in coverage["dropped"]:
        assert entry["label"] == trials[entry["trial_id"]]["human_target_label"]
        assert entry["actor"] == scc.actor_of(entry["trial_id"])
    for part in result.values():
        assert part["B"].shape == part["reference"].shape
        assert len(part["res"]["null"]) == 19
    json.dumps(fam.model_route_summary(result, attrs))


def test_dropping_a_clip_does_not_rechoose_lambda(tmp_path, monkeypatch):
    rng = np.random.default_rng(26)
    human, trials = lookup_file(tmp_path, rng)
    paths = rotation_files(tmp_path, trials, rng, drop=sorted(trials)[:3])
    au = cue_csv(tmp_path, trials, rng)
    calls = []
    real = scc.choose_lambda

    def spy(design, *a, **k):
        calls.append(len(design.trial_ids))
        return real(design, *a, **k)

    monkeypatch.setattr(scc, "choose_lambda", spy)
    result, _ = fam.model_route(paths, human, au, n_perm=9)
    assert calls == [len(trials)] * 3
    assert len(result) == 3


def test_null_draws_are_made_on_kept_clips_only(tmp_path, monkeypatch):
    rng = np.random.default_rng(27)
    human, trials = lookup_file(tmp_path, rng)
    paths = rotation_files(tmp_path, trials, rng, drop=sorted(trials)[:4])
    au = cue_csv(tmp_path, trials, rng)
    sizes = []
    real = fam.null_plan

    def spy(categories, actors, *a, **k):
        sizes.append(len(categories))
        return real(categories, actors, *a, **k)

    monkeypatch.setattr(fam, "null_plan", spy)
    fam.model_route(paths, human, au, n_perm=9)
    assert sizes == [len(trials) - 4] * 3


@pytest.mark.parametrize("n_clips, n_lost, over", [
    (100, 4, False), (100, 5, False), (100, 6, True),  # exactly 5 percent is not over it
    (243, 12, False), (243, 13, True),  # the real clip count: 4.9 and 5.3 percent
    (20, 1, False), (20, 2, True), (243, 0, False),
])
def test_loss_limit_boundary(n_clips, n_lost, over):
    ids = [trial_id("Happy", "B", i) for i in range(n_clips)]
    categories = {t: "Happy" for t in ids}
    coverage = fam.clip_coverage(ids, ids[n_lost:], categories)
    assert coverage["n_dropped"] == n_lost
    assert coverage["over_loss_limit"] is over
    assert fam.rq1_status(coverage).startswith("descriptive") is over


def test_loss_limit_end_to_end(tmp_path):
    rng = np.random.default_rng(28)
    human, trials = lookup_file(tmp_path, rng)  # 108 clips, so 5 percent is 5.4
    au = cue_csv(tmp_path, trials, rng)
    ids = sorted(trials)
    for n_lost, status in ((5, "eligible"), (6, "descriptive")):
        lost = ids[::9][:n_lost]
        paths = rotation_files(tmp_path, trials, rng, drop=lost)
        result, attrs = fam.model_route(paths, human, au, n_perm=9)
        assert fam.model_route_summary(result, attrs)["rq1_status"].startswith(status)


def test_a_clip_failing_in_some_rotations_is_kept_and_listed(tmp_path):
    rng = np.random.default_rng(29)
    human, trials = lookup_file(tmp_path, rng)
    clip = sorted(trials)[0]
    paths = rotation_files(tmp_path, trials, rng, drop=[clip], drop_in=(0, 3))
    _, attrs = fam.model_route(paths, human, cue_csv(tmp_path, trials, rng), n_perm=9)
    coverage = attrs["coverage"]
    assert coverage["n_dropped"] == 0
    assert coverage["partial_rotation_clips"][0]["trial_id"] == clip
    assert coverage["partial_rotation_clips"][0]["n_rotations"] == 3


def test_model_route_stops_if_no_clip_is_usable(tmp_path):
    rng = np.random.default_rng(30)
    human, trials = lookup_file(tmp_path, rng)
    paths = rotation_files(tmp_path, trials, rng, drop=sorted(trials))
    with pytest.raises(ValueError, match="No clip"):
        fam.model_route(paths, human, cue_csv(tmp_path, trials, rng), n_perm=9)


def test_strength_conditions_default_is_unchanged_and_075_is_available():
    assert [c["factor"] for c in fam.pattern_conditions(2.0)] == [0.5, 1.0, 2.0]
    only = fam.pattern_conditions(2.0, (0.75,))
    assert len(only) == 1 and only[0]["scale"] == 1.5


def test_frozen_route_uses_given_lambdas_without_choosing(monkeypatch):
    human, cues = frozen_setup()
    other, _ = build(np.random.default_rng(23), beta=0.0)

    def no_choice(*a, **k):
        raise AssertionError("Ridge strength must not be chosen here")

    monkeypatch.setattr(scc, "choose_lambda", no_choice)
    lams = {"composite": 10.0, "first": 3.0, "second": 30.0}
    out = fam.frozen_route(human, other, cues, n_perm=9, seed=1, scheme="within_category", lams=lams)
    assert {k: v["lam"] for k, v in out.items()} == lams


def test_route_pattern_sim_scores_every_input_and_observer(monkeypatch):
    rng = np.random.default_rng(24)
    responses, _ = build(rng)
    ids = sorted(responses.trial_id.unique())
    names = fam.needed_aus(fam.CUE_SETS["families"])
    frame = pd.DataFrame(rng.normal(0, 1, (len(ids), 4)), index=ids,
                         columns=list(fam.CUE_SETS["families"]))
    template = scc.build_design(responses, frame)
    T0 = rng.normal(0, 1, (len(ids), len(names)))
    W_ref = rng.normal(0, 0.8, (len(template.options), 4))
    cond = {**fam.pattern_conditions(1.0)[2], "_n_resp": np.full(len(ids), 200),
            "_pcc_tables": {"published": fam.AU_PCC}, "_W_ref": W_ref, "_scheme": "within_category",
            "_seed_offset": 200000, "_lams": {"composite": 3.0, "first": 3.0, "second": 3.0}}

    def no_choice(*a, **k):
        raise AssertionError("Ridge strength must not be chosen in the route test")

    monkeypatch.setattr(scc, "choose_lambda", no_choice)
    row = fam.run_route_pattern_sim((template, T0, names, cond, 0, 19))
    assert row["human_like_score"] > row["prior_only_score"]
    summary = fam.summarise_pattern_condition(cond, [row, row])
    assert "size_different_cues" in summary
    first = fam.summarise_pattern_condition(cond, fam.strip_input([row, row], "first"))
    assert "power_human_like" in first


def test_long_table_round_trips_through_the_design():
    rng = np.random.default_rng(25)
    responses, cues = build(rng)
    design = scc.build_design(responses, cues)
    rebuilt = scc.build_design(fam.long_table(design, design.Y), cues)
    assert rebuilt.trial_ids == design.trial_ids
    assert np.array_equal(np.nan_to_num(rebuilt.Y, nan=-1), np.nan_to_num(design.Y, nan=-1))
