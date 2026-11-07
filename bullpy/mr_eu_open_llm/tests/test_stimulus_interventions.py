"""Stimulus interventions on synthetic frames, CPU only. No clip, frame or model is used."""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import multi_frame  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402
from scripts.frame_sampling import frame_indices_for_video  # noqa: E402

H, W = 360, 640
SIZE = (H, W)


def fake_landmarks(cx=320.0, cy=180.0, unit=1.0):
    """Points placed so that named rings sit where a face has them."""
    pts = np.zeros((si.N_LANDMARKS, 2), dtype=np.float64)
    angles = np.linspace(0, 2 * np.pi, si.N_LANDMARKS, endpoint=False)
    pts[:, 0] = cx + 60 * unit * np.cos(angles)
    pts[:, 1] = cy + 80 * unit * np.sin(angles)

    def ring(indices, x, y, rx, ry):
        theta = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
        for i, t in zip(indices, theta):
            pts[i] = [cx + unit * (x + rx * np.cos(t)), cy + unit * (y + ry * np.sin(t))]

    ring(si.MOUTH_RING, 0, 45, 25, 10)
    ring(si.RIGHT_EYE_RING, -30, -15, 14, 6)
    ring(si.LEFT_EYE_RING, 30, -15, 14, 6)
    ring(si.RIGHT_BROW, -30, -38, 18, 3)
    ring(si.LEFT_BROW, 30, -38, 18, 3)
    for eye, x in (("right", -30.0), ("left", 30.0)):
        outer, inner, upper, lower = si.EYE_POINTS[eye]
        sign = -1 if eye == "right" else 1
        pts[outer] = [cx + unit * (x + sign * 14), cy - 15 * unit]
        pts[inner] = [cx + unit * (x - sign * 14), cy - 15 * unit]
        pts[upper] = [cx + unit * x, cy + unit * (-15 - 6)]
        pts[lower] = [cx + unit * x, cy + unit * (-15 + 6)]
    pts[234] = [cx - 60 * unit, cy]
    pts[454] = [cx + 60 * unit, cy]
    return pts


def textured_frame(seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(H // 8, W // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


@pytest.fixture
def geometry():
    lm = fake_landmarks()
    return si.ClipGeometry(
        face_box=si.square_box([lm], SIZE),
        eye_box=si.eye_band_box([lm], SIZE),
        face_width_px=si.face_width(lm),
        first_frame=textured_frame(1),
        first_landmarks=fake_landmarks(cx=322, cy=181),
        neutral_frame=textured_frame(2),
        neutral_landmarks=fake_landmarks(cx=300, cy=170, unit=1.1),
    )


# Temporal conditions -------------------------------------------------------


def test_uniform_indices_match_frame_sampling():
    assert si.uniform_indices(0, 144, 8) == frame_indices_for_video(144, 8.0, fps=1.0, max_frames=8)


def test_derangement_is_deterministic_and_moves_every_frame():
    first = si.derangement(8, "clip")
    assert first == si.derangement(8, "clip")
    assert sorted(first) == list(range(8))
    assert all(i != p for i, p in enumerate(first))
    assert first != si.derangement(8, "other")


def test_temporal_plan_keeps_frame_count_and_defines_each_condition():
    plan = si.temporal_plan(150, "trial", apex=75)
    assert set(plan) == set(si.TEMPORAL_CONDITIONS)
    assert all(len(v) == si.N_SENT_FRAMES for v in plan.values())
    base = plan["original"]
    assert base == sorted(base)
    assert plan["reverse"] == base[::-1]
    assert sorted(plan["shuffle_a"]) == base and plan["shuffle_a"] != base
    assert plan["shuffle_a"] != plan["shuffle_b"]
    assert max(plan["trunc_first_third"]) < 50
    assert min(plan["trunc_last_third"]) >= 100
    assert set(plan["static_apex"]) == {75}


def test_no_apex_means_no_static_condition():
    assert "static_apex" not in si.temporal_plan(150, "trial", apex=None)


def test_read_frames_returns_requested_indices(tmp_path):
    path = tmp_path / "ramp.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    if not writer.isOpened():
        pytest.skip("No mp4 encoder")
    for value in range(0, 250, 5):
        writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()
    frames, meta = si.read_frames(path, [40, 3, 20])
    assert sorted(frames) == [3, 20, 40]
    assert abs(float(frames[20].mean()) - 100) < 6
    assert meta["width"] == 64
    with pytest.raises(si.EditFailure) as err:
        si.read_frames(path, [999])
    assert err.value.reason == "frames_unavailable"


# Geometry ------------------------------------------------------------------


def test_similarity_transform_recovers_a_known_map():
    src = fake_landmarks()[list(si.STABLE_POINTS)]
    angle = 0.2
    rotation = 1.3 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = src @ rotation.T + np.array([12.0, -7.0])
    matrix = si.similarity_transform(src, dst)
    assert np.allclose(si.apply_transform(matrix, src), dst, atol=1e-8)


def test_region_masks_sit_on_their_region_and_follow_face_size():
    lm = fake_landmarks()
    mouth = si.region_mask(SIZE, lm, "mouth")
    assert mouth[225, 320] > 0.99
    assert mouth[165, 290] < 0.01
    eyes = si.region_mask(SIZE, lm, "eyes")
    assert eyes[165, 290] > 0.99 and eyes[165, 350] > 0.99
    brows = si.region_mask(SIZE, lm, "brows")
    assert brows[142, 290] > 0.99 and brows[225, 320] < 0.01
    big = si.region_mask(SIZE, fake_landmarks(unit=1.5), "mouth")
    assert big.sum() > mouth.sum() * 1.8


def test_mask_uses_landmarks_of_that_frame():
    near = si.region_mask(SIZE, fake_landmarks(cx=200), "mouth")
    assert near[225, 200] > 0.99 and near[225, 320] < 0.01


def test_square_box_is_fixed_for_a_clip_and_stays_inside():
    box = si.square_box([fake_landmarks(cx=c) for c in (300, 320, 340)], SIZE)
    assert box[2] - box[0] == box[3] - box[1]
    edge = si.square_box([fake_landmarks(cx=20)], SIZE)
    assert edge[0] == 0 and edge[2] <= W


def test_apex_is_the_frame_most_changed_after_registration():
    lm = {0: fake_landmarks(), 10: fake_landmarks(cx=350), 20: fake_landmarks()}
    moved = fake_landmarks()
    moved[list(si.MOUTH_RING)] += [0, 25]
    lm[30] = moved
    assert si.apex_index(lm, [0, 10, 20, 30]) == 30


def test_apex_needs_two_frames():
    assert si.apex_index({0: fake_landmarks(), 5: None}, [0, 5]) is None


# Masks and fills -----------------------------------------------------------


def test_sham_self_fill_is_exactly_the_unedited_crop(geometry):
    frame, lm = textured_frame(), fake_landmarks()
    base, _ = si.apply_condition("original", frame, lm, geometry)
    sham, _ = si.apply_condition("sham_self_fill", frame, lm, geometry)
    assert np.array_equal(base, sham)


@pytest.mark.parametrize("region", list(si.REGIONS))
def test_fill_types_differ_and_leave_the_rest_alone(geometry, region):
    frame, lm = textured_frame(), fake_landmarks()
    outside = si.region_mask(SIZE, lm, region) == 0
    made = {}
    for fill in si.FILL_TYPES:
        image, log = si.apply_condition(f"mask_{region}_{fill}", frame, lm, geometry, view="full")
        assert log["fill"] == fill and log["region"] == region
        assert np.array_equal(image[outside], frame[outside])
        assert not np.array_equal(image, frame)
        made[fill] = image
    assert not np.array_equal(made["neutral"], made["first"])
    assert not np.array_equal(made["neutral"], made["blur"])


def test_every_edit_is_deterministic(geometry):
    frame, lm = textured_frame(), fake_landmarks()
    for name in si.SPATIAL_CONDITIONS:
        a, log_a = si.apply_condition(name, frame, lm, geometry)
        b, log_b = si.apply_condition(name, frame, lm, geometry)
        assert np.array_equal(a, b), name
        assert log_a == log_b


def test_alignment_residual_is_logged_and_flagged(geometry):
    bad = fake_landmarks()
    bad[list(si.STABLE_POINTS)[:4]] += 25
    geometry.neutral_landmarks = bad
    _, log = si.apply_condition("mask_mouth_neutral", textured_frame(), fake_landmarks(), geometry)
    assert log["align_residual"] > si.MAX_ALIGN_RESIDUAL
    assert log["align_flagged"] is True


def test_missing_donor_or_landmarks_fail_with_a_reason(geometry):
    frame, lm = textured_frame(), fake_landmarks()
    geometry.neutral_frame = None
    with pytest.raises(si.EditFailure) as err:
        si.apply_condition("mask_eyes_neutral", frame, lm, geometry)
    assert err.value.reason == "neutral_clip_no_landmarks"
    with pytest.raises(si.EditFailure) as err:
        si.apply_condition("mask_eyes_blur", frame, None, geometry)
    assert err.value.reason == "no_face_detected"
    image, _ = si.apply_condition("original", frame, None, geometry, view="full")
    assert np.array_equal(image, frame)


def test_apply_condition_does_not_touch_its_input(geometry):
    frame, lm = textured_frame(), fake_landmarks()
    copy = frame.copy()
    for name in si.SPATIAL_CONDITIONS:
        si.apply_condition(name, frame, lm, geometry)
    assert np.array_equal(frame, copy)


def test_eye_crop_and_eye_window(geometry):
    frame, lm = textured_frame(), fake_landmarks()
    band, log = si.apply_condition("eye_crop", frame, lm, geometry)
    x0, y0, x1, y1 = geometry.eye_box
    assert band.shape[1] == si.FACE_SIZE and log["view"] == "eye_band"
    assert band.shape[0] / band.shape[1] == pytest.approx((y1 - y0) / (x1 - x0), abs=0.01)
    assert y1 < 215  # stops above the mouth ring
    window, _ = si.apply_condition("eye_window", frame, lm, geometry, view="full")
    assert np.array_equal(window[y0:y1, x0:x1], frame[y0:y1, x0:x1])
    assert (window[225, 320] == 128).all()


def test_face_crops_are_sent_at_a_fixed_size(geometry):
    frame, lm = textured_frame(), fake_landmarks()
    for name in si.SPATIAL_CONDITIONS:
        if name != "eye_crop":
            image, _ = si.apply_condition(name, frame, lm, geometry)
            assert image.shape == (si.FACE_SIZE, si.FACE_SIZE, 3), name


def test_eyes_and_mouth_mask_covers_both(geometry):
    lm = fake_landmarks()
    both = si.region_mask(SIZE, lm, "eyesmouth")
    assert both[225, 320] > 0.99 and both[165, 290] > 0.99 and both[165, 350] > 0.99
    assert both[142, 320] < 0.01
    image, log = si.apply_condition("mask_eyesmouth_neutral", textured_frame(), lm, geometry, view="full")
    assert log["region"] == "eyesmouth" and not np.array_equal(image, textured_frame())


# Spatial frequency ---------------------------------------------------------


def _grating(cycles_per_image, size=256):
    x = np.arange(size)
    wave = 127.5 + 100 * np.sin(2 * np.pi * cycles_per_image * x / size)
    return np.repeat(np.repeat(wave[None, :, None], size, 0), 3, 2).round().astype(np.uint8)


def test_filters_pass_the_band_they_name():
    fine, coarse = _grating(64), _grating(2)
    amplitude = lambda image: float(image[128:].std())  # noqa: E731
    assert amplitude(si.spatial_frequency_filter(fine, "low", 8, 256)) < 2
    assert amplitude(si.spatial_frequency_filter(coarse, "low", 8, 256)) > 60
    assert amplitude(si.spatial_frequency_filter(coarse, "high", 32, 256)) < 2
    assert amplitude(si.spatial_frequency_filter(fine, "high", 32, 256)) > 50


def test_cutoff_is_half_amplitude_at_the_named_frequency():
    image = _grating(32)
    kept = si.spatial_frequency_filter(image, "low", 32, 256)
    core = lambda a: float(a[64:192, 64:192].std())  # noqa: E731
    assert core(kept) / core(image) == pytest.approx(0.5, abs=0.05)


def test_high_pass_keeps_mean_luminance():
    frame = textured_frame()
    assert si.spatial_frequency_filter(frame, "high", 32, 120).mean() == pytest.approx(frame.mean(), abs=2)


def test_cutoffs_scale_with_face_width():
    frame = textured_frame()
    a = si.spatial_frequency_filter(frame, "low", 8, 100)
    b = si.spatial_frequency_filter(frame, "low", 8, 300)
    assert not np.array_equal(a, b)


# Eye aperture --------------------------------------------------------------


def _lid_frame():
    frame = np.full((H, W, 3), 200, dtype=np.uint8)
    for x in (290, 350):
        frame[157:161, x - 14 : x + 14] = 0
        frame[169:173, x - 14 : x + 14] = 0
    return frame


def _lid_gap(frame, x=290):
    darkness = 200.0 - frame[140:195, x, 0].astype(np.float64)
    rows = np.arange(140, 195)
    upper = rows < 165
    centre = lambda keep: (rows[keep] * darkness[keep]).sum() / darkness[keep].sum()  # noqa: E731
    return centre(~upper) - centre(upper)


def test_scale_one_is_an_exact_copy():
    frame = textured_frame()
    assert np.array_equal(si.scale_eye_aperture(frame, fake_landmarks(), 1.0), frame)


def test_aperture_moves_lids_by_the_dose_in_both_directions():
    frame, lm = _lid_frame(), fake_landmarks()
    base = _lid_gap(frame)
    wide = _lid_gap(si.scale_eye_aperture(frame, lm, 1.25))
    narrow = _lid_gap(si.scale_eye_aperture(frame, lm, 0.75))
    assert wide == pytest.approx(base * 1.25, rel=0.08)
    assert narrow == pytest.approx(base * 0.75, rel=0.08)


def test_aperture_warp_stays_local():
    frame, lm = textured_frame(), fake_landmarks()
    out = si.scale_eye_aperture(frame, lm, 1.25)
    assert np.array_equal(out[200:], frame[200:])
    assert np.array_equal(out[:, :240], frame[:, :240])
    assert np.array_equal(out[:, 400:], frame[:, 400:])
    assert not np.array_equal(out, frame)


def test_closed_eyes_fail_with_a_reason():
    lm = fake_landmarks()
    for eye in si.EYE_POINTS:
        _, _, upper, lower = si.EYE_POINTS[eye]
        lm[upper] = lm[lower]
    with pytest.raises(si.EditFailure) as err:
        si.scale_eye_aperture(textured_frame(), lm, 1.25)
    assert err.value.reason == "eye_closed"


# Inference -----------------------------------------------------------------


def test_signflip_is_exact_and_has_the_right_floor():
    effects = np.ones((10, 1)) + 0.01 * np.arange(10)[:, None]
    two = si.signflip_maxt(effects)
    assert two["exact"] and two["n_flips"] == 2**10
    assert two["p_raw"][0] == pytest.approx(2 / 2**10)
    one = si.signflip_maxt(effects, alternative=[1])
    assert one["p_raw"][0] == pytest.approx(1 / 2**10)


def test_maxt_finds_a_planted_effect_and_is_at_least_raw():
    rng = np.random.default_rng(0)
    effects = rng.normal(0, 1, size=(12, 6))
    effects[:, 0] += 3
    out = si.signflip_maxt(effects)
    assert out["p_maxt"][0] < 0.01
    assert all(adj >= raw - 1e-12 for adj, raw in zip(out["p_maxt"], out["p_raw"]))
    assert min(out["p_maxt"][1:]) > 0.05


def test_maxt_is_monotone_in_the_observed_order():
    rng = np.random.default_rng(1)
    effects = rng.normal(0.4, 1, size=(9, 5))
    out = si.signflip_maxt(effects)
    order = np.argsort(-np.asarray(out["t"]))
    adjusted = np.asarray(out["p_maxt"])[order]
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_maxt_controls_familywise_error_under_the_null():
    rejected = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        effects = rng.normal(size=(9, 6))
        rejected += min(si.signflip_maxt(effects)["p_maxt"]) < 0.05
    assert rejected <= 6


def test_absent_actors_are_skipped():
    effects = np.tile(np.linspace(0.8, 1.2, 8)[:, None], (1, 2))
    effects[4:, 1] = np.nan
    out = si.signflip_maxt(effects, alternative=[1, 1])
    assert out["p_raw"][1] > out["p_raw"][0]


def test_sign_of_a_directional_test_counts():
    effects = -np.linspace(0.8, 1.2, 8)[:, None]
    assert si.signflip_maxt(effects, alternative=[1])["p_raw"][0] > 0.9
    assert si.signflip_maxt(effects, alternative=[-1])["p_raw"][0] < 0.01


def test_js_and_clr():
    assert si.js_divergence([1, 0], [1, 0]) == pytest.approx(0)
    assert si.js_divergence([1, 0], [0, 1]) == pytest.approx(np.log(2))
    assert si.clr([1, 2, 3]).sum() == pytest.approx(0)


# Runner logic --------------------------------------------------------------


def test_lexicon_words_belong_to_one_family_and_cover_every_option():
    counts = Counter(word for _family, word in run.lexicon_words())
    assert max(counts.values()) == 1
    assert all(" " not in word for word in counts)
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    options = {o for v in human.values() for o in v["human_options"]} - {run.ABSTAIN}
    assert options <= set(run.OPTION_TO_FAMILY)
    assert len(run.LEXICON) == 21


def test_blind_prompt_names_no_label():
    words = re.findall(r"[a-z]+", (run.BLIND_PROMPT + " " + run.BLIND_PREFILL).lower())
    forms = {w for _family, w in run.lexicon_words()}
    assert not forms & set(words)


def test_prefix_scoring_needs_one_forward_per_shared_prefix():
    sequences = [[1], [2], [3, 4], [3, 5], [6, 7, 8]]
    assert run.internal_prefixes(sequences) == [(), (3,), (6,), (6, 7)]
    calls = {
        (): {1: -1.0, 2: -2.0, 3: -3.0, 6: -4.0},
        (3,): {4: -0.5, 5: -0.25},
        (6,): {7: -0.1},
        (6, 7): {8: -0.2},
    }
    assert run.needed_tokens(sequences)[(3,)] == [4, 5]
    assert run.joint_logprob([3, 5], calls) == pytest.approx(-3.25)
    assert run.joint_logprob([6, 7, 8], calls) == pytest.approx(-4.3)


def test_family_score_sums_probability_over_forms():
    words = [("A", "a1"), ("A", "a2"), ("B", "b1")]
    scores = run.family_scores(words, np.log([0.1, 0.2, 0.3]).tolist())
    assert np.exp(scores["A"]) == pytest.approx(0.3)
    assert np.exp(scores["B"]) == pytest.approx(0.3)


def test_human_comparison_drops_abstain_and_merges_excitement():
    family = {"Excited": 0.0, "Sad": np.log(3)}
    human = {"Excitement": 0.2, "Sad": 0.6, run.ABSTAIN: 0.2}
    model, votes = run.human_comparable(family, human)
    assert model == pytest.approx([0.25, 0.75])
    assert votes == pytest.approx([0.25, 0.75])


def test_rotations_never_put_the_target_first():
    for position in range(40):
        pair = run.mcq_rotations(position)
        assert 0 not in pair and pair[0] != pair[1]
    assert Counter(run.mcq_rotations(p)[0] for p in range(40)) == {1: 10, 2: 10, 3: 10, 4: 10}


def test_item_order_is_fixed_and_a_prefix_spreads_over_actors():
    items = [{"trial_id": f"{a}{i}", "actor": a} for a in "ABCD" for i in range(3)]
    first = [i["trial_id"] for i in run.select_items([dict(i) for i in items], None)]
    second = [i["trial_id"] for i in run.select_items([dict(i) for i in reversed(items)], None)]
    assert first == second
    assert len({i["actor"] for i in run.select_items([dict(i) for i in items], 4)}) == 4


def test_weak_label_items_are_the_26_named():
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    assert sum(v["human_target_label"] in run.WEAK_LABELS for v in human.values()) == 26


def test_design_hash_changes_with_the_prompt(monkeypatch):
    before = run.design_hash()
    monkeypatch.setattr(run, "BLIND_PROMPT", "Other text.")
    assert run.design_hash() != before


def test_readout_fields_still_give_the_smoke_v2_hash():
    """The smoke's prompt, lexicon, conditions and k are unchanged; only the hash grew."""
    text = json.dumps(run.readout_fields(), sort_keys=True)
    assert run.text_sha256(text) == "a1924d800453298a3406c7ce5f01918895b60a5ad8f06c0058767ccb47321084"


def test_design_hash_covers_edits_mcq_geometry_and_analysis(monkeypatch):
    from scripts import stimulus_design as design

    before = run.design_hash()
    assert before != run.text_sha256(json.dumps(run.readout_fields(), sort_keys=True))
    for owner, name, value in (
        (si, "FEATHER_FRACTION", 0.5),
        (si, "OUTER_EYE_CORNERS", (0, 1)),
        (si, "FACE_EDGES", (0, 1)),
        (si, "EYE_BAND_MARGIN", 0.2),
        (design, "A3_SUMMARY_SHA256", "0"),
        (si, "FACE_BOX_MARGIN", 2.0),
        (si, "SEED", 7),
        (run, "MCQ_PREFILL", "LABEL:"),
        (run, "GEOMETRY_SHA256", "0"),
        (run, "VIDEO_KWARGS", {}),
        (design, "MARGIN_FRACTION", 0.2),
        (design, "CONFIRMATORY", design.CONFIRMATORY[:-1]),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, value)
            assert run.design_hash() != before, name


def test_geometry_pin_matches_the_a3_pin_and_is_checked(tmp_path):
    from scripts import face_input_validity as fiv

    assert run.GEOMETRY_SHA256 == fiv.GEOMETRY_SHA256
    path = tmp_path / "geometry.json"
    path.write_text("{}")
    with pytest.raises(SystemExit, match="not the pinned"):
        run.check_geometry(path)


def test_landmarks_refuse_to_overwrite_geometry(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_text("{}")
    with pytest.raises(SystemExit, match="not overwritten"):
        run.step_landmarks(type("Args", (), {"geometry": path})())


def test_a_full_run_must_state_its_shards():
    assert run.resolve_shards(0, None, False) == 1
    assert run.resolve_shards(2, 3, True) == 3
    with pytest.raises(SystemExit, match="--n_shards"):
        run.resolve_shards(0, None, True)
    for shard, count in ((3, 3), (-1, 3), (0, 0)):
        with pytest.raises(SystemExit, match="not valid"):
            run.resolve_shards(shard, count, True)


@pytest.mark.parametrize(
    "env,message",
    [
        ({}, "N_SHARDS"),
        ({"SLURM_ARRAY_TASK_ID": "2", "SLURM_ARRAY_TASK_COUNT": "1"}, "N_SHARDS"),
        ({"N_SHARDS": "3", "SLURM_ARRAY_TASK_ID": "3"}, "N_SHARDS"),
    ],
)
def test_slurm_script_refuses_bad_shards_before_any_setup(env, message):
    script = ROOT / "slurm_jobs/study3_stimulus_interventions.sh"
    result = subprocess.run(
        ["bash", str(script)],
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "STEP": "run", "CONFIRM_GATES_PASSED": "1", **env},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and message in result.stderr


def test_order_check_blocks_a_shuffle_that_does_not_arrive():
    assert run.order_check({})
    assert run.order_check({"pixels_change_under_shuffle": True})
    assert not run.order_check({"pixels_change_under_shuffle": False})
    assert not run.order_check({"pixels_arrive_in_sent_order": False})


def test_slurm_examples_use_three_shards():
    text = (ROOT / "slurm_jobs/study3_stimulus_interventions.sh").read_text()
    assert "--array=0-2" in text and "--array=0-3" not in text


def _write_clip(path, n=60):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    if not writer.isOpened():
        pytest.skip("No mp4 encoder")
    for i in range(n):
        writer.write(np.full((H, W, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()


def _clip_setup(tmp_path, detect):
    item = {"trial_id": "x/AF1.mp4", "actor": "A", "label": "Sad", "path": str(tmp_path / "a.mp4")}
    neutral = {"trial_id": "x/AF2.mp4", "actor": "A", "label": "Neutral", "path": str(tmp_path / "n.mp4")}
    _write_clip(tmp_path / "a.mp4")
    _write_clip(tmp_path / "n.mp4", 40)
    records = {}
    for entry in (item, neutral):
        record = run.clip_geometry_record(entry, detect)
        record["path"] = entry["path"]
        records[entry["trial_id"]] = record
    return item, records


def test_geometry_and_media_log_fixed_sent_indices(tmp_path):
    calls = {"n": 0}

    def detect(_frame):
        calls["n"] += 1
        return fake_landmarks(cy=180 + calls["n"] % 3)

    item, records = _clip_setup(tmp_path, detect)
    json.dumps(records)  # numeric only
    clip = run.load_clip(item, records, {"A": "x/AF2.mp4"})
    single = run.build_media(clip, "single_face", "mask_mouth_neutral")
    assert single.sent_indices == [clip.apex]
    assert single.log["neutral_donor"]["trial_id"] == "x/AF2.mp4"
    reverse = run.build_media(clip, "video_face", "reverse")
    original = run.build_media(clip, "video_face", "original")
    assert reverse.sent_indices == original.sent_indices[::-1]
    assert len(reverse.images) == si.N_SENT_FRAMES
    assert reverse.images[0].shape == original.images[0].shape
    legacy = run.build_media(clip, "legacy_full", "shuffle_a")
    assert legacy.images[0].shape[:2] == (H, W)


def test_undetected_frames_are_recorded_and_fail_only_what_needs_them(tmp_path):
    def detect(_frame):
        return None

    item, records = _clip_setup(tmp_path, detect)
    record = records[item["trial_id"]]
    assert record["failed_frames"] == record["base_indices"]
    assert record["box_failure"] == "fewer_than_half_frames_have_a_face"
    assert record["apex"] is None
    clip = run.load_clip(item, records, {"A": "x/AF2.mp4"})
    assert run.build_media(clip, "legacy_full", "original").images
    with pytest.raises(si.EditFailure) as err:
        run.build_media(clip, "single_face", "original")
    assert err.value.reason == "no_apex_frame"
    with pytest.raises(si.EditFailure) as err:
        run.build_media(clip, "video_face", "original")
    assert err.value.reason == "no_face_box"


def test_frames_that_decode_differently_from_geometry_fail_the_clip():
    frame = textured_frame()
    record = {"frame_sha256": {"0": si.array_sha256(frame)}}
    run.check_decoder(record, {0: frame})
    with pytest.raises(si.EditFailure) as err:
        run.check_decoder(record, {0: textured_frame(9)})
    assert err.value.reason == "decoder_mismatch"


def test_full_run_is_refused_without_the_gate_flag(tmp_path):
    args = type("Args", (), {"max_items": None, "confirm_gates_passed": False})()
    with pytest.raises(SystemExit, match="G2 to G4"):
        run.step_run(args)


def test_scripts_contain_no_frame_writing_calls():
    banned = ("imwrite", "VideoWriter", ".save(", "imencode", "tofile", "np.save")
    for name in ("stimulus_interventions.py", "run_stimulus_interventions.py"):
        source = (ROOT / "scripts" / name).read_text()
        assert not [b for b in banned if b in source], name


# What the existing pipeline does, pinned so a change there is noticed ------


def test_composite_grid_keeps_only_the_first_eight_frames():
    images = [Image.new("RGB", (64, 36), (i * 20, 0, 0)) for i in range(12)]
    grid, meta = multi_frame.prepare_images_for_model("gemma4", images)
    assert meta["n_frames_used"] == 12 and meta["grid_cells"] == 8
    assert grid.size == (336 * 4, 336 * 2)
    assert grid.getpixel((336 * 3 + 10, 336 + 10))[0] == 7 * 20


def test_composite_grid_squashes_a_wide_frame_to_a_square_cell():
    images = [Image.new("RGB", (1920, 1080), (i, 0, 0)) for i in range(8)]
    grid, _ = multi_frame.prepare_images_for_model("gemma4", images)
    assert grid.size[0] // 4 == grid.size[1] // 2 == 336
