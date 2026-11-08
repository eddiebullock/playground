"""Draft amendment A6, CPU only. No clip, frame or model is used; frames here are made up."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from scripts import a6_analyse as an  # noqa: E402
from scripts import a6_design as d  # noqa: E402
from scripts import a6_recovery as rec  # noqa: E402
from scripts import a6_run as ar  # noqa: E402
from scripts import analyse_stimulus_interventions as a2  # noqa: E402
from scripts import face_input_validity as fiv  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402

D = d.DESIGN
FAST = {"flips": 2047, "boots": 400, "seed": 42}
A, B = an.INDEX["Anger"], an.INDEX["Disgust"]
H, W = 360, 480


@pytest.fixture(scope="module")
def items():
    return ar.load_items(d.SELECTION, Path("."))


@pytest.fixture(scope="module")
def partners(items):
    return ar.partners_for(items)


def fake_landmarks(cx=240.0, cy=180.0, unit=1.0):
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
    pts[133], pts[362] = [cx - 16 * unit, cy - 15 * unit], [cx + 16 * unit, cy - 15 * unit]
    pts[129], pts[358] = [cx - 12 * unit, cy + 14 * unit], [cx + 12 * unit, cy + 14 * unit]
    pts[2] = [cx, cy + 20 * unit]
    pts[33], pts[263] = [cx - 44 * unit, cy - 15 * unit], [cx + 44 * unit, cy - 15 * unit]
    pts[234], pts[454] = [cx - 60 * unit, cy], [cx + 60 * unit, cy]
    return pts


def textured_frame(seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(H // 8, W // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def fake_clip():
    lm = fake_landmarks()
    geometry = si.ClipGeometry(
        face_box=si.square_box([lm], (H, W)),
        face_width_px=si.face_width(lm),
        first_frame=textured_frame(1),
        first_landmarks=fake_landmarks(cx=242, cy=181),
        neutral_frame=textured_frame(2),
        neutral_landmarks=fake_landmarks(cx=232, cy=170, unit=1.1),
    )
    item = {"trial_id": "x", "actor": "1", "label": "Anger", "path": ""}
    record = {"base_indices": [0, 3]}
    return run.Clip(item, record, {3: textured_frame(0)}, {3: lm}, geometry, {}, 3)


# The design block ---------------------------------------------------------------


def test_block_is_the_pinned_and_stated_one():
    assert d.block_hash() == d.DESIGN_SHA256 == d.stated_hash()


def test_a_changed_block_is_refused(tmp_path):
    copy = tmp_path / "draft.md"
    copy.write_text(d.DRAFT.read_text(encoding="utf-8").replace('"seed": 42', '"seed": 43'), encoding="utf-8")
    with pytest.raises(SystemExit):
        d.load_design(copy)


def test_a_wrong_stated_hash_is_refused(tmp_path):
    copy = tmp_path / "draft.md"
    text = d.DRAFT.read_text(encoding="utf-8").replace(d.DESIGN_SHA256, "0" * 64)
    copy.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        d.load_design(copy)


def test_frozen_a2_a3_code_and_selection_are_unchanged():
    d.check_pins(D)
    assert set(D["a2_code_sha256"]) == {"scripts/stimulus_interventions.py", "scripts/run_stimulus_interventions.py",
                                        "scripts/analyse_stimulus_interventions.py"}


def test_a_changed_pinned_file_is_refused(tmp_path):
    for name in d.pins(D):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    d.check_pins(D, tmp_path)
    (tmp_path / "scripts/prompts.py").write_text("changed", encoding="utf-8")
    with pytest.raises(SystemExit):
        d.check_pins(D, tmp_path)


def test_pending_items_block_a_real_run():
    assert d.pending_items(D) == ["geometry_sha256", "weights_sha256"]
    with pytest.raises(SystemExit):
        d.check_frozen(D)


def test_orders_are_a_williams_square():
    d.validate(D)
    for a, b in ((x, y) for i, x in enumerate(D["options"]) for y in D["options"][i + 1:]):
        assert len(d.adjacent_rows(D, (a, b))) == 2
    assert d.adjacent_rows(D, D["target_pair"]) == [0, 3]
    assert d.edit_rows_by_rule(D) == D["edit_order_rows"] == [2, 5]


def test_binding_hash_ignores_only_the_cluster_pins(tmp_path):
    copy = tmp_path / "PREREG_A6_DRAFT.md"
    text = d.DRAFT.read_text(encoding="utf-8")
    copy.write_text(text.replace('"geometry_sha256": "PENDING"', '"geometry_sha256": "' + "a" * 64 + '"'), encoding="utf-8")
    assert d.binding_hash(copy) == d.binding_hash() and d.block_hash(copy) != d.block_hash()
    copy.write_text(text.replace('"seed": 42', '"seed": 43'), encoding="utf-8")
    assert d.binding_hash(copy) != d.binding_hash()


def test_a6_code_is_pinned_and_the_pinned_set_covers_the_imports():
    pinned = set(d.pins(D))
    for name in ("scripts/a6_analyse.py", "scripts/a6_run.py", "scripts/a6_recovery.py", "research/scripts/a6_human_target.py",
                 "slurm_jobs/study3_a6.sh", "scripts/prompts.py", "scripts/evaluate.py", "scripts/frame_sampling.py", "config.py"):
        assert name in pinned
    assert "scripts/a6_design.py" not in pinned  # it holds the block's own hash


def test_cyclic_orders_are_refused():
    options = D["options"]
    cyclic = dict(D)
    cyclic["orders"] = [options[r:] + options[:r] for r in range(6)]
    with pytest.raises(SystemExit):
        d.validate(cyclic)


def test_a_wrong_edit_order_is_refused():
    wrong = dict(D)
    wrong["edit_order_rows"] = [1, 4]
    with pytest.raises(SystemExit):
        d.validate(wrong)


# Human target -------------------------------------------------------------------


def test_human_target_file_is_pinned_and_follows_the_rule():
    d.check_human_target(D)
    human = json.loads(d.HUMAN_TARGET.read_text(encoding="utf-8"))
    assert "design_sha256" not in human  # the block pins this file, so it cannot name the block
    assert human["rule"]["target_pair"] == D["target_pair"]
    assert human["rule"]["shared_family"] == D["shared_family"]
    assert human["selection_sha256"] == D["selection_sha256"]


def test_vote_table_matches_the_draft():
    human = json.loads(d.HUMAN_TARGET.read_text(encoding="utf-8"))
    table = human["vote_share"]
    assert round(table["Anger"][1], 3) == 0.206
    assert round(table["Disgust"][5], 3) == 0.217
    assert round(table["Sad"][4], 3) == 0.391
    assert round(table["Happy"][3], 3) == 0.874


def test_every_table_number_in_the_draft_comes_from_the_file():
    human = json.loads(d.HUMAN_TARGET.read_text(encoding="utf-8"))
    text = d.DRAFT.read_text(encoding="utf-8")
    top = human["pairs"][0]
    assert top["pair"] == D["target_pair"]
    assert f"{top['X']:.3f}"[1:] in text
    for key in ("Anger>Disgust", "Disgust>Anger"):
        assert f"{human['directed'][key]['X']:.3f}"[1:] in text


def test_selection_facts(items, partners):
    assert len(items) == 819 and len({i["actor"] for i in items}) == 91
    assert sum(i["label"] in D["stage3_emotions"] for i in items) == 547
    by_id = {i["trial_id"]: i for i in items}
    assert all(by_id[k]["label"] != by_id[v]["label"] and by_id[k]["actor"] != by_id[v]["actor"] for k, v in partners.items())
    assert len(set(partners.values())) == 819
    donors = ar.neutral_donors(items)
    assert len(donors) == 91
    for actor, donor in donors.items():
        own = sorted(i["trial_id"] for i in items if i["actor"] == actor and i["label"] == "No Emotion")
        assert donor == own[0]


# Regions and edits --------------------------------------------------------------


def test_region_mask_equals_the_frozen_one_on_mouth():
    lm = fake_landmarks()
    mine = ar.region_mask((H, W), lm, si.REGIONS["mouth"], si.REGION_DILATE["mouth"])
    assert np.array_equal(mine, si.region_mask((H, W), lm, "mouth"))


def test_regions_resolve_to_a2_rings_and_a_new_mid_hull():
    assert ar.region_rings("upper") == si.REGIONS["eyes"] + si.REGIONS["brows"]
    assert ar.region_rings("lower") == si.REGIONS["mouth"]
    assert ar.region_rings("mid") == (tuple(D["regions"]["mid"]["hull"]),)
    assert ar.region_rings("midlower") == ar.region_rings("mid") + ar.region_rings("lower")
    assert ar.region_rings("uplow") == ar.region_rings("upper") + ar.region_rings("lower")
    assert ar.region_rings("all") == ar.region_rings("upper") + ar.region_rings("mid") + ar.region_rings("lower")
    assert not set(D["regions"]["mid"]["hull"]) & set(si.MOUTH_RING)


def test_narrowest_side_on_a_known_shape():
    hard = np.zeros((100, 200), dtype=np.uint8)
    hard[10:20, 10:110] = 255
    hard[50:90, 150:190] = 255
    assert ar.narrowest_side(hard) == pytest.approx(10.0, abs=1.0)
    assert ar.narrowest_side(np.zeros((10, 10), dtype=np.uint8)) == 0.0


def test_sham_sends_the_original_pixels():
    clip = fake_clip()
    original = ar.edit_media(clip, "original")
    sham = ar.edit_media(clip, "sham_self_fill")
    assert si.array_sha256(original.images[0]) == si.array_sha256(sham.images[0])
    assert sham.log["region"] == D["sham_region"]


def test_every_condition_builds_and_edits_only_its_region():
    clip = fake_clip()
    original = ar.edit_media(clip, "original").images[0]
    assert original.shape == (si.FACE_SIZE, si.FACE_SIZE, 3)
    for condition in D["conditions"]:
        media = ar.edit_media(clip, condition)
        if condition not in ("original", "sham_self_fill"):
            assert not np.array_equal(media.images[0], original)
            assert media.log["span_view_fraction"] > 0
    lm = clip.landmarks[3]
    size = (H, W)
    upper = ar.region_mask(size, lm, ar.region_rings("upper"), D["region_dilate_fraction"])
    lower = ar.region_mask(size, lm, ar.region_rings("lower"), D["region_dilate_fraction"])
    assert (upper * lower).max() < 1e-3


def test_a_missing_donor_fails_the_cell_with_a_reason():
    clip = fake_clip()
    clip.geometry.neutral_frame = None
    with pytest.raises(si.EditFailure):
        ar.edit_media(clip, "mask_mid_neutral")


def test_no_a6_code_writes_an_image():
    for name in ("scripts/a6_run.py", "scripts/a6_analyse.py", "scripts/a6_recovery.py", "research/scripts/a6_human_target.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "imwrite" not in text and ".save(" not in text and "VideoWriter" not in text


def test_no_emojis_in_a6_files():
    pattern = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]")
    for name in ("scripts/a6_design.py", "scripts/a6_run.py", "scripts/a6_analyse.py", "scripts/a6_recovery.py",
                 "research/scripts/a6_human_target.py", "slurm_jobs/study3_a6.sh", "tests/test_a6.py", "research/PREREG_A6_DRAFT.md"):
        assert not pattern.search((ROOT / name).read_text(encoding="utf-8"))


# Plans, resume, refusals --------------------------------------------------------


def test_stage_plans_match_the_cost_table(items):
    first = [c for i in items for c in ar.stage_plan(i, 1)]
    third = [c for i in items for c in ar.stage_plan(i, 3)]
    assert sum(c["route"] == "single_face" and c["readout"] == "mcq" for c in first) == 4914
    assert sum(c["readout"] == "blind" for c in first) == 819
    assert sum(c["route"] == "video_face" for c in first) == 1638
    assert sum(c["readout"] == "mcq" for c in third) == 15316
    assert sum(c["readout"] == "blind" for c in third) == 0  # dropped: no rule read them
    keys = [ar.cell_key(c) for c in first + third]
    assert len(keys) == len(set(keys))


def test_prime_prompt_has_the_word_and_the_order():
    prompt = ar.prime_prompt("No Emotion", D["orders"][0])
    assert prompt.startswith("A person feels neutral. ")
    assert "1) Anger\n2) Disgust\n3) Sad" in prompt


def test_resume_cuts_a_torn_line(tmp_path):
    path = tmp_path / "cells.jsonl"
    good = {"stage": 1, "trial_id": "x", "route": "single_face", "readout": "mcq", "order": 0, "condition": "original", "design_sha256": "h"}
    path.write_bytes((json.dumps(good) + "\n").encode() + b'{"stage": 1, "tri')
    keys = ar.resume_keys(path, "h")
    assert keys == {ar.cell_key(good)}
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(SystemExit):
        ar.resume_keys(path, "other")


class FakeHandle:
    forwards = 0
    mcq_prefill = [1]

    class tokenizer:  # noqa: N801
        def __new__(cls, text, add_special_tokens=False):
            return type("T", (), {"input_ids": [abs(hash(text)) % 1000]})


def test_score_one_keeps_the_order_and_logs_the_edit(monkeypatch):
    monkeypatch.setattr(run, "encode_media", lambda handle, media, prompt: {"prompt": prompt})
    seen = []

    def score(handle, inputs, prefill, sequences):
        seen.append(inputs["prompt"])
        return [-float(i) for i in range(len(sequences))]

    monkeypatch.setattr(run, "score_sequences", score)
    clip = fake_clip()
    item = {"trial_id": "x", "actor": "1", "label": "Fear", "human_modal_label": "Fear"}
    cell = next(c for c in ar.stage_plan(item, 3) if c["condition"] == "mask_mid_neutral" and c["readout"] == "mcq")
    out = ar.score_one(FakeHandle, dict(cell), ar.media_cache(clip, ""))
    assert out["status"] == "ok" and out["options"] == D["orders"][cell["order"]]
    assert out["joint_logprob"] == [-0.0, -1.0, -2.0, -3.0, -4.0, -5.0]
    assert out["edit"]["region"] == "mid" and len(out["media_sha256"]) == 1
    assert "1) " + D["orders"][cell["order"]][0] in seen[0]
    fields = fiv.mechanics_row(out)
    assert "joint_logprob" not in fields and "correct" not in fields and "p_target" not in fields


def test_a_failed_clip_is_a_failed_cell_with_its_reason():
    item = {"trial_id": "x", "actor": "1", "label": "Fear", "human_modal_label": "Fear"}
    cell = ar.stage_plan(item, 1)[0]
    out = ar.score_one(FakeHandle, dict(cell), ar.media_cache(None, "no_geometry_record"))
    assert out["status"] == "failed" and out["reason"] == "no_geometry_record"


def namespace(**kwargs):
    base = {"a6_frozen": True, "confirm_gates_passed": True, "geometry": Path("nowhere"), "stage2_output": None, "model": "gemma4"}
    return type("Args", (), {**base, **kwargs})()


def test_a_run_needs_the_freeze_the_gates_and_no_pending_pin():
    for kwargs in ({"a6_frozen": False}, {"confirm_gates_passed": False}, {}):
        with pytest.raises(SystemExit):
            ar.check_run_allowed(namespace(**kwargs))


def test_stage3_needs_a_locked_stage2_that_reaches_it(tmp_path):
    with pytest.raises(SystemExit):
        ar.stage3_allowed(namespace())
    path = tmp_path / "s2.json"
    path.write_text(json.dumps({"model": "gemma4", "design_sha256": d.block_hash(), "stage2": {"reaches_stage3": False}}))
    with pytest.raises(SystemExit):
        ar.stage3_allowed(namespace(stage2_output=path))
    path.write_text(json.dumps({"model": "qwen3vl", "design_sha256": d.block_hash(), "stage2": {"reaches_stage3": True}}))
    with pytest.raises(SystemExit):
        ar.stage3_allowed(namespace(stage2_output=path))
    path.write_text(json.dumps({"model": "gemma4", "design_sha256": d.block_hash(), "stage2": {"reaches_stage3": True}}))
    ar.stage3_allowed(namespace(stage2_output=path))


def test_mechanics_stays_outside_the_results():
    with pytest.raises(SystemExit):
        ar.mechanics_output_dir(ar.OUTCOME_DIR)
    assert ar.mechanics_output_dir(ar.MECHANICS_DIR) == ar.MECHANICS_DIR


def test_a_full_run_must_name_its_shards():
    with pytest.raises(SystemExit):
        run.resolve_shards(0, None, True)


@pytest.mark.parametrize("env, needle", [
    ({"STEP": "run", "MODEL": "gemma4", "STAGE": "1"}, "N_SHARDS"),
    ({"STEP": "run", "MODEL": "gemma4", "STAGE": "1", "N_SHARDS": "2"}, "wait on the freeze"),
    ({"STEP": "run", "MODEL": "gemma4", "STAGE": "1", "N_SHARDS": "2", "A6_FROZEN": "1"}, "wait on the freeze"),
    ({"STEP": "run", "MODEL": "gemma4", "STAGE": "1", "N_SHARDS": "2", "CONFIRM_GATES_PASSED": "1"}, "wait on the freeze"),
    ({"STEP": "baselines", "MODEL": "gemma4"}, "wait on the freeze"),
    ({"STEP": "analyse", "STAGE": "1"}, "wait on the freeze"),
    ({"STEP": "run", "MODEL": "gemma4", "STAGE": "1", "N_SHARDS": "0", "A6_FROZEN": "1", "CONFIRM_GATES_PASSED": "1"}, "N_SHARDS"),
    ({"STEP": "run", "MODEL": "molmo2", "STAGE": "1", "N_SHARDS": "2", "A6_FROZEN": "1", "CONFIRM_GATES_PASSED": "1"}, "Molmo2"),
    ({"STEP": "nonsense"}, "STEP must be"),
])
def test_the_slurm_script_refuses(env, needle):
    clean = {k: v for k, v in os.environ.items() if k not in ("STEP", "MODEL", "STAGE", "N_SHARDS", "A6_FROZEN", "CONFIRM_GATES_PASSED", "SLURM_ARRAY_TASK_ID")}
    done = subprocess.run(["bash", str(ROOT / "slurm_jobs/study3_a6.sh")], env={**clean, **env}, capture_output=True, text=True, timeout=30)
    assert done.returncode == 1
    assert needle in done.stderr


# Pair excess --------------------------------------------------------------------


@pytest.fixture(scope="module")
def holder(items):
    return an.Cells(items, [])


def spread(holder, accuracy, prior=None):
    """Option probabilities with acted evidence and no pair excess, from additive scores."""
    prior = np.zeros(6) if prior is None else prior
    evidence = np.log(accuracy / (1 - accuracy) * 5)
    scores = prior + evidence * np.eye(6)[holder.emo]
    return an.softmax(scores)


@pytest.mark.parametrize("accuracy", [0.3, 0.6, 0.9])
def test_x_is_zero_with_no_pair_excess_at_any_accuracy(holder, accuracy):
    assert np.nanmean(an.pair_excess(holder, spread(holder, accuracy), A, B)) == pytest.approx(0.0, abs=1e-12)


def test_x_is_zero_with_any_option_prior(holder):
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = spread(holder, 0.6, rng.normal(0, 1.5, 6))
        for a, b in ((A, B), (B, A), (2, 5)):
            assert np.nanmean(an.pair_excess(holder, p, a, b)) == pytest.approx(0.0, abs=1e-12)


def test_the_first_drafted_measure_had_a_floor(holder):
    """Why the measure changed: p(B)/(1-p(A)) on the other clips keeps their acted label."""
    p = spread(holder, 0.6)
    r = p[:, B] / (1 - p[:, A])
    first = holder.actor_mean(r, holder.emo == A) - holder.actor_mean(r, (holder.emo != A) & (holder.emo != B))
    assert np.nanmean(first) == pytest.approx(0.113, abs=0.001)


def test_x_finds_a_real_excess_and_is_directed(holder):
    scores = 2.0 * np.eye(6)[holder.emo]
    scores[holder.emo == A, B] += 1.0
    p = an.softmax(scores)
    forward = np.nanmean(an.directed_excess(holder, p, A, B))
    back = np.nanmean(an.directed_excess(holder, p, B, A))
    assert forward > 0.2 and back < forward
    assert np.nanmean(an.pair_excess(holder, p, A, B)) == pytest.approx((forward + back) / 2)


def test_a_bias_towards_one_option_gives_no_x(holder):
    scores = 2.0 * np.eye(6)[holder.emo]
    scores[:, B] += 2.0
    p = an.softmax(scores)
    assert abs(np.nanmean(an.pair_excess(holder, p, A, B))) < 1e-12


def test_boost_for_hits_its_target():
    prior = np.array([0.5, -0.5, 0.2, 0.0, 1.0, -1.0])
    g = rec.boost_for(prior, A, B, 0.2)
    kept = [k for k in range(6) if k not in (A, 2)]
    raw = np.exp(prior[kept])
    assert g > 0
    assert raw[kept.index(B)] * np.exp(g) / (raw.sum() + raw[kept.index(B)] * (np.exp(g) - 1)) > raw[kept.index(B)] / raw.sum()


def test_correct_matches_a3_including_ties():
    rng = np.random.default_rng(0)
    logs = rng.normal(0, 1, (40, 6, 6))
    logs[0, 0, :] = 0.0
    logs[1, 2, 1] = logs[1, 2].max() + 1
    logs[1, 2, 3] = logs[1, 2, 1]
    target = rng.integers(0, 6, 40)
    target[1] = 1
    mine = an.correct_matrix(logs, target)
    for i in range(40):
        for o in range(6):
            assert bool(mine[i, o]) == fiv.correct_from_scores(an.OPTIONS, logs[i, o].tolist(), an.OPTIONS[target[i]])["correct"]


def test_mismatch_uses_the_partners_cell_on_the_same_order(items, partners):
    """A clip scored on its own face is always right, and on its partner's face always wrong."""
    rows = []
    for item in items:
        for cell in ar.stage_plan(item, 1):
            row = dict(cell, status="ok")
            if row["readout"] == "mcq" and row["route"] == "single_face":
                row["joint_logprob"] = [5.0 if o == item["label"] else 0.0 for o in row["options"]]
                rows.append(row)
    cells = an.Cells(items, rows)
    out = an.stage1(cells, partners, FAST, descriptive=False)
    assert out["accuracy_real"] == 1.0 and out["accuracy_mismatch"] == 0.0 and out["n_items"] == 819


def test_a_missing_order_loses_the_clip_and_its_partner_row(items, partners):
    cells = an.Cells(items, [])
    cells.mcq1[:] = np.random.default_rng(0).normal(0, 1, cells.mcq1.shape)
    cells.mcq1[0, 3] = np.nan
    out = an.stage1(cells, partners, FAST, descriptive=False)
    assert out["n_items"] == 817
    assert not out["over_lost_item_limit"]
    cells.mcq1[:60, 0] = np.nan
    assert an.stage1(cells, partners, FAST, descriptive=False)["over_lost_item_limit"]


# The outcome --------------------------------------------------------------------


def contrast(name):
    return next(c for c in an.CONTRASTS if c["id"] == name)


def staged(holder, shift):
    """Stage 3 cells whose only change under mask_mid_neutral is shift(emo, scores)."""
    cells = object.__new__(an.Cells)
    cells.__dict__.update(holder.__dict__)
    rng = np.random.default_rng(3)
    cells.mcq3 = np.zeros_like(holder.mcq3)
    base = rng.normal(0, 1, (len(holder.emo), 6))
    cells.mcq3[:] = base[:, None, None, :]
    c = an.CONDITIONS.index("mask_mid_neutral")
    for r in range(2):
        cells.mcq3[:, c, r] = shift(base.copy())
    return cells


def test_fixed_reference_keeps_only_the_partner_and_happy(holder):
    for name in ("C1", "C2", "C3", "C4"):
        c = contrast(name)
        keep = an.outcome_keep(holder, c, "fixed", D["control_emotions"])
        assert sorted(np.array(an.OPTIONS)[keep[0]]) == sorted([c["partner"], "Happy"])
        assert (keep == keep[0]).all()


@pytest.mark.parametrize("reference, cancels", [("fixed", True), ("two_out", False)])
def test_a_fall_in_each_clips_own_label_cannot_read_as_a_fall_in_the_partner(holder, reference, cancels):
    def shift(scores):
        scores[np.arange(len(scores)), holder.emo] -= 2.0
        return scores

    cells = staged(holder, shift)
    effect = np.nanmean(an.double_difference(cells, contrast("C1"), "mask_mid_neutral", reference=reference))
    assert bool(abs(effect) < 1e-9) is cancels
    if not cancels:
        assert effect < 0  # the fault: signal loss on control clips looked like the predicted fall


@pytest.mark.parametrize("pull", ["Anger", "Disgust", "Fear", "No Emotion"])
def test_a_generic_pull_cancels_under_the_primary_reference(holder, pull):
    def shift(scores):
        scores[:, an.INDEX[pull]] += 2.0
        return scores

    cells = staged(holder, shift)
    for name in ("C1", "C2"):
        assert np.nanmean(an.double_difference(cells, contrast(name), "mask_mid_neutral")) == pytest.approx(0.0, abs=1e-9)


def test_beyond_zero():
    assert an.beyond_zero([0.01, 0.2], 1) and not an.beyond_zero([-0.01, 0.2], 1)
    assert an.beyond_zero([-0.2, -0.01], -1) and not an.beyond_zero([-0.2, 0.01], -1)
    assert not an.beyond_zero(None, 1)


def test_a_change_on_acted_clips_alone_is_seen_with_its_sign(holder):
    def shift(scores):
        scores[holder.emo == A, B] -= 1.0
        return scores

    cells = staged(holder, shift)
    assert np.nanmean(an.double_difference(cells, contrast("C1"), "mask_mid_neutral")) == pytest.approx(-0.5)
    assert np.nanmean(an.plain_change(cells, contrast("C1"), "mask_mid_neutral")) == pytest.approx(-0.5)


# The decision table -------------------------------------------------------------


def first_ok():
    return {"vision_dependent": True, "over_lost_item_limit": False}


def second_ok(state="stage3", forward="carries", back="carries"):
    return {"state": state, "directions": {"Anger>Disgust": {"state": forward}, "Disgust>Anger": {"state": back}}}


def row(**kwargs):
    base = {"match": False, "opposes": False, "null": False, "significant": False, "pair_inside_margin": False,
            "reference_stable": True, "share_of_excess_removed": None,
            "margin": 0.1, "mask_uplow_ci90": [0.5, 0.6],
            "null_conditions": {"c2_delivery": True, "c3_dose": True, "c4_pair_can_move": True, "c5_equivalence": False, "c6_reported": True}}
    return {**base, **kwargs}


def third(c1=None, c2=None, sham=True):
    return {"sham_equal": sham, "over_lost_item_limit": False, "contrasts": {"C1": c1 or row(), "C2": c2 or row()}}


GATES = dict(rec.GATES_OK)


@pytest.mark.parametrize("gates, one, two, three, expected, reason", [
    ({}, first_ok(), None, None, an.NOT_TESTED, None),
    ({**GATES, "rq0": False}, first_ok(), None, None, an.NOT_TESTED, None),
    ({**GATES, "delivery": "yes"}, first_ok(), None, None, an.NOT_TESTED, None),
    (GATES, {"vision_dependent": True, "over_lost_item_limit": True}, None, None, an.NOT_TESTED, None),
    (GATES, {"vision_dependent": False, "over_lost_item_limit": False}, None, None, an.NO_FACE_USE, None),
    (GATES, first_ok(), second_ok("not_shared"), None, an.NOT_SHARED, None),
    (GATES, first_ok(), second_ok("shared_unresolved"), None, an.INCONCLUSIVE, "shared_unresolved"),
    (GATES, first_ok(), second_ok("option_dependent"), None, an.OPTION_DEPENDENT, None),
    (GATES, first_ok(), second_ok("option_unresolved"), None, an.INCONCLUSIVE, "option_unresolved"),
    (GATES, first_ok(), second_ok(), third(row(match=True), row(match=True)), an.SAME_CUE, None),
    (GATES, first_ok(), second_ok(), third(row(match=True), row()), an.ONE_DIRECTION, None),
    (GATES, first_ok(), second_ok(), third(row(), row(match=True)), an.ONE_DIRECTION, None),
    (GATES, first_ok(), second_ok(), third(row(match=True), row(opposes=True)), an.OPPOSES, None),
    (GATES, first_ok(), second_ok(), third(row(opposes=True), row()), an.OPPOSES, None),
    (GATES, first_ok(), second_ok(), third(row(null=True), row(null=True)), an.NOT_THIS_CUE, None),
    (GATES, first_ok(), second_ok(), third(row(pair_inside_margin=True), row(pair_inside_margin=True)), an.NO_TESTED_FEATURE, None),
    (GATES, first_ok(), second_ok(), third(row(), row()), an.INCONCLUSIVE, "cue_unresolved"),
    (GATES, first_ok(), second_ok(), third(row(significant=True, pair_inside_margin=True), row(pair_inside_margin=True)), an.INCONCLUSIVE, "cue_unresolved"),
    (GATES, first_ok(), second_ok(), third(row(match=True), row(match=True), sham=False), an.INCONCLUSIVE, "sham_failed"),
    (GATES, first_ok(), second_ok(), third(row(null_conditions={"c2_delivery": False, "c3_dose": True}), row()), an.INCONCLUSIVE, "region_too_small"),
    (GATES, first_ok(), second_ok(), third(row(null_conditions={"c2_delivery": True, "c3_dose": False}), row()), an.INCONCLUSIVE, "dose_failed"),
])
def test_decision_table(gates, one, two, three, expected, reason):
    out = an.verdict(gates, one, two, three)
    assert out["verdict"] == expected and out["reason_code"] == reason
    assert out["verdict"] in an.VERDICTS and (reason is None or reason in an.REASON_CODES)


@pytest.mark.parametrize("forward, back, c1, c2, expected, reason", [
    # One way, both directions of the same outcome: only the direction with an excess decides.
    ("carries", "absent", row(null=True), row(), an.NOT_THIS_CUE, None),
    ("absent", "carries", row(), row(null=True), an.NOT_THIS_CUE, None),
    ("carries", "absent", row(match=True), row(), an.ONE_DIRECTION, None),
    ("carries", "absent", row(opposes=True), row(), an.OPPOSES, None),
    # Plain "at least one" would pass these. A direction with no excess decides nothing.
    ("carries", "absent", row(), row(null=True), an.INCONCLUSIVE, "cue_unresolved"),
    ("absent", "carries", row(null=True), row(), an.INCONCLUSIVE, "cue_unresolved"),
    ("carries", "absent", row(), row(match=True, significant=True), an.INCONCLUSIVE, "cue_unresolved"),
    ("carries", "absent", row(), row(opposes=True, significant=True), an.INCONCLUSIVE, "cue_unresolved"),
    # A significant contrast in a dropped direction contradicts "no excess", so no null past it.
    ("carries", "absent", row(null=True), row(significant=True), an.INCONCLUSIVE, "cue_unresolved"),
    ("carries", "absent", row(pair_inside_margin=True), row(significant=True), an.INCONCLUSIVE, "cue_unresolved"),
    # Doubt about a direction keeps it: both must then be null, as before.
    ("carries", "unresolved", row(null=True), row(), an.INCONCLUSIVE, "cue_unresolved"),
    ("carries", "carries", row(null=True), row(), an.INCONCLUSIVE, "cue_unresolved"),
    # A null needs an excess on the edit orders in every direction in play.
    ("unresolved", "unresolved", row(null=True), row(null=True), an.INCONCLUSIVE, "cue_unresolved"),
    ("carries", "unresolved", row(null=True), row(null=True), an.INCONCLUSIVE, "cue_unresolved"),
    ("carries", "carries", row(null=True), row(null=True), an.NOT_THIS_CUE, None),
    # One way with nothing moving the pair is the other null.
    ("carries", "absent", row(pair_inside_margin=True), row(), an.NO_TESTED_FEATURE, None),
    # No direction left: nothing can test the cue.
    ("absent", "absent", row(null=True), row(null=True), an.INCONCLUSIVE, "cue_unresolved"),
])
def test_only_directions_with_an_excess_decide(forward, back, c1, c2, expected, reason):
    out = an.verdict(GATES, first_ok(), second_ok("stage3", forward, back), third(c1, c2))
    assert out["verdict"] == expected and out["reason_code"] == reason


def test_a_dropped_direction_does_not_enter_the_riders():
    keep = row(null=True, mask_uplow_ci90=[-0.01, 0.01])
    out = an.verdict(GATES, first_ok(), second_ok("stage3", "carries", "absent"), third(keep, row()))
    assert out["riders"]["sufficient_not_necessary"] is True
    assert out["riders"]["directions_used"] == ["Anger>Disgust"]


def test_direction_states_come_from_stage2_and_read_one_way_observers():
    obs = rec.Observer("one_way_cue", 1.0, [11])
    mcq, blind, bridge = obs.stage1_scores()
    cells = an.Cells(obs.w.items, rec.rows_stage1(obs, mcq, blind, bridge))
    states = an.stage2(cells, obs.w.partners, FAST)["directions"]
    assert states["Anger>Disgust"]["state"] == "carries"
    assert states["Disgust>Anger"]["state"] == "absent"


def test_stage3_losses_over_the_limit_are_not_tested():
    lost = {**third(row(match=True), row(match=True)), "over_lost_item_limit": True}
    assert an.verdict(GATES, first_ok(), second_ok(), lost)["verdict"] == an.NOT_TESTED


def test_geometry_gate_counts_clips_with_no_usable_apex(items):
    records = {i["trial_id"]: {"apex": 0, "landmarks": {"0": [[0, 0]]}} for i in items}
    assert an.geometry_passed(records, items)
    for trial in list(records)[:60]:
        records[trial]["landmarks"]["0"] = None
    assert not an.geometry_passed(records, items)


def test_a_missing_stage_or_unknown_state_is_a_fault():
    with pytest.raises(SystemExit):
        an.verdict(GATES, first_ok(), None, None)
    with pytest.raises(SystemExit):
        an.verdict(GATES, first_ok(), second_ok(), None)
    with pytest.raises(SystemExit):
        an.verdict(GATES, first_ok(), second_ok("other"), None)


def test_stage2_state_is_exhaustive():
    import itertools
    keys = ("x_passes", "x_null", "any_direction_carries", "all_directions_absent", "o_significant", "far_passes",
            "far_inside_margin", "blind_significant")
    states = {an.stage2_state(dict(zip(keys, values))) for values in itertools.product((True, False), repeat=len(keys))}
    assert states == {"not_shared", "shared_unresolved", "stage3", "option_dependent", "option_unresolved"}


def test_stage2_state_rules():
    base = {"x_passes": True, "x_null": False, "any_direction_carries": False, "all_directions_absent": True,
            "o_significant": True, "far_passes": False, "far_inside_margin": True, "blind_significant": False}
    assert an.stage2_state(base) == "option_dependent"
    assert an.stage2_state({**base, "blind_significant": True}) == "option_unresolved"
    assert an.stage2_state({**base, "far_inside_margin": False}) == "option_unresolved"
    assert an.stage2_state({**base, "far_passes": True}) == "stage3"
    assert an.stage2_state({**base, "o_significant": False}) == "stage3"
    assert an.stage2_state({**base, "x_passes": False}) == "shared_unresolved"
    assert an.stage2_state({**base, "x_passes": False, "x_null": True}) == "not_shared"
    # X can average a carrying direction away, so a carrying direction is a shared error.
    assert an.stage2_state({**base, "x_passes": False, "x_null": True, "any_direction_carries": True, "o_significant": False}) == "stage3"
    # And Not shared needs every direction absent, not merely none carrying.
    assert an.stage2_state({**base, "x_passes": False, "x_null": True, "all_directions_absent": False}) == "shared_unresolved"


def test_a_strongly_negative_x_is_not_shared_and_not_inconclusive(items, partners):
    """A model that separates the pair has no shared error, so its null must be reachable."""
    rng = np.random.default_rng(0)
    cells = an.Cells(items, [])
    scores = 2.0 * np.eye(6)[cells.emo] + rng.normal(0, 0.5, (len(cells.emo), 6))
    scores[cells.emo == A, B] -= 2.0
    scores[cells.emo == B, A] -= 2.0
    cells.mcq1[:] = scores[:, None, :] + rng.normal(0, 0.1, cells.mcq1.shape)
    cells.blind1[:] = scores
    out = an.stage2(cells, partners, FAST)
    assert out["pairs"][an.PAIR_NAMES[0]]["X"] < -0.05
    assert out["state"] == "not_shared"


def test_x_is_exact_under_a_position_effect_inside_each_order(holder):
    """r is taken per order, so option position cannot make a pair excess."""
    position = np.array([[list(o).index(label) for label in an.OPTIONS] for o in D["orders"]])
    effect = np.array([1.5, 0.75, 0.0, 0.0, -0.75, -1.5])[position]
    scores = 2.0 * np.eye(6)[holder.emo][:, None, :] + effect[None, :, :]
    p = an.softmax(scores)
    assert np.nanmean(an.pair_excess(holder, p, A, B)) == pytest.approx(0.0, abs=1e-12)


def test_text_ranks_read_label_neighbours():
    primes = np.random.default_rng(0).normal(0, 0.1, (6, 6, 6))
    primes[A, :, B] += 4.0
    primes[B, :, A] += 4.0
    out = an.text_ranks(primes)
    assert out["target_ranks"] == {"Anger>Disgust": 1, "Disgust>Anger": 2} or out["target_ranks"] == {"Anger>Disgust": 2, "Disgust>Anger": 1}
    assert out["label_neighbours_live"] is True
    assert an.text_ranks(None) is None


# Gates on the real pinned paths -------------------------------------------------


def gate_fixture(tmp_path, a1=True):
    """A reproduction folder from the real pinned gate files, with a made-up A1 row.

    The A1 row is written here and no A1 result is read. A3's summary, the vision gates and
    the delivery files are the real ones.
    """
    folder = tmp_path / "reproduction"
    folder.mkdir()
    real = ROOT / "research" / "reproduction"
    names = ["face_input_validity_summary.json"] + [f"{stem}_{m}.json" for stem in ("vision_gates", "input_delivery") for m in D["models"]]
    for name in names:
        shutil.copy(real / name, folder / name)
    (folder / "rotation_summary.json").write_text(json.dumps({m: {"vision_dependent": a1} for m in D["models"]}), encoding="utf-8")
    return folder


DELIVERY = {"delivery": [{"single_face": {
    "gate_2_every_frame_changes_output": True, "gate_3_logged_indices_are_the_frames_sent": True,
    "gate_4_ids_unchanged_within_clip": True, "tokens_across_view": 16.0, "repeat_forward_max_abs": 0.0}}]}


@pytest.mark.parametrize("a1", [True, False])
@pytest.mark.parametrize("model", D["models"])
def test_gate_loader_returns_true_on_the_real_pinned_paths(tmp_path, model, a1):
    gates = an.load_gates(gate_fixture(tmp_path, a1), model, DELIVERY, True, tmp_path / "no_a3_output")
    assert an.gates_passed(gates)


def test_gate_loader_says_not_passed_when_a_file_is_missing(tmp_path):
    folder = gate_fixture(tmp_path)
    (folder / "rotation_summary.json").unlink()
    gates = an.load_gates(folder, "gemma4", DELIVERY, True, tmp_path / "x")
    assert gates["rq0"] is False and not an.gates_passed(gates)


def test_gates_refuse_molmo_and_no_delivery(tmp_path):
    folder = gate_fixture(tmp_path)
    for name in ("vision_gates_molmo2.json", "input_delivery_molmo2.json"):
        shutil.copy(ROOT / "research" / "reproduction" / name, folder / name)
    assert not an.gates_passed(an.load_gates(folder, "molmo2", DELIVERY, True, tmp_path / "x"))
    assert not an.gates_passed(an.load_gates(folder, "gemma4", {"delivery": []}, True, tmp_path / "x"))


def test_end_to_end_from_real_gate_files_to_a_verdict(tmp_path):
    """The real paths, then a synthetic observer, then the verdict function."""
    gates = an.load_gates(gate_fixture(tmp_path), "gemma4", DELIVERY, True, tmp_path / "x")
    assert an.tokens_across_view(DELIVERY) == 16.0
    obs = rec.Observer("same_cue", 2.0, [1, 2, 3])
    w = obs.w
    mcq, blind, bridge = obs.stage1_scores()
    cells = an.Cells(w.items, rec.rows_stage1(obs, mcq, blind, bridge))
    first = an.stage1(cells, w.partners, FAST, descriptive=False)
    decided = an.decide_stage1({"gemma4": first})["gemma4"]
    assert decided["vision_dependent"] is True
    second = an.stage2(cells, w.partners, FAST)
    assert second["reaches_stage3"] is True
    cells.ingest(rec.rows_stage3(obs, obs.stage3_scores(mcq), sham_differs=False, span=0.2))
    stage = an.stage3(cells, second["cue_margin"], an.tokens_across_view(DELIVERY), FAST, dose_margins=second["dose_margin"])
    assert an.verdict(gates, decided, second, stage)["verdict"] == an.SAME_CUE


# Loading ------------------------------------------------------------------------


def stage1_rows(items, design="h", model="m"):
    out = []
    for item in items:
        for cell in ar.stage_plan(item, 1):
            out.append(dict(cell, status="ok", design_sha256=design, model=model))
    return out


def write_shard(tmp_path, rows, report):
    cells = tmp_path / "cells.jsonl"
    cells.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(report))
    return cells, run_file


def test_load_shards_accepts_a_complete_set_and_refuses_gaps(tmp_path, items):
    few = items[:5]
    report = {"model": "m", "design_sha256": "h", "shard": 0, "n_shards": 1, "mechanics": False, "dtype": "bfloat16"}
    cells, report_path = write_shard(tmp_path, stage1_rows(few), report)
    rows, _ = an.load_shards([cells], [report_path], "m", 1, "h", few)
    assert len(rows) == 5 * 9
    cells, report_path = write_shard(tmp_path, stage1_rows(few)[:-1], report)
    with pytest.raises(SystemExit):
        an.load_shards([cells], [report_path], "m", 1, "h", few)


def test_load_shards_refuses_other_designs_duplicates_and_partial_shards(tmp_path, items):
    few = items[:3]
    report = {"model": "m", "design_sha256": "h", "shard": 0, "n_shards": 2, "mechanics": False, "dtype": "bfloat16"}
    cells, report_path = write_shard(tmp_path, stage1_rows(few), report)
    with pytest.raises(SystemExit):
        an.load_shards([cells], [report_path], "m", 1, "h", few)
    report["n_shards"] = 1
    cells, report_path = write_shard(tmp_path, stage1_rows(few, design="other"), report)
    with pytest.raises(SystemExit):
        an.load_shards([cells], [report_path], "m", 1, "h", few)
    cells, report_path = write_shard(tmp_path, stage1_rows(few) * 2, report)
    with pytest.raises(SystemExit):
        an.load_shards([cells], [report_path], "m", 1, "h", few)


def test_main_refuses_without_the_freeze(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["a6_analyse", "--stage", "1"])
    with pytest.raises(SystemExit):
        an.main()
    monkeypatch.setattr(sys, "argv", ["a6_analyse", "--stage", "1", "--a6_frozen"])
    with pytest.raises(SystemExit):
        an.main()


# Recovery -----------------------------------------------------------------------


@pytest.mark.parametrize("name, strength, expected", [
    ("prior_only", 1.0, an.NO_FACE_USE),
    ("no_sharing", 1.0, an.NOT_SHARED),
    ("same_cue", 2.0, an.SAME_CUE),
    ("other_cue", 2.0, an.NOT_THIS_CUE),
    ("global_face", 2.0, an.NO_TESTED_FEATURE),
])
def test_observers_reach_their_rows(name, strength, expected, monkeypatch):
    """Mechanics, with the actor-by-region term off. Its effect on power is in the recovery file."""
    monkeypatch.setattr(rec, "SETTINGS", FAST)
    assert rec.one_replicate((name, strength, 0, 1.0, False, {"actor_region_sd": 0.0}))["verdict"] == expected


def test_a_variant_is_applied_once_per_worker(monkeypatch):
    """Applied per task, a variant must not wrap itself again, or a long run overflows."""
    monkeypatch.setattr(rec, "APPLIED", None)
    original = an.direction_states
    try:
        for _ in range(3000):
            rec.apply_variant("both_directions")
        assert an.direction_states is not original
        cells = an.Cells(rec.world().items, [])
        cells.mcq1[:] = np.random.default_rng(0).normal(0, 1, cells.mcq1.shape)
        states = an.direction_states(cells, FAST, range(6))
        assert all(v["state"] == "carries" for v in states.values())
        with pytest.raises(SystemExit):
            rec.apply_variant("first_draft")
    finally:
        an.direction_states = original


def test_the_sham_is_the_original_in_synthetic_cells():
    obs = rec.Observer("same_cue", 1.0, [1])
    mcq = obs.stage1_scores()[0]
    scores = obs.stage3_scores(mcq)
    assert np.array_equal(scores[:, an.CONDITIONS.index("sham_self_fill")], scores[:, an.ORIGINAL])


def test_descriptive_rows_and_sensitivities_run_on_a_synthetic_replicate():
    obs = rec.Observer("same_cue", 1.0, [5])
    w = obs.w
    mcq, blind, bridge = obs.stage1_scores()
    rows = rec.rows_stage1(obs, mcq, blind, bridge)
    for row in rows:
        if row["readout"] == "blind":
            row["blind"]["family_logprob"] = {**row["blind"]["family_logprob"], "Bored": -9.0}
    cells = an.Cells(w.items, rows)
    human = json.loads(d.HUMAN_TARGET.read_text(encoding="utf-8"))
    second = an.stage2(cells, w.partners, FAST)
    assert set(an.stage1(cells, w.partners, FAST)["secondary"]) >= {"accuracy_by_target_position", "within_emotion_r_ptarget_human"}
    pattern = an.pattern(cells, human)
    assert len(pattern["directed"]) == 30 and -1 <= pattern["rho_with_human"] <= 1 and pattern["ceiling"]
    assert "X_video_face" in an.bridge(cells, FAST)
    assert set(an.all_partners_b(cells)) == {"delta", "ci95", "p"}
    assert set(an.stage2_sensitivity(cells, FAST)) == {"Anger>Disgust", "Disgust>Anger"}
    baselines = {**DELIVERY, "baselines": [{"readout": "blind", "condition": "image_free", "status": "ok",
                                            "family_logprob": dict(zip(an.FAMILIES, [-3.0] * 6))}]}
    gate = an.gate_b_on_six(rows, w.partners, baselines)
    assert "B4" in gate and "pass" in gate
    cells.ingest(rec.rows_stage3(obs, obs.stage3_scores(mcq), sham_differs=False, span=0.2))
    third = an.stage3(cells, second["cue_margin"], 16.0, FAST, dose_margins=second["dose_margin"])
    assert isinstance(an.reading(third), str)
    out = an.stage3_sensitivities(cells, second["cue_margin"], 16.0, FAST, second["dose_margin"])
    assert {"reference_with_no_emotion", "reference_two_out_as_first_drafted", "control_Fear_only", "control_Sad_only",
            "edit_order_2_only", "edit_order_5_only", "plain_change_on_acted_clips", "equal_weight_clips"} <= set(out)
    subsets = an.clip_sensitivities(cells, w.partners, second["cue_margin"], 16.0, [], FAST, second["dose_margin"])
    assert {"human_mode_is_intended", "at_or_above_token_floor", "without_short_decode_h264"} <= set(subsets)
    assert subsets["without_short_decode_h264"]["n_clips"] == 819 - len(ar.SHORT_DECODE_CLIPS)


def test_recovery_output_is_on_this_design():
    out = json.loads(d.RECOVERY_OUT.read_text(encoding="utf-8"))
    assert out["binding_sha256"] == d.binding_hash()
    assert out["replicates"] == D["recovery"]["replicates"]
    assert out["rows_not_reached"] == [] and out["reason_codes_not_reached"] == []
    assert out["pass"] == (out["observers_pass"] and out["scenarios_pass"])


def test_shards_on_mixed_versions_or_another_dtype_are_refused(tmp_path, items):
    few = items[:3]
    first = {"model": "m", "design_sha256": "h", "shard": 0, "n_shards": 2, "mechanics": False, "dtype": "bfloat16", "versions": {"torch": "1"}}
    second = {**first, "shard": 1, "versions": {"torch": "2"}}
    rows = stage1_rows(few)
    cells = tmp_path / "cells.jsonl"
    cells.write_text("".join(json.dumps(r) + "\n" for r in rows))
    paths = []
    for n, report in enumerate((first, second)):
        paths.append(tmp_path / f"run{n}.json")
        paths[-1].write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="versions"):
        an.load_shards([cells], paths, "m", 1, "h", few)
    single = {**first, "n_shards": 1, "dtype": "float32"}
    paths[0].write_text(json.dumps(single))
    with pytest.raises(SystemExit, match="float32"):
        an.load_shards([cells], paths[:1], "m", 1, "h", few)


def test_x_is_exactly_zero_under_a_scale_per_emotion(holder):
    rng = np.random.default_rng(4)
    for _ in range(10):
        prior = rng.normal(0, 1.0, 6)
        scale = np.exp(rng.normal(0, 0.5, 6))
        scores = (prior + 2.0 * np.eye(6)[holder.emo]) * scale[holder.emo][:, None]
        p = an.softmax(scores)
        assert np.nanmean(an.pair_excess(holder, p, A, B)) == pytest.approx(0.0, abs=1e-12)


def test_happy_alone_is_seen_as_happy_and_not_as_the_partner(holder):
    def shift(scores):
        scores[np.isin(holder.emo, [A, B]), an.INDEX["Happy"]] += 0.3
        return scores

    cells = staged(holder, shift)
    c1 = contrast("C1")
    assert np.nanmean(an.double_difference(cells, c1, "mask_mid_neutral")) == pytest.approx(-0.15)
    assert np.nanmean(an.reference_change(cells, c1, "mask_mid_neutral")) == pytest.approx(-0.15)
    assert np.nanmean(an.partner_change(cells, c1, "mask_mid_neutral")) == pytest.approx(0.0, abs=1e-9)


def test_the_cue_margin_does_not_grow_with_noise(items, partners):
    """Signed changes are averaged inside cells, so pure noise leaves a small margin."""
    cells = an.Cells(items, [])
    rng = np.random.default_rng(0)
    margins = []
    for sd in (0.5, 2.0):
        cells.mcq1[:] = rng.normal(0, sd, cells.mcq1.shape)
        margins.append(an.cue_margin(cells, partners, contrast("C1")))
    old = []
    for sd in (0.5, 2.0):
        cells.mcq1[:] = rng.normal(0, sd, cells.mcq1.shape)
        keep = an.outcome_keep(cells, contrast("C1"), "fixed", D["control_emotions"])
        own = an.outcome(cells.mcq1[:, [2, 5]], B, keep)
        other = an.outcome(cells.mcq1[an.partner_index(cells, partners)][:, [2, 5]], B, keep)
        old.append(0.1 * float(np.nanmean(np.abs(other - own).mean(1))))
    assert old[1] / old[0] > 3.5  # the first draft's margin grew with the noise
    assert margins[1] < old[1] / 2


def write_stage(folder, model, stage, rows, design):
    for row in rows:
        row.update(model=model, design_sha256=design)
    (folder / f"cells_{model}_stage{stage}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report = {"model": model, "stage": stage, "mechanics": False, "design_sha256": design, "shard": 0, "n_shards": 1,
              "dtype": "bfloat16", "versions": {"torch": "x"}, "weights_sha256": "w"}
    (folder / f"run_{model}_stage{stage}.json").write_text(json.dumps(report), encoding="utf-8")


CLI_ROWS = [
    ("missing", None, an.NOT_TESTED, 1),
    ("prior_only", 1.0, an.NO_FACE_USE, 1),
    ("no_sharing", 1.0, an.NOT_SHARED, 2),
    ("adjacency", 2.0, an.OPTION_DEPENDENT, 2),
    ("same_cue", 2.0, an.SAME_CUE, 3),
    ("one_way_cue", 2.0, an.ONE_DIRECTION, 3),
    ("reverse", 2.0, an.OPPOSES, 3),
    ("other_cue", 2.0, an.NOT_THIS_CUE, 3),
    ("global_face", 2.0, an.NO_TESTED_FEATURE, 3),
    ("region_small", 1.0, an.INCONCLUSIVE, 3),
]


@pytest.mark.parametrize("name, strength, expected, stage_of_verdict", CLI_ROWS)
def test_every_row_reaches_an_output_file(tmp_path, monkeypatch, name, strength, expected, stage_of_verdict):
    """Each named row is written by main, at the stage where the model stops."""
    monkeypatch.setattr(an, "check_ready", lambda args: None)
    monkeypatch.setattr(an, "FULL_SETTINGS", FAST)
    design = d.block_hash()
    model = "gemma4"
    results = tmp_path / "results"
    results.mkdir()
    reproduction = gate_fixture(tmp_path)
    shutil.copy(ROOT / "research/reproduction/crema_d_reliability.json", reproduction)
    geometry = tmp_path / "geometry.json"
    items_now = rec.world().items
    records = {i["trial_id"]: {"apex": 0, "landmarks": {"0": [[0, 0]]}, "face_box": [0, 0, 100, 100], "face_width_px": 80.0} for i in items_now}
    geometry.write_text(json.dumps({"items": records}), encoding="utf-8")
    if name != "missing":
        obs = rec.Observer(name, strength, [4], 1.0, {"actor_region_sd": 0.0})
        mcq, blind, bridge = obs.stage1_scores()
        write_stage(results, model, 1, rec.rows_stage1(obs, mcq, blind, bridge), design)
        span = float(D["recovery"]["too_small_fraction"]) if name == "region_small" else 0.2
        write_stage(results, model, 3, rec.rows_stage3(obs, obs.stage3_scores(mcq), sham_differs=False, span=span), design)
        baselines = {**DELIVERY, "model": model, "design_sha256": design, "dtype": "bfloat16",
                     "baselines": [{"readout": "blind", "condition": "image_free", "order": None, "status": "ok",
                                    "family_logprob": dict(zip(an.FAMILIES, [-3.0] * 6))}]}
        (results / f"baselines_{model}.json").write_text(json.dumps(baselines), encoding="utf-8")

    def stage(number, *extra):
        monkeypatch.setattr(sys, "argv", ["a6_analyse", "--stage", str(number), "--results", str(results), "--geometry", str(geometry),
                                          "--reproduction", str(reproduction), "--a6_frozen", *extra])
        an.main()

    stage(1, "--models", model)
    first = json.loads((results / "a6_stage1.json").read_text(encoding="utf-8"))
    if stage_of_verdict == 1:
        assert first["verdicts"][model]["verdict"] == expected
        return
    assert first["verdicts"][model]["verdict"] is None
    stage(2, "--model", model)
    second = json.loads((results / f"a6_stage2_{model}.json").read_text(encoding="utf-8"))
    if stage_of_verdict == 2:
        assert second["verdict_if_stopped_here"]["verdict"] == expected
        return
    stage(3, "--model", model)
    third = json.loads((results / f"a6_stage3_{model}.json").read_text(encoding="utf-8"))
    assert third["verdict"]["verdict"] == expected
    assert third["stage2_sha256"] == __import__("hashlib").sha256((results / f"a6_stage2_{model}.json").read_bytes()).hexdigest()


def test_a_baselines_file_for_another_model_is_refused():
    with pytest.raises(SystemExit):
        an.check_baselines({"model": "qwen3vl", "design_sha256": d.block_hash(), "dtype": "bfloat16"}, "gemma4", d.block_hash())
    with pytest.raises(SystemExit):
        an.check_baselines({"model": "gemma4", "design_sha256": d.block_hash(), "dtype": "float32"}, "gemma4", d.block_hash())


def test_the_command_line_runs_stage_by_stage_on_synthetic_cells(tmp_path, monkeypatch):
    """Every stage of main, from files, on a synthetic observer standing in for a model."""
    monkeypatch.setattr(an, "check_ready", lambda args: None)  # geometry is PENDING until the cluster
    monkeypatch.setattr(an, "FULL_SETTINGS", FAST)
    design = d.block_hash()
    model = "gemma4"
    results = tmp_path / "results"
    results.mkdir()
    reproduction = gate_fixture(tmp_path)
    shutil.copy(ROOT / "research/reproduction/crema_d_reliability.json", reproduction)
    obs = rec.Observer("same_cue", 2.0, [7], 1.0, {"actor_region_sd": 0.0})
    mcq, blind, bridge = obs.stage1_scores()
    write_stage(results, model, 1, rec.rows_stage1(obs, mcq, blind, bridge), design)
    write_stage(results, model, 3, rec.rows_stage3(obs, obs.stage3_scores(mcq), sham_differs=False, span=0.2), design)
    primes = obs.primes()
    baselines = [{"readout": "mcq", "condition": f"prime_{label}", "order": row, "options": list(D["orders"][row]), "status": "ok",
                  "joint_logprob": primes[an.INDEX[label], row][[an.INDEX[o] for o in D["orders"][row]]].tolist()}
                 for label in an.OPTIONS for row in range(6)]
    baselines.append({"readout": "blind", "condition": "image_free", "order": None, "status": "ok",
                      "family_logprob": dict(zip(an.FAMILIES, [-3.0] * 6))})
    (results / f"baselines_{model}.json").write_text(json.dumps({**DELIVERY, "baselines": baselines, "model": model,
                                                                "design_sha256": design, "dtype": "bfloat16"}), encoding="utf-8")
    geometry = tmp_path / "geometry.json"
    records = {i["trial_id"]: {"apex": 0, "landmarks": {"0": [[0, 0]]}, "face_box": [0, 0, 100, 100], "face_width_px": 80.0} for i in obs.w.items}
    geometry.write_text(json.dumps({"items": records}), encoding="utf-8")

    def stage(number, *extra):
        monkeypatch.setattr(sys, "argv", ["a6_analyse", "--stage", str(number), "--results", str(results), "--geometry", str(geometry),
                                          "--reproduction", str(reproduction), "--a6_frozen", *extra])
        an.main()

    stage(1, "--models", model)
    first = json.loads((results / "a6_stage1.json").read_text(encoding="utf-8"))
    assert first["decided"][model]["vision_dependent"] is True and first["design_sha256"] == design
    assert "B4" in first["descriptive"][model]["gate_b_descriptive"]
    stage(2, "--model", model)
    second = json.loads((results / f"a6_stage2_{model}.json").read_text(encoding="utf-8"))
    assert second["stage2"]["reaches_stage3"] is True
    assert set(second["stage2"]["cue_margin"]) == {"C1", "C2", "C3", "C4"}
    assert second["text_primes"]["target_ranks"]
    stage(3, "--model", model)
    third = json.loads((results / f"a6_stage3_{model}.json").read_text(encoding="utf-8"))
    assert third["verdict"]["verdict"] == an.SAME_CUE
    assert third["repeat_check_max_abs"] == 0.0
    assert third["verdict"]["scope"] == an.SCOPE
    with pytest.raises(SystemExit):  # an output is never overwritten
        stage(3, "--model", model)
