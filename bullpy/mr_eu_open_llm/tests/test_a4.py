import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from scripts import a4_analyse as an  # noqa: E402
from scripts import a4_design as d  # noqa: E402
from scripts import a4_recovery as rec  # noqa: E402
from scripts import a4_run as run  # noqa: E402
import a4_stimuli as stim  # noqa: E402

AU = an.hae.AU_COLS


def fake_frames(n=40):
    rng = np.random.default_rng(0)
    frames = {}
    for key in d.DATASETS:
        rows = []
        for i in range(n):
            au = {c: 0.0 for c in AU}
            au[AU[i % 33]] = 0.5
            for g in ("F", "M"):
                rows.append({"cfg": f"{key}_{i}", "face_gender": g, **au})
        frames[key] = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(100)))
    return frames


def manifest(monkeypatch, n=40):
    monkeypatch.setattr(d, "CONFIGS_PER_DATASET", 20)
    monkeypatch.setattr(d, "REPLICATE_CONFIGS", 5)
    monkeypatch.setattr(d, "SCALE_CONFIGS", 4)
    monkeypatch.setattr(d, "BLIND_CONFIGS", 6)
    return stim.build(fake_frames(n), AU).to_dict("records")


def test_rotations_are_complete_and_abstain_stays_last():
    for task, options in d.OPTIONS.items():
        named = len(options) - 1
        assert len(d.rotations(task)) == named
        first = {d.rotated_options(task, r)[0] for r in d.rotations(task)}
        assert len(first) == named
        assert all(d.rotated_options(task, r)[-1] == d.ABSTAIN for r in d.rotations(task))


def test_design_hash_changes_with_a_constant(monkeypatch):
    before = d.design_hash()
    monkeypatch.setattr(d, "SESOI_CFA", 0.31)
    assert d.design_hash() != before


def test_options_line_up_with_human_columns():
    assert list(d.HUMAN_LABELS["basic"]) == an.hae.LABEL_ORDER["emotion"]
    assert list(d.HUMAN_LABELS["conversational"]) == an.hae.LABEL_ORDER["state"]


def test_manifest_is_deterministic_and_order_free(monkeypatch):
    a = manifest(monkeypatch)
    frames = {k: v.sample(frac=1, random_state=3) for k, v in fake_frames().items()}
    b = stim.build(frames, AU).to_dict("records")
    assert a == b


def test_manifest_counts_and_identity_group(monkeypatch):
    rows = pd.DataFrame(manifest(monkeypatch))
    main = rows[rows.set == "main"]
    assert (main.groupby("dataset").size() == 20).all()
    assert set(main[main.dataset.str.endswith("EA")].identity) <= {"ea_female", "ea_male"}
    assert set(main[main.dataset.str.endswith("WC")].identity) <= {"female_a", "female_b", "male_a", "male_b"}
    replicate = rows[(rows.set == "replicate")]
    assert len(replicate) == 5 * 4
    merged = replicate.merge(main, on=["dataset", "cfg"])
    assert ((merged.identity_x != merged.identity_y) | (merged.seed_x != merged.seed_y)).all()


def test_one_au_set_has_every_column_and_dose():
    rows = pd.DataFrame(stim.one_au_rows(AU))
    assert len(rows[rows.set == "neutral"]) == len(d.ONE_AU_IDENTITIES)
    assert len(rows[rows.set == "one_au"]) == 33 * 3 * len(d.ONE_AU_IDENTITIES)
    gender = rows.drop_duplicates("identity").set_index("identity").gender
    assert gender["ea_female"] == "F" and gender["male_a"] == "M" and gender["female_b"] == "F"


def test_scaling_stays_in_range():
    assert stim.scaled({"AU01": 0.9, "AU02L": 0.4}, 1.25) == {"AU01": 1.0, "AU02L": 0.5}


def test_stim_id_shares_an_identical_image():
    assert stim.stim_id("ea_male", 0, {"AU01": 0.5}) == stim.stim_id("ea_male", 0, {"AU01": 0.5})
    assert stim.stim_id("ea_male", 1, {"AU01": 0.5}) != stim.stim_id("ea_male", 0, {"AU01": 0.5})


def test_cells_partition_across_shards(monkeypatch):
    rows = manifest(monkeypatch)
    one = [run.cell_key(c) for c in run.planned_cells(rows, 0, 1)]
    split = [run.cell_key(c) for s in range(3) for c in run.planned_cells(rows, s, 3)]
    assert len(one) == len(set(one))
    assert sorted(one) == sorted(split)
    readouts = {k[3] for k in one}
    assert readouts == {"mcq", "image_free", "blind"}


def test_a_stimulus_needs_a_full_rotation_set():
    rows = []
    for r in d.rotations("conversational"):
        rows.append({"stim_id": "a", "task": "conversational", "rotation": r, "readout": "mcq", "status": "ok",
                     "options": d.rotated_options("conversational", r), "joint_logprob": [0, -1, -2, -3, -4]})
    rows.append({"stim_id": "b", "task": "conversational", "rotation": 0, "readout": "mcq", "status": "ok",
                 "options": d.rotated_options("conversational", 0), "joint_logprob": [0, -1, -2, -3, -4]})
    out = an.stimulus_responses(rows, "conversational")
    assert set(out) == {"a"}
    assert out["a"].sum() == pytest.approx(1)


def test_position_preference_cancels_over_rotations():
    # A system that always favours position 1: over a full set the named options tie.
    rows = [{"stim_id": "a", "task": "basic", "rotation": r, "readout": "mcq", "status": "ok",
             "options": d.rotated_options("basic", r), "joint_logprob": [0, -3, -3, -3, -3, -3, -3]}
            for r in d.rotations("basic")]
    p = an.stimulus_responses(rows, "basic")["a"]
    assert np.ptp(p[:-1]) < 1e-12


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.choice((0.0, 0.5, 1.0), size=(n, 33), p=(0.85, 0.1, 0.05))
    gender = rng.choice((-0.5, 0.5), size=n)
    W = rng.normal(size=(5, 33))
    W -= W.mean(axis=0)
    return X, gender, W, rng


def test_fit_tau_recovers_a_cue_matrix_and_sums_to_zero():
    X, gender, W, rng = synthetic()
    p = rec.respond(W, np.log(np.ones(5) / 5), X, gender, rng, 0.0)
    tau = an.fit_tau(p, X, gender, 0.1)
    assert tau.shape == (5, 25)
    assert np.abs(tau.sum(axis=0)).max() < 1e-8
    other = rec.respond(W, np.log(np.ones(5) / 5), np.random.default_rng(1).permutation(X), gender, rng, 0.0)
    assert an.cosine(tau, an.fit_tau(other, np.random.default_rng(1).permutation(X), gender, 0.1)) > 0.97


def test_cue_null_rejects_a_common_effect():
    rng = np.random.default_rng(0)
    human = rng.normal(size=(5, 25))
    human -= human.mean(axis=0)
    common = np.tile(human.mean(axis=1, keepdims=True) + rng.normal(size=(5, 1)), (1, 25))
    assert an.cue_null_p(common, human, 999, rng)[0] > 0.2
    assert an.cue_null_p(human, human, 999, rng)[0] < 0.01


def test_signal_test_separates_cues_from_noise():
    X, gender, W, rng = synthetic()
    base = np.log(np.ones(5) / 5)
    assert an.signal_test(rec.respond(W, base, X, gender, rng, 0.5), X, gender, 99, rng) <= 0.02
    flat = rec.respond(np.zeros_like(W), base, X, gender, rng, 0.5)
    ps = [an.signal_test(flat, X, gender, 99, np.random.default_rng(i)) for i in range(3)]
    assert max(ps) > 0.02


@pytest.mark.parametrize("gates,p_sig,p_holm,cfa,ci,want", [
    ({"a": True}, 0.01, 0.01, 0.7, (0.6, 0.8), "Aligned"),
    ({"a": True}, 0.01, 0.01, 0.2, (0.1, 0.25), "Not aligned"),
    ({"a": True}, 0.01, 0.01, 0.35, (0.2, 0.5), "Aligned"),
    ({"a": True}, 0.01, 0.2, 0.35, (0.2, 0.5), "Inconclusive"),
    ({"a": False}, 0.01, 0.01, 0.7, (0.6, 0.8), "Inconclusive"),
    ({}, 0.01, 0.01, 0.7, (0.6, 0.8), "Inconclusive"),
    ({"a": None}, 0.01, 0.01, 0.7, (0.6, 0.8), "Inconclusive"),
    ({"a": True}, 0.5, 0.01, 0.1, (0.0, 0.2), "Inconclusive"),
    ({"a": True}, float("nan"), 0.01, 0.7, (0.6, 0.8), "Inconclusive"),
])
def test_decision_rule(gates, p_sig, p_holm, cfa, ci, want):
    assert an.decide(gates, p_sig, p_holm, cfa, ci) == want


def test_apply_rule_holm_and_molmo_descriptive():
    base = {"p_signal": 0.01, "cfa": 0.6, "ci": (0.5, 0.7), "lost_ok": True}
    results = {"gemma4": {**base, "p_cue": 0.03}, "qwen3vl": {**base, "p_cue": 0.04}, "molmo2": {**base, "p_cue": 0.0001}}
    an.apply_rule(results, {m: {"rq0": True} for m in results})
    assert results["gemma4"]["p_holm"] == pytest.approx(0.06)
    assert results["gemma4"]["decision"] == "Inconclusive"
    assert results["molmo2"]["decision"] == "Descriptive"


def test_lost_stimuli_fail_the_gate():
    base = {"p_signal": 0.01, "cfa": 0.6, "ci": (0.5, 0.7), "p_cue": 0.001, "lost_ok": False}
    results = {"gemma4": dict(base)}
    an.apply_rule(results, {"gemma4": {"rq0": True}})
    assert results["gemma4"]["decision"] == "Inconclusive"


def test_replicate_check_finds_a_carried_config():
    rng = np.random.default_rng(0)
    first = rng.dirichlet(np.ones(5), size=100)
    near = 0.9 * first + 0.1 * rng.dirichlet(np.ones(5), size=100)
    assert an.replicate_check(first, near, rng)["p"] < 0.01
    assert an.replicate_check(first, rng.dirichlet(np.ones(5), size=100), rng)["p"] > 0.01


def test_validity_gate_treats_missing_as_failed():
    p = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (5, 1))
    gate = an.validity_gate(p, {}, None, None)
    assert not any(gate.values()) or gate == {"v1_detector": False, "v2_image_dependence": False,
                                              "v3_config_carries": False, "v4_domain": True}
    passed = an.validity_gate(p, {"p": 0.001, "delta_js": 0.1}, 0.3, {"pass": True})
    assert all(passed.values())
    all_abstain = an.validity_gate(np.tile([0, 0, 0, 0.05, 0.95], (5, 1)), {"p": 0.001, "delta_js": 0.1}, 0.3, {"pass": True})
    assert all_abstain["v4_domain"] is False


def test_detector_gate_lists_unconfirmed_columns():
    rows = [{"au": "AU12L", "identity": "a", "amplitude": a, "intensity": a} for a in d.ONE_AU_AMPLITUDES]
    rows += [{"au": "AU04", "identity": "a", "amplitude": a, "intensity": 1 - a} for a in d.ONE_AU_AMPLITUDES]
    out = an.detector_gate(rows)
    assert out["share_rising"] == 0.5 and out["pass"] is False
    assert "AU22" in out["unconfirmed_columns"] and "AU04:a" in out["failing"]


def test_behaviour_metrics_can_be_high_when_effects_differ():
    a = np.random.default_rng(0).dirichlet(np.ones(4), size=50)
    out = an.behaviour_metrics(a, a)
    assert out["output_r"] == pytest.approx(1) and out["modal_kappa"] == pytest.approx(1)


def test_direct_effects_sum_bilateral_sides():
    rows = [{"set": "neutral", "identity": "i", "seed": 0, "stim_id": "n", "cfg": "neutral", "au_json": "{}"},
            {"set": "one_au", "identity": "i", "seed": 0, "stim_id": "l", "cfg": "AU12L", "au_json": '{"AU12L": 1.0}'},
            {"set": "one_au", "identity": "i", "seed": 0, "stim_id": "r", "cfg": "AU12R", "au_json": '{"AU12R": 1.0}'}]
    responses = {"n": np.array([0.5, 0.5]), "l": np.array([0.6, 0.4]), "r": np.array([0.7, 0.3])}
    tau, dose = an.direct_effects(responses, rows)
    assert tau[:, an.FAMILIES.index("AU12")] == pytest.approx([0.3, -0.3])
    assert len(dose) == 1


def test_read_shards_refuses_other_hash(tmp_path):
    path = tmp_path / "cells_gemma4_0.jsonl"
    path.write_text(json.dumps({"design_sha256": "x"}) + "\n")
    with pytest.raises(SystemExit):
        an.read_shards([path], "y")


def test_runner_refuses_before_the_freeze(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["a4_run", "--model", "gemma4", "--shard", "0", "--n_shards", "2"])
    pd_stub = pytest.importorskip("PIL")  # noqa: F841
    with pytest.raises(SystemExit, match="not frozen"):
        run.main()


def test_pair_swap_is_exact():
    W = np.random.default_rng(0).normal(size=(5, 33))
    swapped = rec.swap_pairs(W)
    i, j = [an.FAMILY_MEMBERS[an.FAMILIES.index(n)] for n in ("AU25", "AU26")]
    assert swapped[:, i[0]] == pytest.approx(W[:, j].sum(axis=1) / len(i))
    unpaired = an.FAMILY_MEMBERS[an.FAMILIES.index("AU38")]
    assert swapped[:, unpaired] == pytest.approx(W[:, unpaired])


def test_recovery_smoke(monkeypatch):
    rows = pd.DataFrame(manifest(monkeypatch, 60))
    assert len(rows)
    X, gender, _W, _ = synthetic(300)
    W = _W[:5]
    human = np.random.default_rng(1).normal(size=(5, 25))
    human -= human.mean(axis=0)
    task = ("human_like", "conversational_EA", 1.0, 0, X, gender, W, np.log(np.ones(5) / 5), human, 1.0, 5, 99, 19)
    out = rec.one_replicate(task)
    assert out["decision"] in an.OUTCOMES and -1 <= out["cfa"] <= 1


def test_verdict_rule_binds_at_half_and_one():
    good = {"aligned": 0.9, "not_aligned": 0.9, "signal_rate": 0.05, "alignment_p_rate": 0.9,
            "share_cfa_error_within_limit": 0.95}
    cells = {o: dict(good, aligned=0.0) for o in rec.OBSERVERS}
    cells["human_like"] = cells["half_shared"] = good
    out = rec.verdict({"basic_EA|1.0": cells, "basic_EA|0.25": {**cells, "human_like": {**good, "aligned": 0.1}}})
    assert all(out.values()) and "basic_EA|0.25|human_like" not in out
    worse = rec.verdict({"basic_EA|1.0": {**cells, "human_like": {**good, "aligned": 0.5}}})
    assert worse["basic_EA|1.0|human_like"] is False


def test_renders_go_into_one_tar_atomically(monkeypatch, tmp_path):
    import tarfile
    import types

    fake = types.SimpleNamespace(render_face=lambda au, identity, seed: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setitem(sys.modules, "render_au_faces", fake)
    monkeypatch.setattr(d, "RENDER_TARS", 2)
    frame = pd.DataFrame([{"stim_id": f"s{i}", "identity": "x", "seed": 0, "au_json": "{}"} for i in range(6)])
    total = sum(stim.render_tar(frame, tmp_path, i) for i in range(2))
    assert total == 6
    assert sorted(p.name for p in tmp_path.iterdir()) == ["renders_00.tar", "renders_01.tar"]
    store = run.RenderStore(tmp_path)
    assert store.image("s3").shape == (8, 8, 3)
    with tarfile.open(tmp_path / "renders_00.tar") as tar:
        assert all(m.name.endswith(".png") for m in tar.getmembers())
    assert stim.render_tar(frame, tmp_path, 0) == 0


def test_design_asset_default_is_not_a_scratch_path():
    import render_au_faces as raf

    assert "scratchpad" not in str(raf.DEFAULT_ASSETS)
