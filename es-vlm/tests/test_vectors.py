from __future__ import annotations

import json

import numpy as np

from esvlm.analysis.separability import residualize, unique_fraction
from esvlm.cli.analyze_extract import cosine
from esvlm.cli.extract import parse_layers
from esvlm.cli.report_rq1 import report_model
from esvlm.vectors.extract import difference_of_means, directions_for_layers
from esvlm.vectors.contrasts import default_contrasts
from esvlm.vectors.narratives import CONTROL_NARRATIVES, E_NARRATIVES, S_NARRATIVES
from esvlm.vectors.personas import persona_matrix, persona_texts
from esvlm.vectors.validate import projection_auroc


def test_narratives_are_original_and_long_enough() -> None:
    assert len(E_NARRATIVES) >= 40
    assert len(S_NARRATIVES) >= 40
    assert len(CONTROL_NARRATIVES) >= 40
    assert set(E_NARRATIVES).isdisjoint(S_NARRATIVES)
    assert set(E_NARRATIVES).isdisjoint(CONTROL_NARRATIVES)
    assert set(S_NARRATIVES).isdisjoint(CONTROL_NARRATIVES)
    # crude guard against pasted questionnaire items
    banned = ("I can easily tell if someone else wants to enter a conversation",)
    blob = " ".join(E_NARRATIVES + S_NARRATIVES + CONTROL_NARRATIVES)
    for phrase in banned:
        assert phrase not in blob


def test_default_contrasts_share_control() -> None:
    pairs = default_contrasts()
    control = list(CONTROL_NARRATIVES)
    assert pairs["E"].positive == list(E_NARRATIVES)
    assert pairs["S"].positive == list(S_NARRATIVES)
    assert pairs["E"].negative == control
    assert pairs["S"].negative == control
    assert pairs["E"].positive != pairs["S"].positive
    assert pairs["E_minus_S"].positive == list(E_NARRATIVES)
    assert pairs["E_minus_S"].negative == list(S_NARRATIVES)
    assert pairs["E_template"].negative == pairs["S_template"].negative
    assert pairs["E_template"].positive != pairs["S_template"].positive


def test_difference_of_means_separates() -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(20, 16)) + np.array([2.0] + [0.0] * 15)
    neg = rng.normal(size=(20, 16)) - np.array([2.0] + [0.0] * 15)
    v = difference_of_means(pos, neg)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert projection_auroc(pos, neg, v) > 0.9


def test_parse_layers_and_stack() -> None:
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("0,3", 4) == [0, 3]
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(10, 3, 8)) + 1.0
    neg = rng.normal(size=(10, 3, 8)) - 1.0
    vecs = directions_for_layers(pos, neg)
    assert vecs.shape == (3, 8)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    a = np.array([1.0, 0.0])
    assert abs(cosine(a, a) - 1.0) < 1e-6
    assert abs(cosine(a, np.array([0.0, 1.0]))) < 1e-6


def test_unique_fraction_and_residualize() -> None:
    e = np.array([1.0, 0.0, 0.0])
    s_par = np.array([2.0, 0.0, 0.0])
    s_orth = np.array([0.0, 1.0, 0.0])
    assert unique_fraction(e, s_par) < 1e-6
    assert abs(unique_fraction(e, s_orth) - 1.0) < 1e-6
    leftover = residualize(np.array([1.0, 1.0, 0.0]), e)
    assert abs(float(leftover @ (e / np.linalg.norm(e)))) < 1e-6


def test_persona_texts_are_mundane_and_unique() -> None:
    texts = persona_texts(80, seed=0)
    assert len(texts) == 80
    assert len(set(texts)) == 80
    banned = ("I can easily tell if someone else wants to enter a conversation",)
    blob = " ".join(texts)
    for phrase in banned:
        assert phrase not in blob
    assert set(texts).isdisjoint(E_NARRATIVES)
    assert set(texts).isdisjoint(S_NARRATIVES)
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(40, 8))
    e = np.zeros(8)
    e[0] = 1.0
    s = np.zeros(8)
    s[0] = 1.0
    scores = persona_matrix(acts, e, s)
    assert abs(scores.correlation - 1.0) < 1e-6


def test_report_rq1_from_saved_json(tmp_path) -> None:
    extract = tmp_path / "extract"
    extract.mkdir()
    rng = np.random.default_rng(0)
    e = rng.normal(size=8)
    e = e / np.linalg.norm(e)
    s = residualize(rng.normal(size=8), e)
    np.savez_compressed(extract / "toy__layer_00.npz", E=e, S=s)
    sep = {
        "model": "toy",
        "n_layers": 1,
        "cos_E_S_mean": 0.0,
        "E_held_out_auroc_mean": 0.9,
        "S_held_out_auroc_mean": 0.91,
        "layers": [
            {
                "layer": 0,
                "cos_E_S": 0.0,
                "E": {"held_out_auroc": 0.9},
                "S": {"held_out_auroc": 0.91},
                "E_minus_S": {"held_out_auroc": 0.88},
            }
        ],
    }
    out = tmp_path / "agg"
    out.mkdir()
    (out / "separability_toy.json").write_text(json.dumps(sep), encoding="utf-8")
    summary = report_model(extract, out)
    assert summary["model"] == "toy"
    assert summary["candidate_layer"] == 0
    assert abs(float(summary["E_vs_S_held_mean"]) - 0.88) < 1e-9
    assert summary["unique_E_frac_mean"] > 0.9
    assert (out / "rq1_toy.json").exists()
