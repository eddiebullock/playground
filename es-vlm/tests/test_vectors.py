from __future__ import annotations

import numpy as np

from esvlm.cli.analyze_extract import cosine
from esvlm.cli.extract import parse_layers
from esvlm.vectors.extract import difference_of_means, directions_for_layers
from esvlm.vectors.contrasts import default_contrasts
from esvlm.vectors.narratives import CONTROL_NARRATIVES, E_NARRATIVES, S_NARRATIVES
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
