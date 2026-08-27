"""Tests for human confusion RDM builder (study3 EU mech)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_human_confusion import (
    build_confusion_artifacts,
    build_rdm,
    js_divergence,
)


def test_js_divergence_identical_is_zero() -> None:
    p = [0.5, 0.3, 0.2]
    assert js_divergence(p, p) == 0.0


def test_build_confusion_on_repo_data() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data" / "eu_emotions_full_manifest.json"
    human = root / "data" / "eu_emotions_human_entropy.json"
    if not manifest.is_file() or not human.is_file():
        return

    out = build_confusion_artifacts(manifest, human)
    rdm = out["_rdm"]
    assert rdm.shape == (243, 243)
    assert out["n_trials"] == 243
    assert len(out["label_pair_confusion"]) > 0
    assert np.allclose(np.diag(rdm), 0.0)
    upper = rdm[np.triu_indices(243, k=1)]
    assert float(upper.min()) >= 0.0
    assert float(upper.max()) <= np.log(2) + 0.01  # JS upper bound for same support


def test_build_rdm_symmetric() -> None:
    vecs = [np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1]), np.array([0.33, 0.33, 0.34])]
    rdm = build_rdm(vecs)
    assert rdm.shape == (3, 3)
    assert np.allclose(rdm, rdm.T)
