"""Known-answer tests for paper-v2 metric primitives."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_v2.metrics import (  # noqa: E402
    binomial_gt_chance,
    cosine_rdm,
    loo_ridge_rho,
    rsa_spearman,
    shannon_entropy,
    spearman_rho,
)
from paper_v2.schema import actor_id_from_eu_filename, validate_item  # noqa: E402


def test_entropy_uniform_six():
    h = shannon_entropy([1, 1, 1, 1, 1, 1])
    assert abs(h - math.log(6)) < 1e-12


def test_entropy_delta_is_zero():
    assert shannon_entropy([1, 0, 0, 0, 0, 0]) == 0.0


def test_spearman_perfect():
    rho, p = spearman_rho([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert abs(rho - 1.0) < 1e-12
    assert p < 1e-3


def test_spearman_anti():
    rho, _ = spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert abs(rho + 1.0) < 1e-12


def test_rdm_identical_rows_zero():
    X = np.ones((4, 3))
    rdm = cosine_rdm(X)
    assert rdm.shape == (4, 4)
    assert np.allclose(rdm, 0.0)


def test_rdm_orthogonal():
    X = np.eye(3)
    rdm = cosine_rdm(X)
    assert np.allclose(np.diag(rdm), 0.0)
    off = rdm[np.triu_indices(3, k=1)]
    assert np.allclose(off, 1.0)


def test_rsa_identical_rdms():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    rdm = cosine_rdm(X)
    assert abs(rsa_spearman(rdm, rdm) - 1.0) < 1e-12


def test_loo_ridge_recovers_linear_target():
    rng = np.random.default_rng(0)
    n, d = 40, 6
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = X @ w
    rho = loo_ridge_rho(X, y)
    assert rho > 0.9


def test_binomial_molmo_not_above_chance():
    p = binomial_gt_chance(48, 243, 1.0 / 6.0)
    assert p > 0.05
    assert abs(p - 0.11554654230151924) < 1e-12


def test_binomial_qwen_above_chance():
    p = binomial_gt_chance(92, 243, 1.0 / 6.0)
    assert p < 1e-14


def test_validate_item_and_actor_parse():
    item = validate_item(
        {
            "item_id": "emotions/EDITED/Unfriendly/NF25_cut.mp4",
            "stimulus_path": "emotions/EDITED/Unfriendly/NF25_cut.mp4",
            "options": ["Unfriendly", "Bored", "Hurt", "Frustrated", "Surprised", "None of the above"],
            "target_idx": 0,
            "human_counts": [34, 18, 12, 5, 0, 4],
            "human_n": 73,
            "category": "Unfriendly",
        }
    )
    assert actor_id_from_eu_filename(item["item_id"]) == "N"
    assert actor_id_from_eu_filename("emotions_11/EDITED/Ashamed/Of13_cut.mp4") == "O"
    assert item["target_idx"] == 0
