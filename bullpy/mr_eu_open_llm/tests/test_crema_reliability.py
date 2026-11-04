from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "research/scripts/crema_reliability.py"
spec = importlib.util.spec_from_file_location("crema_reliability", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_krippendorff_alpha_hand_worked_case():
    assert abs(mod.krippendorff_alpha(np.array([[2, 0], [0, 2], [1, 1]])) - (1 - 10 / 18)) < 1e-12


def test_krippendorff_alpha_is_one_under_perfect_agreement():
    assert mod.krippendorff_alpha(np.array([[5, 0, 0], [0, 4, 0], [0, 0, 6]])) == 1.0


def test_half_mask_splits_each_clip_and_ignores_padding():
    votes = np.array([[0, 1, 2, 3, -1, -1], [0, 0, 1, 1, 2, -1]])
    rng = np.random.default_rng(0)
    for _ in range(50):
        mask = mod.half_mask(votes, rng)
        assert not (mask & (votes < 0)).any()
        assert mask[0].sum() == 2
        assert mask[1].sum() in (2, 3)


def test_leave_one_rater_modal_shares_credit_on_ties():
    assert mod.leave_one_rater_modal(np.array([[4, 0]]))[0] == 1.0
    # A rater from the 2-vote option sees 1 v 1 (half credit); the 1-vote rater sees 2 v 0 (none).
    assert abs(mod.leave_one_rater_modal(np.array([[2, 1]]))[0] - 1 / 3) < 1e-12


def test_ceiling_drops_when_noise_is_added():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(6), size=400)
    clean = np.stack([rng.multinomial(10, row) for row in p])
    votes = np.stack([np.repeat(np.arange(6), row) for row in clean])
    emo = np.arange(400) % 6
    raters = np.arange(votes.size).reshape(votes.shape) % 500
    high = mod.analyse(votes, raters, emo, emo, np.zeros(400, bool), n_splits=20, n_rater_splits=2)
    noisy = np.where(rng.random(votes.shape) < 0.7, rng.integers(0, 6, votes.shape), votes)
    low = mod.analyse(noisy, raters, emo, emo, np.zeros(400, bool), n_splits=20, n_rater_splits=2)
    assert low["all_clips"]["p_target_pooled"]["half_r"] < high["all_clips"]["p_target_pooled"]["half_r"]
