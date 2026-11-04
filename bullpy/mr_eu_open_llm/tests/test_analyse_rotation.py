"""Decision statistics for prereg amendment A1."""

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "research/scripts/analyse_rotation.py"
spec = importlib.util.spec_from_file_location("analyse_rotation", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_sign_flip_detects_effect_shared_by_all_actors():
    actors = np.repeat(list("abcdefghijklmnopq"), 5)
    assert mod.actor_sign_flip_p(np.full(len(actors), 0.2), actors, n_perm=20000) < 0.001


def test_sign_flip_null_when_effect_cancels_across_actors():
    actors = np.repeat(list("abcdefghijklmnopq"), 5)
    signs = np.repeat([1.0, -1.0] * 8 + [0.0], 5)
    assert mod.actor_sign_flip_p(signs, actors, n_perm=20000) > 0.2


def test_holm_matches_hand_calculation():
    assert mod.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_bootstrap_ci_contains_mean():
    actors = np.repeat(list("abcdef"), 4)
    delta = np.arange(len(actors), dtype=float) / 10
    low, high = mod.actor_bootstrap_ci(delta, actors, n_boot=2000)
    assert low < delta.mean() < high
