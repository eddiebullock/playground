"""Tests for research/scripts/residual_cea_attenuation.py (G9 numerical check; no model output)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "scripts" / "residual_cea_attenuation.py"


def _load():
    spec = importlib.util.spec_from_file_location("residual_cea_attenuation_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_projection_equals_frozen_remove_common():
    rca = _load()
    from scripts.a4_analyse import remove_common  # frozen (tag a4-freeze); imported, never edited

    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 7, 25))
    X -= X.mean(axis=1, keepdims=True)
    X[0] = 0.0  # zero matrix: remove_common returns it unchanged
    got = rca.remove_common_batch(X)
    for i in range(len(X)):
        assert np.allclose(got[i], remove_common(X[i]))


def test_truth_has_set_shares_and_residual_cosine():
    rca = _load()
    rng = np.random.default_rng(1)
    t = rca.truth(7, 25, 0.3, 0.6, 1.0, rng)
    for tau, unit in ((t["tau_s"], t["u_s"]), (t["tau_h"], t["u_h"])):
        assert np.allclose(tau.sum(axis=0), 0.0)  # zero column sums
        mean = tau.mean(axis=1)
        assert np.isclose(abs(mean @ unit) / np.linalg.norm(mean), 1.0)  # mean profile along the set unit
    common = np.outer(t["u_h"], t["u_h"] @ t["tau_h"])
    assert np.isclose(np.sum(common ** 2) / np.sum(t["tau_h"] ** 2), rca.HUMAN_COMMON_SHARE)
    res_s = rca.remove_common_batch(t["tau_s"][None])[0]
    res_h = rca.remove_common_batch(t["tau_h"][None])[0]
    assert np.isclose(rca.cosine_batch(res_s[None], res_h[None])[0], 0.6)


def test_full_correction_close_to_truth_in_one_cell():
    rca = _load()
    cell = rca.run_cell(7, 25, 0.3, 0.3, 0.85, 1.0, 400, np.random.default_rng(2))
    assert abs(cell["bias_full_correction_residual"]) < 0.02
    assert abs(cell["bias_full_correction_raw"]) < 0.02
    assert cell["share_r_s_residual_not_positive"] == 0.0
