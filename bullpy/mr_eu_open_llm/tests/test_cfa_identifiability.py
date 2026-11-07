import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cfa", ROOT / "research/scripts/cfa_identifiability.py")
cfa = importlib.util.module_from_spec(spec)
sys.modules["cfa"] = cfa
spec.loader.exec_module(cfa)


def test_any_correlation_is_reachable_with_behaviour_fixed():
    for row in cfa.checks()["theorem_1"]:
        assert row["r_achieved"] == pytest.approx(row["r_requested"], abs=1e-6)
        assert row["max_entry_error"] < 1e-12


def test_infeasible_target_is_refused():
    f = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        cfa.theorem_1_system(f, np.array([[0.9, -0.9]]))


def test_proposition_3_bound_is_attained():
    tight = cfa.checks()["proposition_3_tight_case"]
    assert tight["ratio"] == pytest.approx(tight["one_over_lambda_min"], abs=1e-5)


def test_two_cue_case_matches_simulation():
    rng = np.random.default_rng(0)
    rho = 0.9
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=400000)
    case = cfa.two_cue_case(rho)
    assert np.corrcoef(z[:, 0], z[:, 1])[0, 1] == pytest.approx(case["output_correlation"], abs=0.01)
    assert ((z[:, 0] - z[:, 1])**2).mean() == pytest.approx(case["mean_squared_difference"], abs=0.01)


def test_spectrum_of_identity_design():
    table = pd.DataFrame(np.random.default_rng(1).normal(size=(5000, 3)))
    assert cfa.spectrum(table)["lambda_min"] == pytest.approx(1, abs=0.1)
