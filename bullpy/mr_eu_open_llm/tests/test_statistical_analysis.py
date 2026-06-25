import numpy as np
import pytest

from scripts.statistical_analysis import (
    binomial_test_vs_chance,
    cohen_h,
    two_proportion_z_test,
    wilson_ci,
)


def test_wilson_ci():
    lo, hi = wilson_ci(35, 118)
    assert lo < 0.35 < hi


def test_cohen_h_math():
    h = cohen_h(0.5, 0.25)
    assert h > 0


def test_two_proportion_z_test():
    z, p = two_proportion_z_test(0.35, 118, 0.40, 100)
    assert isinstance(z, float) and isinstance(p, float)


def test_binomial_vs_chance():
    p = binomial_test_vs_chance(50, 118, 0.25)
    assert p < 0.001
