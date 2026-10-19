from __future__ import annotations

import numpy as np

from esvlm.vectors.extract import difference_of_means, directions_for_layers
from esvlm.vectors.validate import held_out_auroc


def test_task_stacks_form_a_direction() -> None:
    rng = np.random.default_rng(0)
    social = rng.normal(size=(20, 3, 8)) + np.array([2.0] + [0.0] * 7)
    mech = rng.normal(size=(24, 3, 8)) - np.array([2.0] + [0.0] * 7)
    vecs = directions_for_layers(social, mech)
    assert vecs.shape == (3, 8)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    held = held_out_auroc(social[:, 1, :], mech[:, 1, :], seed=0)
    assert held > 0.8
    flipped = difference_of_means(mech[:, 1, :], social[:, 1, :])
    assert float(vecs[1] @ flipped) < 0
