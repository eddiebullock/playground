from __future__ import annotations

import numpy as np
import pytest

from esvlm.cli.steer import resolve_direction
from esvlm.steering.sweep import expand_grid, shard_path


def test_expand_grid_add_crosses_alphas_ablate_does_not() -> None:
    add = expand_grid([27], [-2.0, 0.0, 2.0], modes=("add",), vectors=("E", "S"))
    assert len(add) == 6
    ablate = expand_grid([27], [-2.0, 0.0, 2.0], modes=("ablate",), vectors=("E", "S"))
    assert len(ablate) == 2
    assert {c.alpha for c in ablate} == {0.0}
    both = expand_grid([27], [-1.0, 1.0], modes=("add", "ablate"), vectors=("E",))
    assert len(both) == 3


def test_shard_path_and_resolve_direction(tmp_path) -> None:
    cells = expand_grid([27], [1.0], vectors=("E",))
    path = shard_path(tmp_path, "qwen3vl_8b", "rmet", cells[0], "0")
    assert path.name.startswith("rmet__qwen3vl_8b__")
    e = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    s = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vecs = {"E": e, "S": s, "E_minus_S": e - s}
    assert np.allclose(resolve_direction(vecs, "E"), e)
    uniq = resolve_direction(vecs, "E_unique")
    assert abs(float(uniq @ (s / np.linalg.norm(s)))) < 1e-6
    with pytest.raises(SystemExit):
        resolve_direction(vecs, "nope")
