"""Option-order permutations.

Four options: all 24 orders. Six options: one cyclic Latin square (6 rows)
plus extra random orders from a seed, with duplicates removed.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def all_orders(options: Sequence[str]) -> list[tuple[str, ...]]:
    if len(options) != 4:
        raise ValueError("all_orders is the 24 permutations of 4 options")
    return list(itertools.permutations(tuple(options)))


def latin_square(n: int = 6) -> list[tuple[int, ...]]:
    """Cyclic Latin square. Row i, column j is (i + j) mod n."""
    if n < 1:
        raise ValueError("Latin square order must be positive")
    return [tuple((i + j) % n for j in range(n)) for i in range(n)]


def orders_for_six(
    options: Sequence[str],
    *,
    n_random: int = 6,
    seed: int = 20261029,
) -> list[tuple[str, ...]]:
    if len(options) != 6:
        raise ValueError("orders_for_six expects 6 options")
    base = tuple(options)
    rows = [tuple(base[i] for i in row) for row in latin_square(6)]
    seen = set(rows)
    rng = np.random.default_rng(seed)
    extra: list[tuple[str, ...]] = []
    guard = 0
    while len(extra) < n_random and guard < 10000:
        guard += 1
        order = tuple(rng.permutation(base))
        if order not in seen:
            seen.add(order)
            extra.append(order)
    if len(extra) < n_random:
        raise RuntimeError("could not draw enough distinct random orders")
    return rows + extra
