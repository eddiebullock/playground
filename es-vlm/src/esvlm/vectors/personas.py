"""Persona-vector-style pipeline for a simulated E/S population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from esvlm.vectors.extract import difference_of_means


@dataclass(frozen=True)
class PersonaScores:
    e: np.ndarray
    s: np.ndarray

    @property
    def correlation(self) -> float:
        if len(self.e) < 3:
            return float("nan")
        return float(np.corrcoef(self.e, self.s)[0, 1])


def persona_matrix(
    activations: np.ndarray,
    e_dir: np.ndarray,
    s_dir: np.ndarray,
) -> PersonaScores:
    """``activations`` is [n_personas, D] final-token residuals."""

    e = activations @ (e_dir / np.linalg.norm(e_dir))
    s = activations @ (s_dir / np.linalg.norm(s_dir))
    return PersonaScores(e=e, s=s)


def resample_direction(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap difference-of-means directions for uncertainty."""

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = pos[rng.integers(0, len(pos), size=len(pos))]
        q = neg[rng.integers(0, len(neg), size=len(neg))]
        out.append(difference_of_means(p, q))
    return np.stack(out, axis=0)
