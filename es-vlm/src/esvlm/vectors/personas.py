"""Persona-vector-style pipeline for a simulated E/S population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from esvlm.vectors.extract import difference_of_means

# Mundane identity slots. Not EQ/SQ-R items and not E/S-loaded prompts.
_ROLES = (
    "a primary-school teacher",
    "a night-shift nurse",
    "a bus driver",
    "a baker",
    "a librarian",
    "a carpenter",
    "a hotel receptionist",
    "a gardener",
    "a shop assistant",
    "a postal worker",
    "a cook in a works canteen",
    "a house painter",
    "a plumber's mate",
    "a hotel porter",
    "a warehouse picker",
    "a swimming coach",
    "a museum attendant",
    "a farm worker",
    "a taxi driver",
    "a barista",
    "a tailor",
    "a dentist's receptionist",
    "a warehouse clerk",
    "a groundskeeper",
    "a bookshop owner",
)

_PLACES = (
    "a small market town",
    "a coastal city",
    "a hillside village",
    "a river suburb",
    "a new-build estate",
    "an old terrace street",
    "a harbour quarter",
    "a university district",
    "the industrial edge of town",
    "a square with a weekly market",
)

_ROUTINES = (
    "I keep a short list of errands and do them after work.",
    "I eat the same breakfast on most weekdays.",
    "I walk to the shop on Tuesdays and carry the bags home.",
    "I leave a spare key with a neighbour and collect the post myself.",
    "I wash the work clothes on Sunday and hang them in the kitchen.",
    "I take the early bus when I can and the later one when I cannot.",
    "I write appointments on a card on the fridge.",
    "I buy bread from the same counter and pay with coins when I have them.",
    "I keep the radio on in the kitchen while I make tea.",
    "I put the chairs back when I leave a room.",
    "I check the weather once in the morning and dress for that.",
    "I return library books on the day they are due if I remember.",
)

_OPENERS = (
    "I am {role} in {place}.",
    "People around here know me as {role} from {place}.",
)


@dataclass(frozen=True)
class PersonaScores:
    e: np.ndarray
    s: np.ndarray

    @property
    def correlation(self) -> float:
        if len(self.e) < 3:
            return float("nan")
        return float(np.corrcoef(self.e, self.s)[0, 1])


def persona_texts(n: int = 500, *, seed: int = 0) -> list[str]:
    """Combinatorial mundane first-person sketches. Not questionnaire items."""

    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    limit = n * 50
    while len(out) < n and attempts < limit:
        attempts += 1
        role = str(rng.choice(_ROLES))
        place = str(rng.choice(_PLACES))
        routine = str(rng.choice(_ROUTINES))
        opener = str(rng.choice(_OPENERS))
        text = f"{opener.format(role=role, place=place)} {routine}"
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    if len(out) < n:
        raise RuntimeError(f"could only build {len(out)} unique persona texts")
    return out


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
