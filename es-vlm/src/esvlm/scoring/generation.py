"""Generation fallback for API models only. Open-weight evals use log-probs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def generate_choice(
    client: Any,
    *,
    prompt: str,
    options: Sequence[str],
    media: Any | None = None,
    max_tokens: int = 32,
) -> str:
    """Thin wrapper. ``client`` is a Mac-side API object, never a cluster GPU."""

    raise NotImplementedError(
        "API generation runs on the Mac only. Wire a client in scripts/ "
        "and import study4 RMET API results rather than re-paying for them."
    )
