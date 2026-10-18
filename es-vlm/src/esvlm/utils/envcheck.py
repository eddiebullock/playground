"""Refuse misspelled / lowercase variants of known job env vars."""

from __future__ import annotations

import os

KNOWN = (
    "MODEL",
    "MODE",
    "LAYER",
    "LAYERS",
    "ALPHA",
    "ALPHAS",
    "FRAMES",
    "N_SAMPLES",
    "SHARD",
    "TASK",
    "VECTOR",
    "TRIAL_OFFSET",
    "TRIAL_LIMIT",
)


def check_job_env(environ: dict[str, str] | None = None) -> None:
    """Exit-worthy error if a known var is set only in a wrong case.

    A previous job silently ran the smoke path because ``MODE`` was ignored.
    """

    env = environ if environ is not None else dict(os.environ)
    present = {k for k in env if k in KNOWN}
    mistakes = []
    for name in KNOWN:
        lower = name.lower()
        if lower in env and name not in present:
            mistakes.append(f"{lower}={env[lower]!r} (did you mean {name}?)")
        title = name.title()
        if title in env and title not in KNOWN and name not in present:
            mistakes.append(f"{title}={env[title]!r} (did you mean {name}?)")
    if mistakes:
        joined = "; ".join(mistakes)
        raise SystemExit(f"refusing misspelled job env vars: {joined}")
