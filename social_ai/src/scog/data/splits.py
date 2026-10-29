"""Deterministic stratified splits. Membership lists are hashed, never committed.

Channel 4 holds out 20% exploratory, then 10% of the full sample as the
Study A2 test set from inside the remaining 80%, leaving 70% confirmatory.
Other participant datasets are 20/80. Strata smaller than 5 rows all go to
the confirmatory portion so a singleton cannot land in the exploratory set
by rounding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_SEED = 20261029
EXPLORE_FRAC = 0.20
A2_FRAC = 0.10
MIN_STRATUM = 5

AGE_BINS = (18, 30, 45, 60, 90)


def age_band(age: float) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "age_missing"
    if age < 18:
        return "under_18"
    if age < 30:
        return "18_29"
    if age < 45:
        return "30_44"
    if age < 60:
        return "45_59"
    if age <= 90:
        return "60_90"
    return "age_missing"


def stratum_labels(group: pd.Series, sex: pd.Series, age: pd.Series) -> pd.Series:
    bands = age.map(lambda v: age_band(float(v)) if pd.notna(v) else "age_missing")
    sex_s = sex.fillna("sex_missing").astype(str)
    group_s = group.fillna("group_missing").astype(str)
    return group_s + "|" + sex_s + "|" + bands


def assign_splits(
    ids: pd.Series,
    strata: pd.Series,
    *,
    seed: int = SPLIT_SEED,
    with_a2: bool = False,
) -> pd.DataFrame:
    """Return one row per id with a `split` column. Ids must be unique."""
    if ids.duplicated().any():
        raise RuntimeError("split ids are not unique")
    if len(ids) == 0:
        raise RuntimeError("split: zero rows")
    frame = pd.DataFrame({"id": ids.astype(str).to_numpy(), "stratum": strata.astype(str).to_numpy()})
    rng = np.random.default_rng(seed)
    split = np.empty(len(frame), dtype=object)
    split[:] = None
    small = 0
    for stratum, index in frame.groupby("stratum").groups.items():
        positions = np.array(list(index))
        order = rng.permutation(len(positions))
        positions = positions[order]
        n = len(positions)
        if n < MIN_STRATUM:
            split[positions] = "confirmatory"
            small += n
            continue
        n_explore = int(np.floor(n * EXPLORE_FRAC))
        n_a2 = int(np.floor(n * A2_FRAC)) if with_a2 else 0
        if n_explore + n_a2 >= n:
            n_a2 = max(0, n - n_explore - 1)
        split[positions[:n_explore]] = "exploratory"
        split[positions[n_explore:n_explore + n_a2]] = "a2_test"
        split[positions[n_explore + n_a2:]] = "confirmatory"
        del stratum
    if any(label is None for label in split):
        raise RuntimeError("split assignment left a row without a label")
    frame["split"] = split
    frame.attrs["n_small_stratum_forced_confirmatory"] = small
    return frame


def hash_ids(ids: list[str]) -> str:
    payload = ("\n".join(sorted(str(i) for i in ids)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_id_list(path: Path, ids: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(str(i) for i in ids)
    path.write_text("\n".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
    digest = hash_ids(ordered)
    check = hashlib.sha256(path.read_bytes()).hexdigest()
    if check != digest:
        raise RuntimeError(f"split file hash mismatch for {path.name}")
    return digest


def lock_lines(digests: dict[str, str]) -> str:
    lines = [f"{name} {digest}" for name, digest in sorted(digests.items())]
    return "\n".join(lines) + "\n"


def assert_split_allowed(split: str) -> None:
    """Confirmatory rows stay unread until a preregistration is frozen."""
    if split != "confirmatory":
        return
    frozen = Path(__file__).resolve().parents[3] / "prereg" / "FROZEN"
    if not frozen.is_file():
        raise RuntimeError(
            "confirmatory split is locked until prereg/FROZEN exists"
        )


def read_lock(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, digest = line.split()
        if len(digest) != 64:
            raise RuntimeError(f"split lock entry {name} is not a sha256 hex digest")
        out[name] = digest
    if not out:
        raise RuntimeError("split lock is empty")
    return out
