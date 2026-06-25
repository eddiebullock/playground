from __future__ import annotations

"""Deterministic 4-AFC foil generation (sha256(trial_id|seed))."""

import hashlib
import random
from pathlib import Path
from typing import List, Optional, Sequence


def _trial_rng(trial_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{trial_id}|{seed}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def load_eu_emotion_label_pool(labels_file: Path) -> List[str]:
    if not labels_file.is_file():
        raise FileNotFoundError(f"EU emotion label list not found: {labels_file}")
    labels = [ln.strip() for ln in labels_file.read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    if len(labels) < 4:
        raise ValueError(f"EU emotion label pool too small ({len(labels)}): {labels_file}")
    return labels


def build_emotion_pool_from_trials(trials: Sequence[dict]) -> List[str]:
    pool = set()
    for t in trials:
        for key in ("correct_label", "emotion", "label"):
            if t.get(key):
                pool.add(str(t[key]).strip())
    return sorted(pool, key=str.casefold)


_REPO_EU_LABELS_FILE = Path(__file__).resolve().parents[1] / "data" / "eu_emotion_states_list.txt"

# Canonical EU-Emotions states (used when label list file is not on disk yet).
_BUILTIN_EU_EMOTION_LABELS = [
    "afraid",
    "afraid low intensity",
    "angry",
    "angry low intensity",
    "ashamed",
    "bored",
    "disappointed",
    "disgusted",
    "disgusted low intensity",
    "excited",
    "frustrated",
    "happy",
    "happy low intensity",
    "hurt",
    "interested",
    "jealous",
    "joking",
    "kind",
    "neutral",
    "proud",
    "sad",
    "sad low intensity",
    "sneaky",
    "surprised",
    "surprised low intensity",
    "unfriendly",
    "worried",
]


def resolve_eu_emotion_pool(
    *,
    label_paths: Optional[Sequence[Optional[Path]]] = None,
    trials_fallback: Optional[Sequence[dict]] = None,
) -> List[str]:
    """Full EU emotion label pool for 4-AFC foils (independent of --max_trials slice)."""
    checked: set[str] = set()
    for raw in label_paths or ():
        if raw is None:
            continue
        path = Path(raw)
        key = str(path)
        if key in checked:
            continue
        checked.add(key)
        if path.is_file():
            return load_eu_emotion_label_pool(path)
    if _REPO_EU_LABELS_FILE.is_file():
        return load_eu_emotion_label_pool(_REPO_EU_LABELS_FILE)
    if trials_fallback:
        pool = build_emotion_pool_from_trials(trials_fallback)
        if len(pool) >= 4:
            return pool
    return list(_BUILTIN_EU_EMOTION_LABELS)


def generate_candidate_labels(
    correct_label: str, emotion_pool: Sequence[str], *, trial_id: str, seed: int, n_options: int = 4
) -> List[str]:
    correct = correct_label.strip()
    pool = sorted({str(e).strip() for e in emotion_pool if str(e).strip()}, key=str.casefold)
    if not any(e.casefold() == correct.casefold() for e in pool):
        pool = sorted(set(pool) | {correct}, key=str.casefold)
    others = [e for e in pool if e.casefold() != correct.casefold()]
    n_foils = n_options - 1
    if len(others) < n_foils:
        raise ValueError(f"Not enough foils for trial_id={trial_id!r}")
    rng = _trial_rng(str(trial_id), seed)
    labels = [correct] + rng.sample(others, n_foils)
    rng.shuffle(labels)
    return labels


def resolve_candidate_labels(
    trial: dict, emotion_pool: Sequence[str], *, seed: int, trial_index: Optional[int] = None
) -> List[str]:
    existing = trial.get("candidate_labels")
    if isinstance(existing, list) and len(existing) == 4:
        return list(existing)
    correct_label = trial.get("correct_label") or trial.get("emotion") or trial.get("label")
    if not correct_label:
        raise ValueError(f"Trial missing label (trial_id={trial.get('trial_id')!r})")
    trial_id = trial.get("trial_id") or f"trial_{trial_index if trial_index is not None else 0}"
    generated = generate_candidate_labels(str(correct_label), emotion_pool, trial_id=str(trial_id), seed=seed)
    trial["candidate_labels"] = generated
    trial["candidate_labels_generated"] = True
    return generated
