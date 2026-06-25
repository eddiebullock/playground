"""Load EU-Emotions trials as Inspect AI Samples (provider-agnostic metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATASETS, SEED
from scripts.evaluate import load_eu_emotions_manifest, resolve_dataset_root
from scripts.trial_foils import resolve_candidate_labels, resolve_eu_emotion_pool


def _trial_to_sample(trial: Dict[str, Any], *, seed: int, emotion_pool: List[str], idx: int) -> Dict[str, Any]:
    label = trial.get("label") or trial.get("correct_label")
    opts = resolve_candidate_labels(dict(trial), emotion_pool, seed=seed, trial_index=idx)
    return {
        "id": trial.get("trial_id"),
        "input": trial.get("trial_id"),
        "target": label,
        "metadata": {
            "stimulus_path": trial.get("stimulus_path"),
            "label": label,
            "candidate_labels": opts,
            "trial_id": trial.get("trial_id"),
        },
    }


def load_eu_emotions_samples(
    *,
    manifest: Optional[Path] = None,
    data_root: Optional[Path] = None,
    max_trials: Optional[int] = None,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    dataset_root = data_root or resolve_dataset_root("eu_emotions")
    if manifest is None:
        manifest = Path(
            DATASETS["eu_emotions"].get(
                "manifest_local", dataset_root.parent / "eu_emotions_118_manifest.json"
            )
        )
    trials, _ = load_eu_emotions_manifest(manifest, dataset_root)
    if max_trials is not None:
        trials = trials[: int(max_trials)]
    pool = resolve_eu_emotion_pool()
    return [_trial_to_sample(t, seed=seed, emotion_pool=pool, idx=i) for i, t in enumerate(trials)]


def to_inspect_samples(samples: List[Dict[str, Any]]) -> List[Any]:
    """Convert dict samples to inspect_ai.dataset.Sample if inspect-ai is installed."""
    try:
        from inspect_ai.dataset import Sample

        out = []
        for s in samples:
            meta = s.get("metadata") or {}
            out.append(
                Sample(
                    id=str(s.get("id")),
                    input=s.get("input"),
                    target=s.get("target"),
                    metadata=meta,
                )
            )
        return out
    except ImportError:
        return samples
