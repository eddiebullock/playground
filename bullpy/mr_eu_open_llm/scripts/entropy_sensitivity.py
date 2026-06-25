#!/usr/bin/env python3
"""
Post-hoc semantic-entropy robustness from saved eval JSON (no VLM re-run).

Recomputes trial-level entropy under alternate temperature tau and/or embedding model,
then reports rank stability vs the primary metric stored in the JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import (
    ENTROPY_COLLAPSE_INTENSITY,
    ENTROPY_EXCLUDE_LABELS,
    ENTROPY_LOG_BASE,
    ENTROPY_SENSITIVITY_EMBEDDING_MODELS,
    ENTROPY_SENSITIVITY_TEMPERATURES,
    ENTROPY_TEMPERATURE,
    ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
    EMBEDDING_MODEL,
    LOCAL_RESULTS_DIR,
    PROTOCOL_VERSION,
)
from scripts.semantic_entropy import (
    compute_entropy_bundle,
    load_or_compute_label_embeddings,
    prepare_entropy_label_pool,
    strip_boilerplate_response,
)
from scripts.study1_baselines import discover_all_canonical


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def recompute_trials(
    obj: Dict[str, Any],
    *,
    temperatures: Sequence[float],
    embedding_models: Sequence[str],
) -> Dict[str, Any]:
    trials = obj.get("trials", [])
    stored_primary = []
    variant_series: Dict[str, List[float]] = {}

    emotion_pool = obj.get("entropy_label_pool")
    if not emotion_pool:
        # Reconstruct from fine label list in first trial or default file
        from scripts.trial_foils import resolve_eu_emotion_pool

        emotion_pool = resolve_eu_emotion_pool()
    entropy_labels = prepare_entropy_label_pool(emotion_pool, exclude=ENTROPY_EXCLUDE_LABELS)

    emb_cache: Dict[str, np.ndarray] = {}
    for model_name in embedding_models:
        emb_cache[model_name] = load_or_compute_label_embeddings(
            entropy_labels,
            model_name,
            rich_prompts=ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
        )

    for t in trials:
        s1 = t.get("stage1") or {}
        text = strip_boilerplate_response(s1.get("free_response_text") or "")
        stored = s1.get("semantic_entropy")
        stored_primary.append(float(stored) if stored is not None and stored == stored else np.nan)

        for model_name in embedding_models:
            for tau in temperatures:
                key = f"{model_name}|tau={tau}"
                bundle = compute_entropy_bundle(
                    text,
                    entropy_labels,
                    true_label=t.get("label"),
                    label_embeddings=emb_cache[model_name],
                    model_name=model_name,
                    temperature=tau,
                    log_base=ENTROPY_LOG_BASE,
                    rich_prompts=ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
                    collapse_intensity=ENTROPY_COLLAPSE_INTENSITY,
                )
                h = bundle.get("semantic_entropy")
                variant_series.setdefault(key, []).append(
                    float(h) if h is not None and h == h else np.nan
                )

    stored_arr = np.asarray(stored_primary, dtype=np.float64)
    correlations: Dict[str, Optional[float]] = {}
    mean_deltas: Dict[str, Optional[float]] = {}
    for key, vals in variant_series.items():
        arr = np.asarray(vals, dtype=np.float64)
        mask = np.isfinite(stored_arr) & np.isfinite(arr)
        if mask.sum() >= 3:
            correlations[key] = _spearman(stored_arr[mask], arr[mask])
            mean_deltas[key] = float(np.mean(arr[mask] - stored_arr[mask]))
        else:
            correlations[key] = None
            mean_deltas[key] = None

    return {
        "model": obj.get("model"),
        "path": obj.get("_path"),
        "n_trials": len(trials),
        "primary_embedding_model": obj.get("embedding_model", EMBEDDING_MODEL),
        "primary_temperature": obj.get("entropy_temperature", ENTROPY_TEMPERATURE),
        "n_entropy_labels": len(entropy_labels),
        "variant_correlations_spearman_vs_stored": correlations,
        "variant_mean_delta_vs_stored": mean_deltas,
        "temperatures": list(temperatures),
        "embedding_models": list(embedding_models),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic entropy sensitivity from eval JSON.")
    ap.add_argument("--results", type=Path, default=None, help="Single eval JSON (default: all canonical).")
    ap.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS_DIR / "stats" / "entropy_sensitivity.json",
    )
    ap.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=list(ENTROPY_SENSITIVITY_TEMPERATURES),
    )
    ap.add_argument(
        "--embedding-models",
        nargs="*",
        default=list(ENTROPY_SENSITIVITY_EMBEDDING_MODELS),
    )
    args = ap.parse_args()

    paths: List[Path]
    if args.results is not None:
        paths = [args.results]
    else:
        paths = list(discover_all_canonical().values())

    reports: List[Dict[str, Any]] = []
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("protocol_version") != PROTOCOL_VERSION:
            continue
        obj["_path"] = str(path)
        reports.append(
            recompute_trials(
                obj,
                temperatures=args.temperatures,
                embedding_models=args.embedding_models,
            )
        )

    out = {
        "protocol_version": PROTOCOL_VERSION,
        "description": (
            "Spearman correlation between stored primary semantic_entropy and recomputed "
            "values under alternate tau or embedding model. High rho => conclusions robust."
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
