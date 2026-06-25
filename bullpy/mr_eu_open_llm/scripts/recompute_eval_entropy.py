#!/usr/bin/env python3
"""
Recompute Stage 1 semantic entropy in saved eval JSONs (CPU only).

Use after updating scripts/semantic_entropy.py (rich labels, no neutral, base collapse).
Does not re-run VLMs; Stage 2 accuracy and predictions are unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import (
    ENTROPY_COLLAPSE_INTENSITY,
    ENTROPY_EXCLUDE_LABELS,
    ENTROPY_LOG_BASE,
    ENTROPY_TEMPERATURE,
    ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
    EMBEDDING_MODEL,
    LOCAL_RESULTS_DIR,
    PROTOCOL_VERSION,
)
from scripts.semantic_entropy import (
    _load_sentence_transformer,
    compute_entropy_bundle,
    load_or_compute_label_embeddings,
    prepare_entropy_label_pool,
    strip_boilerplate_response,
)
from scripts.study1_baselines import discover_all_canonical
from scripts.trial_foils import resolve_eu_emotion_pool


def recompute_eval_json(
    obj: Dict[str, Any],
    *,
    label_embeddings: np.ndarray,
    entropy_labels: List[str],
    st_model=None,
) -> Dict[str, Any]:
    entropies: List[float] = []
    entropies_fine: List[float] = []
    entropies_base: List[float] = []
    trials = obj.get("trials", [])
    n_trials = len(trials)

    for i, t in enumerate(trials):
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == n_trials:
            print(f"  trial {i + 1}/{n_trials}", flush=True)
        s1 = t.get("stage1") or {}
        text = strip_boilerplate_response(s1.get("free_response_text") or "")
        bundle = compute_entropy_bundle(
            text,
            entropy_labels,
            true_label=t.get("label"),
            label_embeddings=label_embeddings,
            model_name=EMBEDDING_MODEL,
            temperature=ENTROPY_TEMPERATURE,
            log_base=ENTROPY_LOG_BASE,
            rich_prompts=ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
            collapse_intensity=ENTROPY_COLLAPSE_INTENSITY,
            st_model=st_model,
        )
        if not s1:
            s1 = {}
            t["stage1"] = s1
        s1["semantic_entropy"] = bundle.get("semantic_entropy")
        s1["semantic_entropy_fine"] = bundle.get("semantic_entropy_fine")
        s1["semantic_entropy_base"] = bundle.get("semantic_entropy_base")
        s1["label_probs"] = bundle.get("label_probs")
        s1["base_label_probs"] = bundle.get("base_label_probs")
        s1["base_labels"] = bundle.get("base_labels")
        s1["top_labels"] = bundle.get("top_labels")
        s1["p_correct"] = bundle.get("p_correct")
        s1["margin_correct"] = bundle.get("margin_correct")
        s1["correct_in_entropy_pool"] = bundle.get("correct_in_entropy_pool")
        s1["n_entropy_labels"] = bundle.get("n_entropy_labels")
        s1["embedding_model"] = EMBEDDING_MODEL
        s1["entropy_temperature"] = ENTROPY_TEMPERATURE
        s1["entropy_rich_label_embeddings"] = ENTROPY_USE_RICH_LABEL_EMBEDDINGS
        s1["entropy_exclude_labels"] = list(ENTROPY_EXCLUDE_LABELS)
        s1["entropy_collapse_intensity"] = ENTROPY_COLLAPSE_INTENSITY

        h = s1["semantic_entropy"]
        if h is not None and h == h:
            entropies.append(float(h))
        hf = s1["semantic_entropy_fine"]
        if hf is not None and hf == hf:
            entropies_fine.append(float(hf))
        hb = s1["semantic_entropy_base"]
        if hb is not None and hb == hb:
            entropies_base.append(float(hb))

    def _agg(xs: List[float]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"mean": None, "median": None, "std": None}
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    primary = _agg(entropies)
    obj["mean_semantic_entropy"] = primary["mean"]
    obj["median_semantic_entropy"] = primary["median"]
    obj["std_semantic_entropy"] = primary["std"]
    obj["mean_semantic_entropy_fine"] = _agg(entropies_fine)["mean"]
    obj["mean_semantic_entropy_base"] = _agg(entropies_base)["mean"]
    obj["embedding_model"] = EMBEDDING_MODEL
    obj["entropy_temperature"] = ENTROPY_TEMPERATURE
    obj["entropy_log_base"] = ENTROPY_LOG_BASE
    obj["entropy_exclude_labels"] = list(ENTROPY_EXCLUDE_LABELS)
    obj["entropy_rich_label_embeddings"] = ENTROPY_USE_RICH_LABEL_EMBEDDINGS
    obj["entropy_collapse_intensity"] = ENTROPY_COLLAPSE_INTENSITY
    obj["entropy_label_pool"] = entropy_labels
    obj["entropy_recomputed_at_utc"] = datetime.now(timezone.utc).isoformat()
    obj["entropy_definition"] = (
        "primary semantic_entropy = H over base emotions after softmax on "
        f"{len(entropy_labels)} fine labels (neutral excluded), rich label prompts, "
        "intensity collapsed"
    )
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute semantic entropy in eval JSONs (CPU).")
    ap.add_argument("--results", type=Path, nargs="*", default=None, help="Eval JSON(s); default canonical baselines.")
    ap.add_argument("--backup", action="store_true", help="Write .bak copy before overwriting.")
    ap.add_argument("--dry-run", action="store_true", help="Print paths only; do not write.")
    args = ap.parse_args()

    paths: List[Path]
    if args.results:
        paths = list(args.results)
    else:
        paths = list(discover_all_canonical().values())

    if not paths:
        raise RuntimeError("No eval JSONs found.")

    emotion_pool = resolve_eu_emotion_pool()
    entropy_labels = prepare_entropy_label_pool(emotion_pool, exclude=ENTROPY_EXCLUDE_LABELS)
    print(f"Loading embedding model {EMBEDDING_MODEL} (first run may download to HF_HOME)...", flush=True)
    st_model = _load_sentence_transformer(EMBEDDING_MODEL)
    print(f"Computing label embeddings for {len(entropy_labels)} labels...", flush=True)
    label_embeddings = load_or_compute_label_embeddings(
        entropy_labels,
        EMBEDDING_MODEL,
        rich_prompts=ENTROPY_USE_RICH_LABEL_EMBEDDINGS,
    )
    print(f"Found {len(paths)} eval JSON(s) to update.", flush=True)

    for path in paths:
        print(f"Processing {path}...", flush=True)
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Not protocol v2: {path}")
        updated = recompute_eval_json(
            obj,
            label_embeddings=label_embeddings,
            entropy_labels=entropy_labels,
            st_model=st_model,
        )
        if args.dry_run:
            print(f"would update: {path}  mean_H={updated.get('mean_semantic_entropy')}")
            continue
        if args.backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        print(
            f"updated {path.name}: mean_semantic_entropy={updated.get('mean_semantic_entropy'):.4f} "
            f"(fine={updated.get('mean_semantic_entropy_fine'):.4f}, "
            f"base={updated.get('mean_semantic_entropy_base'):.4f})"
        )


if __name__ == "__main__":
    main()
