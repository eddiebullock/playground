"""EU-Emotion adapter → common Item schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_v2.metrics import shannon_entropy
from paper_v2.schema import Item, actor_id_from_eu_filename, validate_item

ROOT = Path(__file__).resolve().parents[3]


def load_eu_emotion_items(
    *,
    manifest_path: Optional[Path] = None,
    human_path: Optional[Path] = None,
) -> List[Item]:
    manifest_path = manifest_path or (ROOT / "data" / "eu_emotions_full_manifest.json")
    human_path = human_path or (ROOT / "data" / "eu_emotions_human_entropy.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    lookup: Dict[str, Any] = human["trials"]
    items: List[Item] = []
    for trial in manifest["trials"]:
        tid = trial["trial_id"]
        if tid not in lookup:
            raise KeyError(f"manifest trial missing from human lookup: {tid}")
        h = lookup[tid]
        options = list(h["human_options"])
        dist = h["human_response_distribution"]
        n = int(h["n_respondents"])
        counts = [float(dist.get(opt, 0.0)) * n for opt in options]
        target_label = h["human_target_label"]
        try:
            target_idx = options.index(target_label)
        except ValueError as exc:
            raise ValueError(f"{tid}: target {target_label!r} not in options {options}") from exc
        item = validate_item(
            {
                "item_id": tid,
                "stimulus_path": trial["stimulus_path"],
                "options": options,
                "target_idx": target_idx,
                "human_counts": counts,
                "human_n": n,
                "category": trial.get("correct_label") or trial.get("label") or target_label,
                "metadata": {
                    "actor_id": actor_id_from_eu_filename(tid),
                    "human_entropy": h["human_entropy"],
                    "p_target": h["p_target"],
                    "modality": human.get("modality"),
                    "edition": trial.get("edition"),
                    "pack": trial.get("pack"),
                },
            }
        )
        items.append(item)
    return items


def item_human_entropy(item: Item) -> float:
    stored = (item.get("metadata") or {}).get("human_entropy")
    if stored is not None:
        return float(stored)
    n = float(item["human_n"])
    if n <= 0:
        return 0.0
    probs = [c / n for c in item["human_counts"]]
    return shannon_entropy(probs)
