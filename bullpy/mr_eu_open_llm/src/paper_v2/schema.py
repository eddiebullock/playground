"""Common item schema for dataset adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, TypedDict


class Item(TypedDict, total=False):
    item_id: str
    stimulus_path: str
    options: List[str]
    target_idx: int
    human_counts: List[float]
    human_n: int
    category: str
    metadata: Dict[str, Any]


REQUIRED_FIELDS = (
    "item_id",
    "stimulus_path",
    "options",
    "target_idx",
    "human_counts",
    "human_n",
    "category",
)


def validate_item(item: Mapping[str, Any]) -> Item:
    missing = [f for f in REQUIRED_FIELDS if f not in item]
    if missing:
        raise ValueError(f"item missing fields {missing}: {item.get('item_id')}")
    options = list(item["options"])
    counts = list(item["human_counts"])
    if len(options) != len(counts):
        raise ValueError(
            f"{item['item_id']}: len(options)={len(options)} != len(human_counts)={len(counts)}"
        )
    target_idx = int(item["target_idx"])
    if not (0 <= target_idx < len(options)):
        raise ValueError(f"{item['item_id']}: target_idx {target_idx} out of range")
    out: Item = {
        "item_id": str(item["item_id"]),
        "stimulus_path": str(item["stimulus_path"]),
        "options": options,
        "target_idx": target_idx,
        "human_counts": [float(c) for c in counts],
        "human_n": int(item["human_n"]),
        "category": str(item["category"]),
        "metadata": dict(item.get("metadata") or {}),
    }
    return out


def actor_id_from_eu_filename(item_id: str) -> Optional[str]:
    """EU-Emotion filenames encode actor as the letter before F, e.g. NF25 -> N."""
    import re

    match = re.search(r"([A-Za-z])F\s*\d+", item_id, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None
