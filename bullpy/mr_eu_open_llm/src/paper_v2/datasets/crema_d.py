"""CREMA-D face-only adapter -> common Item schema.

Reads the committed selection table, so it needs neither the raw ratings nor the videos.
See research/CREMA_D_PLAN.md for the selection rule and the option wording.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from paper_v2.metrics import shannon_entropy
from paper_v2.option_score import rotate_options
from paper_v2.schema import Item, validate_item

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION = ROOT / "research" / "reproduction" / "crema_d_selection.csv"

# Raters chose one of five emotions or "No Emotion" (Cao et al. 2014). The data codes
# that last choice N and the filenames call the condition NEU, so the wording raters
# saw and the category name deliberately differ.
OPTION_BY_CODE: Dict[str, str] = {
    "A": "Anger",
    "D": "Disgust",
    "F": "Fear",
    "H": "Happy",
    "N": "No Emotion",
    "S": "Sad",
}
CATEGORY_BY_CODE: Dict[str, str] = {**OPTION_BY_CODE, "N": "Neutral"}
OPTION_LABELS: List[str] = list(OPTION_BY_CODE.values())


def options_with_target_at(target_label: str, position: int) -> List[str]:
    """Cyclic rotation of the one fixed option set placing the target at `position`."""
    n = len(OPTION_LABELS)
    shift = (OPTION_LABELS.index(target_label) - position) % n
    return rotate_options(OPTION_LABELS, shift)


def load_crema_d_items(
    *,
    selection_path: Optional[Path] = None,
    rotation: int = 0,
) -> List[Item]:
    """Selected clips with target position rotated across clips.

    `rotation` adds to each clip's assigned position, so rotations 0..5 together put
    every target in every position once, as in amendment A1. `human_counts` follows the
    rotated option order, so index i always refers to the same option in both lists.
    """
    path = selection_path or DEFAULT_SELECTION
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    items: List[Item] = []
    for row in rows:
        code = row["emotion"]
        target_label = OPTION_BY_CODE[code]
        position = (int(row["target_position"]) + rotation) % len(OPTION_LABELS)
        options = options_with_target_at(target_label, position)
        votes = {OPTION_BY_CODE[c]: int(row[f"votes_{c}"]) for c in OPTION_BY_CODE}
        counts = [float(votes[opt]) for opt in options]
        n = int(row["n_face"])
        if sum(counts) != n:
            raise ValueError(f"{row['clip_id']}: votes sum to {sum(counts)}, n_face is {n}")
        items.append(
            validate_item(
                {
                    "item_id": row["clip_id"],
                    "stimulus_path": f"crema_d/{row['clip_id']}.flv",
                    "options": options,
                    "target_idx": position,
                    "human_counts": counts,
                    "human_n": n,
                    "category": CATEGORY_BY_CODE[code],
                    "metadata": {
                        "actor_id": row["actor_id"],
                        "human_entropy": shannon_entropy([c / n for c in counts]),
                        "p_target": votes[target_label] / n,
                        "modality": "Face",
                        "sentence": row["sentence"],
                        "level": row["level"],
                        "sex": row["sex"],
                        "age": int(row["age"]),
                        "race": row["race"],
                        "ethnicity": row["ethnicity"],
                        "rotation": rotation,
                    },
                }
            )
        )
    return items


def actor_groups(items: Sequence[Item]) -> List[str]:
    """Per-item actor ids, aligned with `items`, for grouped cross-validation."""
    return [item["metadata"]["actor_id"] for item in items]
