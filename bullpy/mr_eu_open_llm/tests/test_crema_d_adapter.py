from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from paper_v2.datasets.crema_d import (  # noqa: E402
    OPTION_LABELS,
    actor_groups,
    load_crema_d_items,
)
from paper_v2.metrics import shannon_entropy  # noqa: E402


def test_adapter_loads_selection_with_actor_metadata():
    items = load_crema_d_items()
    assert len(items) == 819
    assert len({it["item_id"] for it in items}) == 819
    assert len(set(actor_groups(items))) == 91
    first = items[0]
    assert first["metadata"]["actor_id"] == first["item_id"].split("_")[0]
    assert first["metadata"]["modality"] == "Face"
    assert first["stimulus_path"] == f"crema_d/{first['item_id']}.flv"


def test_every_clip_offers_the_same_six_options():
    for item in load_crema_d_items():
        assert sorted(item["options"]) == sorted(OPTION_LABELS)
    assert "No Emotion" in OPTION_LABELS and "Neutral" not in OPTION_LABELS


def test_target_is_the_intended_emotion_and_position_varies():
    items = load_crema_d_items()
    code = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "No Emotion", "SAD": "Sad"}
    for item in items:
        assert item["options"][item["target_idx"]] == code[item["item_id"].split("_")[2]]
    positions = Counter(item["target_idx"] for item in items)
    assert set(positions) == set(range(6))
    assert max(positions.values()) - min(positions.values()) <= 1


def test_six_rotations_put_each_target_in_each_position_once():
    seen = {}
    for rotation in range(6):
        for item in load_crema_d_items(rotation=rotation):
            seen.setdefault(item["item_id"], []).append(item["target_idx"])
    assert all(sorted(v) == list(range(6)) for v in seen.values())


def test_rotation_keeps_counts_attached_to_their_labels():
    base = load_crema_d_items()
    for rotation in range(1, 6):
        for a, b in zip(base, load_crema_d_items(rotation=rotation)):
            assert dict(zip(a["options"], a["human_counts"])) == dict(zip(b["options"], b["human_counts"]))
            assert a["metadata"]["p_target"] == b["metadata"]["p_target"]


def test_human_summaries_match_counts():
    for item in load_crema_d_items():
        n = item["human_n"]
        assert sum(item["human_counts"]) == n
        assert abs(item["metadata"]["p_target"] - item["human_counts"][item["target_idx"]] / n) < 1e-12
        probs = [c / n for c in item["human_counts"]]
        assert abs(item["metadata"]["human_entropy"] - shannon_entropy(probs)) < 1e-12


def test_actor_groups_keep_actors_out_of_both_sides_of_a_fold():
    items = load_crema_d_items()
    groups = np.array(actor_groups(items))
    for train, test in GroupKFold(n_splits=5).split(np.zeros(len(items)), groups=groups):
        assert not set(groups[train]) & set(groups[test])
