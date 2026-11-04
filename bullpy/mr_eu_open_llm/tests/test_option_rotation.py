"""Option rotation and vision-mismatch controls for the deconfounded 6AFC."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_v2.option_score import (  # noqa: E402
    ABSTAIN_LABEL,
    rotate_options,
    vision_mismatch_partners,
    write_rotated_human_lookup,
)

OPTIONS = ["Target", "A", "B", "C", "D", ABSTAIN_LABEL]


def test_rotation_puts_target_in_each_named_position_once():
    positions = [rotate_options(OPTIONS, r).index("Target") for r in range(5)]
    assert sorted(positions) == [0, 1, 2, 3, 4]


def test_rotation_keeps_abstain_last_and_labels_unchanged():
    for r in range(5):
        rotated = rotate_options(OPTIONS, r)
        assert rotated[-1] == ABSTAIN_LABEL
        assert sorted(rotated) == sorted(OPTIONS)


def test_rotation_zero_is_identity():
    assert rotate_options(OPTIONS, 0) == OPTIONS


def test_rotated_lookup_balances_positions_on_real_norms(tmp_path):
    src = ROOT / "data" / "eu_emotions_human_entropy.json"
    counts = Counter()
    for r in range(5):
        dest = tmp_path / f"rot{r}.json"
        write_rotated_human_lookup(src, dest, rotation=r)
        trials = json.loads(dest.read_text(encoding="utf-8"))["trials"]
        for rec in trials.values():
            counts[rec["human_options"].index(rec["human_target_label"])] += 1
    assert counts == {0: 243, 1: 243, 2: 243, 3: 243, 4: 243}


def test_mismatch_differs_in_label_and_actor_and_is_deterministic():
    ids = [f"t{i}" for i in range(40)]
    labels = [f"L{i % 5}" for i in range(40)]
    actors = [f"a{i % 7}" for i in range(40)]
    first = vision_mismatch_partners(ids, labels, actors, seed=42)
    assert first == vision_mismatch_partners(ids, labels, actors, seed=42)
    label_of = dict(zip(ids, labels))
    actor_of = dict(zip(ids, actors))
    for tid, partner in first.items():
        assert partner != tid
        assert label_of[partner] != label_of[tid]
        assert actor_of[partner] != actor_of[tid]


def test_mismatch_reuses_videos_evenly():
    ids = [f"t{i}" for i in range(40)]
    labels = [f"L{i % 5}" for i in range(40)]
    actors = [f"a{i % 7}" for i in range(40)]
    uses = Counter(vision_mismatch_partners(ids, labels, actors, seed=42).values())
    assert max(uses.values()) <= 2
