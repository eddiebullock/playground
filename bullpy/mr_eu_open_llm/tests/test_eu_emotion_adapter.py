from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_v2.behaviour import report_all, summarise_eval  # noqa: E402
from paper_v2.datasets.eu_emotion import load_eu_emotion_items  # noqa: E402
from paper_v2.metrics import shannon_entropy  # noqa: E402


def test_adapter_loads_243_items_with_actors():
    items = load_eu_emotion_items()
    assert len(items) == 243
    assert all(len(it["options"]) == 6 for it in items)
    actors = {it["metadata"]["actor_id"] for it in items}
    assert None not in actors
    assert len(actors) == 17
    assert items[0]["item_id"] == "emotions/EDITED/Unfriendly/NF25_cut.mp4"
    assert items[0]["metadata"]["actor_id"] == "N"
    assert items[0]["metadata"]["modality"] == "Face"


def test_adapter_entropy_matches_stored():
    items = load_eu_emotion_items()
    item = items[0]
    n = item["human_n"]
    probs = [c / n for c in item["human_counts"]]
    recomputed = shannon_entropy(probs)
    stored = item["metadata"]["human_entropy"]
    assert abs(recomputed - stored) < 1e-6


def test_behaviour_molmo_flagged():
    payload = report_all()
    molmo = payload["models"]["molmo2"]["accuracy"]
    qwen = payload["models"]["qwen3vl"]["accuracy"]
    assert molmo["n_correct"] == 48
    assert molmo["above_chance_alpha_05"] is False
    assert qwen["above_chance_alpha_05"] is True
    assert qwen["n_correct"] == 92
    qwen_ent = payload["models"]["qwen3vl"]["entropy_alignment"]
    gemma_ent = payload["models"]["gemma4"]["entropy_alignment"]
    molmo_ent = payload["models"]["molmo2"]["entropy_alignment"]
    assert qwen_ent["n"] == 243
    assert abs(qwen_ent["spearman_rho"] - (-0.0013253211000191954)) < 1e-9
    assert abs(gemma_ent["spearman_rho"] - 0.11724630621919127) < 1e-9
    assert abs(molmo_ent["spearman_rho"] - (-0.0344232402583643)) < 1e-9
    assert molmo_ent["p"] > 0.05
    assert gemma_ent["p"] > 0.05
