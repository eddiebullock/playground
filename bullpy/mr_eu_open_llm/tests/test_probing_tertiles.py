"""Tests for entropy tertile probing."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.probing import entropy_tertiles


def test_entropy_tertiles_split(tmp_path: Path) -> None:
    trials = []
    for i in range(9):
        trials.append(
            {
                "trial_id": f"t{i}",
                "label": "Happy",
                "stage1": {"semantic_entropy": float(i)},
            }
        )
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"trials": trials}), encoding="utf-8")
    low, high = entropy_tertiles(eval_path)
    assert len(low) == 3
    assert len(high) == 3
    assert "t0" in low
    assert "t8" in high
