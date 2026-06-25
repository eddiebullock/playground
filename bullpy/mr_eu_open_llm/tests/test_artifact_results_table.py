"""Tests for artifact master table builder."""

import json
from pathlib import Path

from scripts.artifact_results_table import build_table, to_markdown
from scripts.augment_eval_artifact import augment_eval_json

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eval_v2_mini.json"


def test_build_table_from_augmented_fixture(tmp_path):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    aug = augment_eval_json(data)
    p = tmp_path / "eval_artifact_fixture_seed42.json"
    p.write_text(json.dumps(aug), encoding="utf-8")
    table = build_table([p], min_baseline_n=1)
    assert table["n_rows"] == 1
    row = table["sections"]["baselines"][0]
    assert row["strict_4afc_accuracy"] == 2 / 3
    assert row["free_response_judge_accuracy"] is not None
    md = to_markdown(table)
    assert "fixture" in md
    assert "6AFC" in md
