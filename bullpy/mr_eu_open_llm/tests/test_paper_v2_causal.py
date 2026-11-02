"""Ablation trial-summary unit tests (no model forward)."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_v2.causal import summarise_path, summarise_trials  # noqa: E402


def _row(tid, axis, pred, correct, confused="False", label="Unfriendly"):
    return {
        "trial_id": tid,
        "axis": axis,
        "prediction": pred,
        "correct": correct,
        "confused_with_top_foil": confused,
        "correct_label": label,
    }


def test_flip_count_and_delta_acc():
    rows = [
        _row("t1", "baseline", "Unfriendly", "True"),
        _row("t1", "entropy", "Bored", "False"),
        _row("t1", "random", "Unfriendly", "True"),
        _row("t2", "baseline", "Bored", "False"),
        _row("t2", "entropy", "Bored", "False"),
        _row("t2", "random", "Bored", "False"),
    ]
    out = summarise_trials(rows)
    assert out["n_trials"] == 2
    assert out["n_flips"] == 1
    assert out["n_axis_trials"] == 4
    assert out["axes"]["entropy"]["n_flips"] == 1
    assert out["axes"]["entropy"]["mean_delta_accuracy"] == -0.5
    assert out["axes"]["random"]["n_flips"] == 0
    assert out["axes"]["random"]["mean_delta_accuracy"] == 0.0


def test_pilot_n36_five_flips():
    path = ROOT / "results/mech/pilot_n36/ablate_trials_qwen3vl_layer4.csv"
    if not path.is_file():
        return
    out = summarise_path(path)
    assert out["n_trials"] == 36
    assert out["n_axis_trials"] == 180
    assert out["n_flips"] == 5
    assert out["flips_by_axis"]["entropy"] == 0
    assert out["flips_by_axis"]["random"] == 1
    assert out["flips_by_axis"]["pair_bored_unfriendly"] == 1
    assert out["flips_by_axis"]["pair_disappointed_worried"] == 1
    assert out["flips_by_axis"]["pair_interested_kind"] == 2
    assert out["nonselective"] is True


def test_roundtrip_json():
    rows = [
        _row("t1", "baseline", "A", "True"),
        _row("t1", "entropy", "A", "True"),
    ]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        csv_path = d / "trials.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        outp = d / "sum.json"
        payload = summarise_path(csv_path, outp)
        assert outp.is_file()
        assert payload["n_flips"] == 0
