"""Tests for study3_summarize."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.study3_summarize import build_markdown, summarize_model


def test_summarize_model_missing_files(tmp_path: Path) -> None:
    m = summarize_model("qwen2vl", tmp_path)
    assert m["model"] == "qwen2vl"
    assert m["modality"] == "video_only"
    assert isinstance(m["missing"], list)


def test_build_markdown_empty_probes(tmp_path: Path) -> None:
    models = [
        {
            "model": "gemma4",
            "modality": "multimodal",
            "baseline_4afc_acc": 0.4,
            "finetuned_4afc_acc": 0.05,
            "delta_pp": -35.0,
            "probes": None,
            "rsa": None,
            "patching": None,
        }
    ]
    md = build_markdown(models)
    assert "gemma4" in md
    assert "Study 3" in md
