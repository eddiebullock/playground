import json
from pathlib import Path

from config import PROTOCOL_VERSION
from scripts.study1_baselines import discover_canonical_baseline


def _write_eval(tmp_path: Path, model: str, *, n_scored: int, condition: str, entropy: float) -> Path:
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "protocol_version": PROTOCOL_VERSION,
        "model": model,
        "condition": condition,
        "n_scored": n_scored,
        "accuracy": 0.4,
        "mean_semantic_entropy": entropy,
        "trials": [],
    }
    path = model_dir / f"eval_v2_eu_emotions_{model}_{condition}_seed42.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_discover_prefers_full_trial_matching_condition(tmp_path: Path) -> None:
    _write_eval(tmp_path, "qwen2vl", n_scored=2, condition="video_only", entropy=2.5)
    full = _write_eval(tmp_path, "qwen2vl", n_scored=118, condition="video_only", entropy=2.8)
    _write_eval(tmp_path, "qwen2vl", n_scored=118, condition="multimodal", entropy=2.9)
    got = discover_canonical_baseline("qwen2vl", results_root=tmp_path)
    assert got == full
