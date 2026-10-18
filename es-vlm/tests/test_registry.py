from __future__ import annotations

from pathlib import Path

from esvlm.models.registry import get_spec, resolve_local_dir


def test_resolve_cluster_alias_dir(tmp_path: Path) -> None:
    weights = tmp_path / "qwen3vl"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    found = resolve_local_dir(get_spec("qwen3vl_8b"), tmp_path)
    assert found == weights


def test_resolve_canonical_key_dir(tmp_path: Path) -> None:
    weights = tmp_path / "qwen3vl_8b"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    found = resolve_local_dir(get_spec("qwen3vl_8b"), tmp_path)
    assert found == weights


def test_resolve_new_models_dir_env(tmp_path: Path, monkeypatch) -> None:
    new_root = tmp_path / "new"
    weights = new_root / "qwen3vl_2b"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ESVLM_NEW_MODELS_DIR", str(new_root))
    found = resolve_local_dir(get_spec("qwen3vl_2b"), tmp_path / "missing")
    assert found == weights
