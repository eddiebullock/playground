from __future__ import annotations

from pathlib import Path

from esvlm.models.registry import disk_names_for, get_spec, resolve_local_dir


def test_disk_names_include_cluster_aliases() -> None:
    eight = get_spec("qwen3vl_8b")
    names = disk_names_for(eight)
    assert "qwen3vl_8b" in names
    assert "qwen3vl" in names
    assert disk_names_for(get_spec("qwen3vl")) == names
    gemma = disk_names_for(get_spec("gemma4_e4b"))
    assert "gemma4" in gemma
    assert "gemma4_e4b" in gemma


def test_resolve_local_dir_uses_cluster_alias(tmp_path: Path) -> None:
    (tmp_path / "qwen3vl").mkdir()
    (tmp_path / "qwen3vl" / "config.json").write_text("{}", encoding="utf-8")
    found = resolve_local_dir(get_spec("qwen3vl_8b"), tmp_path)
    assert found is not None
    assert found.name == "qwen3vl"
    assert resolve_local_dir(get_spec("qwen3vl_2b"), tmp_path) is None
