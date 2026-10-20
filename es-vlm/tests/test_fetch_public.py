from __future__ import annotations

from pathlib import Path

import pytest

from esvlm.cli.fetch_public import main
from esvlm.tasks.prompts import template_ids


def test_public_templates_exist() -> None:
    assert template_ids("nextqa") == ["direct", "video", "numbered"]
    assert template_ids("mmtom") == ["direct", "belief", "numbered"]


def test_fetch_public_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_nextqa(root: Path, log) -> Path:
        seen.append(f"nextqa:{root}")
        dest = Path(root) / "nextqa" / "pairs_val.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}", encoding="utf-8")
        return dest

    def fake_mmtom(root: Path, log) -> Path:
        seen.append(f"mmtom:{root}")
        dest = Path(root) / "mmtom" / "manifest.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}", encoding="utf-8")
        return dest

    monkeypatch.setattr("esvlm.cli.fetch_public.fetch_nextqa_annotations", fake_nextqa)
    monkeypatch.setattr("esvlm.cli.fetch_public.fetch_mmtom_text", fake_mmtom)
    assert main(["--dataset", "all", "--root", str(tmp_path)]) == 0
    assert seen == [f"nextqa:{tmp_path}", f"mmtom:{tmp_path}"]
    seen.clear()
    assert main(["--dataset", "nextqa", "--root", str(tmp_path)]) == 0
    assert seen == [f"nextqa:{tmp_path}"]
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["--dataset", "videos", "--root", str(tmp_path)])


def test_fetch_public_requires_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ESVLM_PUBLIC_DIR", raising=False)
    with pytest.raises(SystemExit, match="ESVLM_PUBLIC_DIR"):
        main(["--dataset", "mmtom"])
