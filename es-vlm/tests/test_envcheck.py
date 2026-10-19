from __future__ import annotations

import pytest

from esvlm.utils.envcheck import check_job_env


def test_accepts_canonical_vars() -> None:
    check_job_env({"MODEL": "qwen3vl_2b", "MODE": "smoke"})


def test_rejects_lowercase_mode() -> None:
    with pytest.raises(SystemExit, match="mode="):
        check_job_env({"mode": "smoke"})
