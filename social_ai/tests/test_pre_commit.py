"""Unit tests for the social_ai pre-commit guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pre_commit_check", ROOT / "hooks" / "pre_commit_check.py"
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_deny_strings_reads_host_and_accounts_only():
    text = """
SITE_HOST="hpc.example.test"
SITE_USER="not-a-deny-value"
SITE_GPU_ACCOUNT="EXAMPLE-GPU"
SITE_CPU_ACCOUNT='EXAMPLE-CPU'
SITE_RDS_ROOT="${HOME}/rds"
"""
    assert mod.deny_strings_from_site(text) == [
        "hpc.example.test",
        "EXAMPLE-GPU",
        "EXAMPLE-CPU",
    ]


def test_refuses_data_suffixes_oversize_and_site_strings():
    deny = ["hpc.example.test"]
    assert mod.problems_for(
        "social_ai/data.csv",
        size=10,
        blob=b"a,b\n",
        deny_strings=deny,
    )
    assert mod.problems_for(
        "social_ai/weights.pt",
        size=10,
        blob=b"not-really",
        deny_strings=deny,
    )
    big = mod.problems_for(
        "social_ai/src/scog/big.py",
        size=5 * 1024 * 1024 + 1,
        blob=b"x = 1\n",
        deny_strings=deny,
    )
    assert any("5 MB" in err for err in big)
    leaked = mod.problems_for(
        "social_ai/jobs/run.sh",
        size=20,
        blob=b"ssh hpc.example.test\n",
        deny_strings=deny,
    )
    assert any("site identifier" in err for err in leaked)


def test_ignores_files_outside_social_ai_and_allows_clean_code():
    deny = ["hpc.example.test"]
    assert (
        mod.problems_for(
            "notes/data.csv",
            size=9_000_000,
            blob=b"hpc.example.test",
            deny_strings=deny,
        )
        == []
    )
    assert (
        mod.problems_for(
            "social_ai/src/scog/data/channel4.py",
            size=100,
            blob=b"def load():\n    return 1\n",
            deny_strings=deny,
        )
        == []
    )


def test_refuses_local_directory():
    errs = mod.problems_for(
        "social_ai/local/site.sh",
        size=10,
        blob=b"SITE_HOST=x\n",
        deny_strings=[],
    )
    assert errs and "local/" in errs[0]
