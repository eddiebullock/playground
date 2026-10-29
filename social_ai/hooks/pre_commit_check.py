#!/usr/bin/env python3
"""Refuse unsafe commits of files under social_ai/.

Does not print the site identifiers it is looking for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BAD_SUFFIXES = {
    ".csv",
    ".xlsx",
    ".parquet",
    ".pkl",
    ".npy",
    ".safetensors",
    ".pt",
}
# Host and account names only. The laptop path in AGENTS.md contains the
# CRSid, which the hook rules do not treat as a cluster account name.
DENY_KEYS = ("SITE_HOST", "SITE_GPU_ACCOUNT", "SITE_CPU_ACCOUNT")


def repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    )
    return Path(out.strip())


def staged_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        text=True,
    )
    return [line for line in out.splitlines() if line.strip()]


def deny_strings_from_site(site_text: str) -> list[str]:
    """Pull literal host, user and account names out of site.sh."""
    found: list[str] = []
    for raw in site_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key not in DENY_KEYS:
            continue
        value = value.strip().strip('"').strip("'")
        if not value or "$" in value:
            continue
        found.append(value)
    return found


def staged_blob(path: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "show", f":{path}"])
    except subprocess.CalledProcessError:
        return None


def staged_size(path: str) -> int | None:
    try:
        out = subprocess.check_output(["git", "cat-file", "-s", f":{path}"], text=True)
    except subprocess.CalledProcessError:
        return None
    return int(out.strip())


def problems_for(
    path: str,
    *,
    size: int | None,
    blob: bytes | None,
    deny_strings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not path.startswith("social_ai/"):
        return errors
    if path.startswith("social_ai/local/"):
        errors.append(f"{path}: local/ must not be committed")
        return errors
    suffix = Path(path).suffix.lower()
    if suffix in BAD_SUFFIXES:
        errors.append(f"{path}: data file type {suffix} is not allowed")
    if size is not None and size > MAX_BYTES:
        errors.append(f"{path}: {size} bytes exceeds 5 MB")
    if blob is None or not deny_strings:
        return errors
    if b"\0" in blob[:1024]:
        return errors
    text = blob.decode("utf-8", errors="ignore")
    for secret in deny_strings:
        if secret and secret in text:
            errors.append(
                f"{path}: contains a site identifier listed in local/site.sh"
            )
            break
    return errors


def main() -> int:
    root = repo_root()
    files = [path for path in staged_files() if path.startswith("social_ai/")]
    if not files:
        return 0
    site = root / "social_ai" / "local" / "site.sh"
    deny_strings = (
        deny_strings_from_site(site.read_text(encoding="utf-8"))
        if site.is_file()
        else []
    )
    errors: list[str] = []
    for path in files:
        errors.extend(
            problems_for(
                path,
                size=staged_size(path),
                blob=staged_blob(path),
                deny_strings=deny_strings,
            )
        )
    if errors:
        print("pre-commit refused this commit:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
