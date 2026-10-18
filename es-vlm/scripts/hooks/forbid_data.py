#!/usr/bin/env python3
"""Block licensed data, site files, media outside fixtures, and files >5 MB."""

from __future__ import annotations

import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_PREFIXES = ("data/", "es-vlm/data/")
BLOCKED_LOCAL = ("local/", "es-vlm/local/")
ALLOWED_LOCAL = {"local/site.example.sh", "es-vlm/local/site.example.sh"}
MEDIA_SUFFIX = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}
FIXTURE_MARK = "tests/fixtures/synthetic/"


def rel(path: str) -> str:
    return path.replace("\\", "/")


def check(path: str) -> str | None:
    p = rel(path)
    if p.startswith(BLOCKED_PREFIXES) or p == "data" or p.endswith("/data"):
        return f"{p}: do not commit data/"
    if p.startswith(BLOCKED_LOCAL) and p not in ALLOWED_LOCAL:
        return f"{p}: do not commit local/ (except site.example.sh)"
    suffix = Path(p).suffix.lower()
    if suffix in MEDIA_SUFFIX and FIXTURE_MARK not in p:
        return f"{p}: media / weights outside tests/fixtures/synthetic/"
    try:
        size = Path(path).stat().st_size
    except OSError:
        return None
    if size > MAX_BYTES:
        return f"{p}: {size} bytes exceeds 5 MB"
    return None


def main(argv: list[str]) -> int:
    errors = []
    for path in argv:
        err = check(path)
        if err:
            errors.append(err)
    if errors:
        sys.stderr.write("forbid_data:\n" + "\n".join(errors) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
