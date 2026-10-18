"""Load image / uniformly sampled video frames for scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esvlm.types import MCQItem


def uniform_indices(n_src: int, n_keep: int) -> list[int]:
    if n_src <= 0:
        raise ValueError("no source frames")
    if n_keep <= 0:
        raise ValueError("n_keep must be positive")
    if n_src == 1 or n_keep == 1:
        return [0]
    take = min(n_keep, n_src)
    raw = np.linspace(0, n_src - 1, take)
    return [int(i) for i in np.round(raw)]


def load_media(item: MCQItem, *, frames: int = 8) -> list:
    """Return PIL images. Empty list for text-only items."""

    if item.media_type == "none" or not item.media:
        return []
    path = Path(item.media)
    if not path.is_file():
        raise FileNotFoundError(path)
    if item.media_type == "image":
        from PIL import Image

        return [Image.open(path).convert("RGB")]
    if item.media_type in {"video", "audio+video"}:
        return _video_frames(path, frames)
    raise ValueError(f"unsupported media_type {item.media_type}")


def _video_frames(path: Path, n_keep: int) -> list:
    from PIL import Image

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("video scoring needs the av package") from exc
    container = av.open(str(path))
    stream = container.streams.video[0]
    decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    container.close()
    if not decoded:
        raise ValueError(f"no frames in {path}")
    return [Image.fromarray(decoded[i]) for i in uniform_indices(len(decoded), n_keep)]
