"""
Fair multi-frame input across architectures (protocol v2).

- Qwen2-VL, Qwen3-VL: native multi-image chat template.
- LLaVA-NeXT, Gemma4, Molmo2: composite frame grid when len(images) > 1
  so all models receive the same temporal information without silent 1-frame truncation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from PIL import Image

# Models whose processor accepts a list of frames directly (no composite grid).
NATIVE_MULTI_IMAGE_MODELS = {"qwen2vl", "qwen3vl"}


def make_frame_grid(images: List[Image.Image], max_cells: int = 8, cell_size: Tuple[int, int] = (336, 336)) -> Image.Image:
    """
    Tile up to max_cells frames into a single RGB image (row-major), 2x4 style when 8 frames.
    """
    if not images:
        raise ValueError("make_frame_grid requires at least one image")
    if len(images) == 1:
        return images[0].convert("RGB")

    n = min(len(images), max_cells)
    imgs = [im.convert("RGB").resize(cell_size, Image.Resampling.BILINEAR) for im in images[:n]]
    cols = 4 if n > 4 else max(2, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    w, h = cell_size
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * w, r * h))
    return canvas


def prepare_images_for_model(
    model_key: str,
    images: List[Image.Image],
    enforce_multi_frame: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Returns (images_for_processor, metadata).

    metadata keys: n_frames_input, n_frames_used, multi_frame_strategy
    """
    meta: Dict[str, Any] = {
        "n_frames_input": len(images),
        "n_frames_used": len(images),
        "multi_frame_strategy": "native",
    }
    if not images:
        raise ValueError("No frames provided")

    if not enforce_multi_frame or len(images) == 1:
        if model_key in NATIVE_MULTI_IMAGE_MODELS and len(images) > 1:
            meta["multi_frame_strategy"] = "native_list"
            return images, meta
        single = images[0]
        meta["n_frames_used"] = 1
        meta["multi_frame_strategy"] = "single_first_frame"
        return single, meta

    if model_key in NATIVE_MULTI_IMAGE_MODELS:
        meta["multi_frame_strategy"] = "native_list"
        return images, meta

    # Composite grid for single-image chat models
    grid = make_frame_grid(images, max_cells=min(8, len(images)))
    meta["n_frames_used"] = len(images)
    meta["multi_frame_strategy"] = "composite_grid"
    meta["grid_cells"] = min(8, len(images))
    return grid, meta
