"""Inversion and 2x5 block-scrambling, applied in code from upright originals."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def invert_image(image: Image.Image) -> Image.Image:
    """180-degree rotation. Does not write a copy to disk."""

    return image.rotate(180)


def scramble_blocks(image: Image.Image, rows: int = 2, cols: int = 5, seed: int = 0) -> Image.Image:
    """Deterministic 2x5 block scramble. Same seed => same permutation."""

    import random

    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    block_w = width // cols
    block_h = height // rows
    if block_w < 1 or block_h < 1:
        raise ValueError(f"image {width}x{height} too small for {rows}x{cols} scramble")
    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (c * block_w, r * block_h, (c + 1) * block_w, (r + 1) * block_h)
            tiles.append(rgb.crop(box))
    rng = random.Random(seed)
    order = list(range(len(tiles)))
    rng.shuffle(order)
    out = Image.new("RGB", (block_w * cols, block_h * rows))
    for i, src in enumerate(order):
        r, c = divmod(i, cols)
        out.paste(tiles[src], (c * block_w, r * block_h))
    return out


def apply_transform(image: Image.Image, name: str, *, seed: int = 0) -> Image.Image:
    if name in {"upright", "none", ""}:
        return image
    if name == "invert":
        return invert_image(image)
    if name == "scramble":
        return scramble_blocks(image, seed=seed)
    raise ValueError(f"unknown transform {name!r}")
