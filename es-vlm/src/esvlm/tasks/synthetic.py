"""Synthetic PIL fixtures for Mac smoke tests. No licensed faces."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from esvlm.types import MCQItem

COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (200, 40, 40)),
    ("green", (40, 160, 70)),
    ("blue", (40, 80, 200)),
    ("yellow", (220, 200, 40)),
    ("purple", (140, 50, 180)),
    ("orange", (230, 120, 30)),
]

SHAPES = ("circle", "square", "triangle")


def write_fixtures(root: Path, *, size: int = 128, n_color: int = 6, n_shape: int = 6) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (name, rgb) in enumerate(COLORS[:n_color]):
        img = Image.new("RGB", (size, size), rgb)
        path = root / f"color_{i:02d}_{name}.png"
        img.save(path)
        paths.append(path)
    for i in range(n_shape):
        shape = SHAPES[i % len(SHAPES)]
        bg = (240, 240, 240)
        fg = COLORS[i % len(COLORS)][1]
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)
        m = size // 6
        if shape == "circle":
            draw.ellipse((m, m, size - m, size - m), fill=fg)
        elif shape == "square":
            draw.rectangle((m, m, size - m, size - m), fill=fg)
        else:
            draw.polygon(
                [(size // 2, m), (size - m, size - m), (m, size - m)],
                fill=fg,
            )
        path = root / f"shape_{i:02d}_{shape}.png"
        img.save(path)
        paths.append(path)
    return paths


def items_from_fixtures(root: Path) -> list[MCQItem]:
    items: list[MCQItem] = []
    color_names = [c[0] for c in COLORS]
    for path in sorted(root.glob("color_*.png")):
        name = path.stem.split("_")[-1]
        options = tuple(color_names)
        items.append(
            MCQItem(
                item_id=path.stem,
                task="synthetic",
                domain="control",
                media=str(path),
                media_type="image",
                prompt_text="",
                options=options,
                correct_idx=options.index(name),
                meta={"kind": "color"},
            )
        )
    for path in sorted(root.glob("shape_*.png")):
        name = path.stem.split("_")[-1]
        options = SHAPES
        items.append(
            MCQItem(
                item_id=path.stem,
                task="synthetic",
                domain="control",
                media=str(path),
                media_type="image",
                prompt_text="",
                options=options,
                correct_idx=options.index(name),
                meta={"kind": "shape"},
            )
        )
    return items
