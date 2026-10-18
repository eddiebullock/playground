from __future__ import annotations

from PIL import Image

from esvlm.tasks.transforms import invert_image, scramble_blocks


def test_invert_is_180() -> None:
    img = Image.new("RGB", (20, 10), (0, 0, 0))
    img.putpixel((0, 0), (255, 0, 0))
    out = invert_image(img)
    assert out.size == img.size
    assert out.getpixel((19, 9)) == (255, 0, 0)
    assert out.getpixel((0, 0)) == (0, 0, 0)


def test_scramble_is_deterministic_and_permutes() -> None:
    img = Image.new("RGB", (50, 20), (10, 10, 10))
    # distinct colours per 2x5 block
    colors = [
        (10 * i, 20, 30)
        for i in range(10)
    ]
    w, h = 10, 10
    idx = 0
    for r in range(2):
        for c in range(5):
            for x in range(c * w, (c + 1) * w):
                for y in range(r * h, (r + 1) * h):
                    img.putpixel((x, y), colors[idx])
            idx += 1
    a = scramble_blocks(img, seed=7)
    b = scramble_blocks(img, seed=7)
    c = scramble_blocks(img, seed=8)
    assert list(a.getdata()) == list(b.getdata())
    assert list(a.getdata()) != list(img.getdata())
    assert list(a.getdata()) != list(c.getdata())
