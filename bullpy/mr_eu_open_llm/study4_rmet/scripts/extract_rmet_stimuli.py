"""
Extract eyes-only RMET stimuli from the adult PDF booklets.

The PDFs place four word options around each photograph. This script:
  1) renders each PDF page to a full-bleed PNG (via the Swift PDFKit helper, or
     optional pymupdf if available),
  2) detects the central grayscale photograph (largest dark contiguous region /
     centered dark rectangle),
  3) writes ONLY the cropped eye image — no item numbers, no 4AFC labels.

Outputs (study4_rmet/data/rmet/):
  stimuli/item_XX.png          eyes-only crops (model input; no label leakage)
  _debug/fullpage/item_XX.png  optional full page renders for QC
  stimuli/manifest.json        paths + labels from answer key (labels NOT in image)

Does not modify or import study3 code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

STUDY4_ROOT = Path(__file__).resolve().parents[1]
SWIFT_RENDERER = STUDY4_ROOT / "scripts" / "render_rmet_pdf_pages.swift"
DEFAULT_ANSWER_KEY = STUDY4_ROOT / "data" / "rmet" / "answer_key" / "rmet_adult_answer_key.json"
DEFAULT_STIM_DIR = STUDY4_ROOT / "data" / "rmet" / "stimuli"
DEFAULT_DEBUG_DIR = STUDY4_ROOT / "data" / "rmet" / "_debug" / "fullpage"


def render_pdf_pages(pdf: Path, out_dir: Path, dpi: float = 150.0) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear prior pages for this PDF stem to avoid stale mixes
    for old in out_dir.glob("page_*.png"):
        old.unlink()
    cmd = ["swift", str(SWIFT_RENDERER), str(pdf), str(out_dir), str(int(dpi))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"PDF render failed for {pdf}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    pages = sorted(out_dir.glob("page_*.png"))
    if not pages:
        raise RuntimeError(f"No pages rendered from {pdf}")
    return pages


def _find_photo_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    """
    Locate the central eye photograph on a white page with corner labels.

    Strategy: threshold near-white background; take the largest non-white
    connected component that is horizontally centered and wider than tall
    (eye-region photos are landscape rectangles).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    # Near-white background
    white = (rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245)
    ink = ~white

    # Ignore top banner (item number / "practice") — top 8% of page
    ink[: int(0.08 * h), :] = False
    # Ignore a thin outer margin (labels sit near corners but photo is central)
    # Keep full width search; labels are sparse text and form small components.

    # Connected components via simple flood fill on a downsampled mask for speed
    scale = 4
    small = ink[::scale, ::scale]
    sh, sw = small.shape
    visited = np.zeros_like(small, dtype=bool)
    best = None  # (area, y0, x0, y1, x1) in full-res coords

    def neighbors(y: int, x: int):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < sh and 0 <= nx < sw:
                yield ny, nx

    for y in range(sh):
        for x in range(sw):
            if not small[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            ys: List[int] = []
            xs: List[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in neighbors(cy, cx):
                    if small[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            y0, y1 = min(ys), max(ys) + 1
            x0, x1 = min(xs), max(xs) + 1
            area = (y1 - y0) * (x1 - x0)
            # Reject tiny text blobs
            if area < (sh * sw) * 0.01:
                continue
            bh, bw = (y1 - y0), (x1 - x0)
            aspect = bw / max(bh, 1)
            if aspect < 1.2:  # eye photos are wider than tall
                continue
            cy = 0.5 * (y0 + y1) / sh
            cx = 0.5 * (x0 + x1) / sw
            # Prefer centrally located blobs
            if not (0.25 < cy < 0.75 and 0.2 < cx < 0.8):
                continue
            cand = (area, y0 * scale, x0 * scale, y1 * scale, x1 * scale)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        # Fallback: centered crop heuristic (middle 55% width × 35% height)
        x0 = int(0.225 * w)
        x1 = int(0.775 * w)
        y0 = int(0.30 * h)
        y1 = int(0.65 * h)
        return x0, y0, x1, y1

    _, y0, x0, y1, x1 = best
    # Small pad inside the photo border only (still well clear of corner labels)
    pad = max(2, int(0.01 * min(h, w)))
    x0 = max(0, x0 + pad)
    y0 = max(0, y0 + pad)
    x1 = min(w, x1 - pad)
    y1 = min(h, y1 - pad)
    return x0, y0, x1, y1


def crop_eyes(page_png: Path) -> Image.Image:
    img = Image.open(page_png).convert("RGB")
    x0, y0, x1, y1 = _find_photo_bbox(img)
    crop = img.crop((x0, y0, x1, y1))
    return crop


def extract_from_pdfs(
    part1: Path,
    part2: Path,
    *,
    stim_dir: Path,
    debug_dir: Optional[Path],
    answer_key: Path,
    dpi: float = 150.0,
    keep_fullpage: bool = True,
) -> dict:
    stim_dir.mkdir(parents=True, exist_ok=True)
    key = json.loads(answer_key.read_text(encoding="utf-8"))
    items = key["items"]
    assert len(items) == 36

    tmp_root = STUDY4_ROOT / "data" / "rmet" / "_debug" / "render_pages"
    p1_dir = tmp_root / "part1"
    p2_dir = tmp_root / "part2"
    pages1 = render_pdf_pages(part1, p1_dir, dpi=dpi)
    pages2 = render_pdf_pages(part2, p2_dir, dpi=dpi)

    # Adult booklet: part1 = practice + items 1..N1; part2 = remaining items.
    # Drop practice (first page of part1) if it is labeled practice / has no number.
    # Empirically ARC adult_part1 starts with practice then items 1..
    all_item_pages: List[Path] = []
    if len(pages1) + len(pages2) == 37:
        # practice + 36
        all_item_pages = pages1[1:] + pages2
    elif len(pages1) + len(pages2) == 36:
        all_item_pages = pages1 + pages2
    else:
        # Heuristic: if first page of part1 looks like practice (we'll still skip if 37 total fails)
        # Prefer: use all pages except the first of part1 when total == 37-ish
        combined = pages1 + pages2
        if len(combined) > 36:
            all_item_pages = combined[len(combined) - 36 :]
        else:
            raise RuntimeError(
                f"Unexpected page counts part1={len(pages1)} part2={len(pages2)}; expected 36 or 37 total"
            )

    if len(all_item_pages) != 36:
        raise RuntimeError(f"Expected 36 item pages, got {len(all_item_pages)}")

    if debug_dir is not None and keep_fullpage:
        debug_dir.mkdir(parents=True, exist_ok=True)

    manifest_trials = []
    for idx, page in enumerate(all_item_pages):
        item = items[idx]
        item_n = int(item["item"])
        crop = crop_eyes(page)
        out_path = stim_dir / f"item_{item_n:02d}.png"
        crop.save(out_path)
        if debug_dir is not None and keep_fullpage:
            # Copy full page for QC (contains labels — NEVER use as model input)
            dbg = debug_dir / f"item_{item_n:02d}_fullpage.png"
            Image.open(page).convert("RGB").save(dbg)

        # Leakage QC: OCR not available; approximate by checking that crop is much
        # smaller than full page and not nearly full-bleed.
        full = Image.open(page)
        fw, fh = full.size
        cw, ch = crop.size
        area_frac = (cw * ch) / float(fw * fh)
        if area_frac > 0.55:
            raise RuntimeError(
                f"Crop for item {item_n} covers {area_frac:.0%} of page — possible label leakage. Abort."
            )

        manifest_trials.append(
            {
                "trial_id": f"rmet_{item_n:02d}",
                "item": item_n,
                "image": str(out_path.relative_to(STUDY4_ROOT)),
                "options": item["options"],
                "correct_option": item["correct_option"],
                "correct_label": item["correct_label"],
                "crop_area_fraction_of_page": round(area_frac, 4),
            }
        )

    manifest = {
        "dataset": "rmet_adult",
        "n_items": 36,
        "stimuli_are_eyes_only": True,
        "label_leakage_guard": "central photo crop; area_frac must be <= 0.55",
        "answer_key": str(answer_key.relative_to(STUDY4_ROOT)),
        "trials": manifest_trials,
    }
    man_path = stim_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part1", type=Path, required=True, help="adult_part1.pdf")
    ap.add_argument("--part2", type=Path, required=True, help="adult_part2.pdf")
    ap.add_argument("--answer_key", type=Path, default=DEFAULT_ANSWER_KEY)
    ap.add_argument("--stim_dir", type=Path, default=DEFAULT_STIM_DIR)
    ap.add_argument("--debug_dir", type=Path, default=DEFAULT_DEBUG_DIR)
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--no_fullpage_debug", action="store_true")
    args = ap.parse_args(argv)

    man = extract_from_pdfs(
        args.part1,
        args.part2,
        stim_dir=args.stim_dir,
        debug_dir=None if args.no_fullpage_debug else args.debug_dir,
        answer_key=args.answer_key,
        dpi=args.dpi,
        keep_fullpage=not args.no_fullpage_debug,
    )
    print(f"wrote {len(man['trials'])} eyes-only stimuli -> {args.stim_dir}")
    print(f"manifest -> {args.stim_dir / 'manifest.json'}")
    fracs = [t["crop_area_fraction_of_page"] for t in man["trials"]]
    print(f"crop area frac min/median/max: {min(fracs):.3f} / {sorted(fracs)[len(fracs)//2]:.3f} / {max(fracs):.3f}")
    return 0


if __name__ == "__main__":
    # Pillow/numpy are required; keep imports local to study4 tooling.
    try:
        import numpy  # noqa: F401
        from PIL import Image as _Image  # noqa: F401
    except ImportError:
        print("Need pillow + numpy in the active Python (study4 local tooling).", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(main())
