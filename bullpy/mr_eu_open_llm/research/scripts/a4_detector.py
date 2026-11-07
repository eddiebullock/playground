#!/usr/bin/env python3
"""Amendment A4, gate V1: LibreFace on the one-AU renders. No vision-language model.

Renders the `one_au` and `neutral` rows in memory, one image at a time, and writes one CSV row
per column, identity and amplitude. Nothing is saved but the CSV. LibreFace is bilateral, so
L and R columns read the same output: the AU moved is known from the manifest, and laterality
is not tested. AUs with intensity use it. AUs with presence only use the presence score.
Eight families have no readout and are listed, not counted.

Usage, with the au_tools env, which has LibreFace and the renderer's dependencies:
    python research/scripts/a4_detector.py --manifest results/a4/manifest.csv \
        --weights venvs/au_tools/weights/libreface --out results/a4/detector_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT))

INTENSITY_AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
PRESENCE_AUS = (1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24)
COLUMNS = ("au", "identity", "seed", "amplitude", "readout", "intensity", "neutral", "face_found")


def readout_for(column: str) -> Optional[Tuple[str, int]]:
    """Which detector output reads a column, or None. AU06L and AU06R both read AU 6."""
    au = int(column[2:4])
    if au in INTENSITY_AUS:
        return "intensity", INTENSITY_AUS.index(au)
    if au in PRESENCE_AUS:
        return "presence", PRESENCE_AUS.index(au)
    return None


def main(argv=None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    import extract_au_measures as eam
    import render_au_faces as raf

    assert tuple(eam.LF_INTENSITY_AUS) == INTENSITY_AUS and tuple(eam.LF_OCCURRENCE_AUS) == PRESENCE_AUS
    manifest = pd.read_csv(args.manifest)
    libreface = eam.LibreFace(args.weights, args.device)

    def measure(row) -> Optional[Tuple[List[float], List[float]]]:
        image = raf.render_face(json.loads(row.au_json), row.identity, int(row.seed))
        aligned = libreface.align(image)
        if aligned is None:
            return None
        intensity, presence = libreface.predict([aligned[0]])
        return list(map(float, intensity[0])), list(map(float, presence[0]))

    neutral: Dict[Tuple[str, int], Optional[Tuple[List[float], List[float]]]] = {}
    for row in manifest[manifest.set == "neutral"].itertuples():
        neutral[(row.identity, int(row.seed))] = measure(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in manifest[manifest.set == "one_au"].itertuples():
            column = row.cfg
            mapped = readout_for(column)
            if mapped is None:
                continue
            kind, index = mapped
            amplitude = float(next(iter(json.loads(row.au_json).values())))
            result = measure(row)
            base = neutral[(row.identity, int(row.seed))]
            part = 0 if kind == "intensity" else 1
            writer.writerow([column, row.identity, int(row.seed), amplitude, kind,
                             "" if result is None else result[part][index],
                             "" if base is None else base[part][index], int(result is not None)])
            handle.flush()
    print("Done:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
