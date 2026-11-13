"""
Snoek et al.: how far the basic and conversational experiments share AU configurations.

Written 2026-11-12 after the independent draft review (research/reviews/draft_review_2026-11-13.md,
finding 7). Reads the public archive with the frozen loader (`research/scripts/human_au_effects.py`,
`load_zip`, no extraction) and counts, per dataset, the unique AU configurations (each row's values on
the 33 AU columns, rounded to 6 decimals; presence coding is reported too), and their overlap within and
across cultures. Descriptive: no frozen rule uses it. It explains why the cue spectra of basic EA and
conversational EA are identical, and it bounds how separate A4's basic and conversational claims can be.

Writes research/reproduction/snoek_config_overlap.json with the archive's sha256.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_au_effects as hae  # noqa: E402

PAIRS = (("basic_EA", "conversational_EA"), ("basic_WC", "conversational_WC"), ("basic_WC", "basic_EA"),
         ("conversational_WC", "conversational_EA"))


def configurations(frame, presence=False):
    X = frame[hae.AU_COLS].to_numpy(dtype=float)
    X = (X > 0).astype(int) if presence else np.round(X, 6)
    return {tuple(r) for r in X.tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--zip", type=Path, default=ROOT / "data/external/snoek_au/au_emotion_data.zip")
    ap.add_argument("--output", type=Path, default=ROOT / "research/reproduction/snoek_config_overlap.json")
    args = ap.parse_args(argv)
    frames, _ = hae.load_zip(args.zip)
    out = {"description": __doc__.strip().splitlines()[0],
           "archive_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "coding": {}}
    for coding, presence in (("values", False), ("presence", True)):
        sets = {k: configurations(f, presence) for k, (f, _) in frames.items()}
        block = {"unique_configurations": {k: len(v) for k, v in sets.items()}, "pairs": {}}
        for a, b in PAIRS:
            A, B = sets[a], sets[b]
            block["pairs"][f"{a}|{b}"] = {"overlap": len(A & B), "identical": A == B,
                                         "second_within_first": B <= A, "first_within_second": A <= B}
        out["coding"][coding] = block
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["coding"]["values"], indent=1))


if __name__ == "__main__":
    main()
