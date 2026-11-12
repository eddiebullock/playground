"""
A1 descriptive: how frames were delivered, per trial, in the locked A1 files.

Written 2026-11-12 after the independent draft review (research/reviews/draft_review_2026-11-13.md,
finding 24). Reads the 30 rotated A1 eval files (the paths research/scripts/analyse_rotation.py reads) and
counts each trial's recorded `multi_frame_strategy` per model and condition, and lists every clip delivered
as a single first frame (`single_first_frame`), by the video actually shown (`video_path`). No decision
uses it. Standard library only. Writes research/reproduction/a1_frame_strategies.json with input hashes.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = "results/baseline/eu_emotions/{m}/eval_v2_eu_emotions_{m}_video_only_seed42_rot{r}_{c}.json"
MODELS = ("gemma4", "qwen3vl", "molmo2")
CONDITIONS = ("real", "mismatch")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, default=ROOT / "research/reproduction/a1_frame_strategies.json")
    args = ap.parse_args(argv)
    out = {"description": __doc__.strip().splitlines()[0], "models": {}, "single_first_frame": {},
           "inputs_sha256": {}, "script_sha256": sha256(__file__)}
    single = collections.Counter()
    for m in MODELS:
        out["models"][m] = {}
        for c in CONDITIONS:
            counts = collections.Counter()
            for r in range(5):
                path = args.root / EVAL.format(m=m, r=r, c=c)
                out["inputs_sha256"][str(path.relative_to(args.root))] = sha256(path)
                obj = json.loads(path.read_text(encoding="utf-8"))
                for t in obj.get("trials") or obj["results"]:
                    strategy = t.get("multi_frame_strategy")
                    counts[str(strategy)] += 1
                    if strategy == "single_first_frame":
                        video = str(t.get("video_path") or "")
                        key = video[video.find("emotions_"):] if "emotions_" in video else video
                        single[f"{m}|{c}|{key}"] += 1
            out["models"][m][c] = dict(counts)
    out["single_first_frame"] = dict(sorted(single.items()))
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["single_first_frame"], indent=1))


if __name__ == "__main__":
    main()
