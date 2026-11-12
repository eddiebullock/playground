"""
A1 descriptive: share of cells whose answer is "None of the above".

Written 2026-11-12 after the independent draft review (research/reviews/draft_review_2026-11-13.md,
finding 5), on the locked A1 files. Reads the 30 rotated A1 eval files (the paths
research/scripts/analyse_rotation.py reads) and, per model and condition (real, mismatch), counts the
trials whose recorded answer, `stage2.prediction`, is "None of the above". This is the A1 counterpart of
research/scripts/a3_nota_shares.py. It is outside the frozen A1 summary and no decision uses it.

Denominator: every trial in the five rotation files of that model and condition. Trials with an error
(`error` not null, or no stage2) are counted separately and stay in the denominator, as the frozen A1
scoring keeps them (scored wrong, DECISIONS 2026-11-08). Standard library only. Writes
research/reproduction/a1_nota_shares.json with the sha256 of every input file.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = "results/baseline/eu_emotions/{m}/eval_v2_eu_emotions_{m}_video_only_seed42_rot{r}_{c}.json"
MODELS = ("gemma4", "qwen3vl", "molmo2")
CONDITIONS = ("real", "mismatch")
ROTATIONS = range(5)
NOTA = "None of the above"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count(paths):
    n = nota = errored = 0
    for path in paths:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        for trial in obj.get("trials") or obj["results"]:
            n += 1
            stage2 = trial.get("stage2")
            if trial.get("error") is not None or not stage2:
                errored += 1
                continue
            nota += stage2.get("prediction") == NOTA
    return {"n_trials": n, "n_none_of_the_above": nota, "n_errored": errored,
            "share_none_of_the_above": round(nota / n, 4) if n else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, default=ROOT / "research/reproduction/a1_nota_shares.json")
    args = ap.parse_args(argv)
    out = {"description": __doc__.strip().splitlines()[0], "answer_field": "stage2.prediction", "models": {},
           "inputs_sha256": {}, "script_sha256": sha256(__file__)}
    for m in MODELS:
        out["models"][m] = {}
        for c in CONDITIONS:
            paths = [args.root / EVAL.format(m=m, r=r, c=c) for r in ROTATIONS]
            for p in paths:
                out["inputs_sha256"][str(p.relative_to(args.root))] = sha256(p)
            out["models"][m][c] = count(paths)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for m, block in out["models"].items():
        print(m, {c: (v["n_none_of_the_above"], v["n_trials"], v["share_none_of_the_above"]) for c, v in block.items()})


if __name__ == "__main__":
    main()
