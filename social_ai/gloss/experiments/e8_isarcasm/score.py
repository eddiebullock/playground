"""Score hand-labelled product runs (E4 or E8), including wide runs.

  python3 score.py <items.json> <run.jsonl> <labels.txt> [<run2.jsonl> <labels2.txt> ...]

Non-literal items (anything not Sincere): top-1 = the top *shown* reading is
labelled 1; recall shown = any shown reading is 1; recall written = any
written reading (shown or hidden) is 1. Sincere items: the top shown reading
is labelled 1 (the plain reading comes first). Accuracy by clarity band.
With two runs over the same items, a paired sign test on top-1. Bootstrap
95% CIs over items.
"""

import json
import math
import random
import sys
from collections import defaultdict


def ci(xs, n=2000, seed=0):
    if not xs:
        return [float("nan")] * 2
    r = random.Random(seed)
    ms = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [round(ms[int(0.025 * n)], 3), round(ms[int(0.975 * n)], 3)]


def fmt(xs):
    return f"{sum(xs) / len(xs):.3f} {ci(xs)} n={len(xs)}" if xs else "n=0"


def sign_p(a, b):
    """Two-sided exact sign test on discordant pairs."""
    k = min(a, b)
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def score(items_path, run, labels):
    items = {i["id"]: i for i in json.load(open(items_path))}
    lab = {k: [c == "1" for c in v] for k, v in (l.split() for l in open(labels) if l.strip())}
    per = {}
    for line in open(run):
        r = json.loads(line)
        if r["id"] not in lab:
            continue
        hs = lab[r["id"]]
        assert len(hs) == len(r["readings"]), r["id"]
        shown = [h for h, x in zip(hs, r["readings"]) if not x.get("hidden")]
        sincere = items[r["id"]]["task"] == "Sincere"
        per[r["id"]] = {"sincere": sincere, "top": int(shown[0]), "rshown": int(any(shown)),
                        "rwritten": int(any(hs)), "clarity": r["clarity"], "task": items[r["id"]]["task"]}
    nl = [v for v in per.values() if not v["sincere"]]
    si = [v for v in per.values() if v["sincere"]]
    band = defaultdict(list)
    for v in per.values():
        band[v["clarity"]].append(v["top"])
    by_task = defaultdict(list)
    for v in nl:
        by_task[v["task"]].append(v["top"])
    return per, {
        "run": run, "top1": fmt([v["top"] for v in nl]), "recall_shown": fmt([v["rshown"] for v in nl]),
        "recall_written": fmt([v["rwritten"] for v in nl]),
        "top1_given_written": fmt([v["top"] for v in nl if v["rwritten"]]),
        "sincere_plain_top": fmt([v["top"] for v in si]),
        "top1_by_task": {t: f"{sum(x) / len(x):.2f} (n={len(x)})" for t, x in sorted(by_task.items())},
        "correct_by_clarity": {b: f"{sum(x) / len(x):.2f} (n={len(x)}, {len(x) / len(per):.0%})" for b, x in band.items()},
    }


if __name__ == "__main__":
    items_path, *rest = sys.argv[1:]
    runs = [score(items_path, rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
    for _, s in runs:
        print(json.dumps(s, indent=1))
    if len(runs) == 2:
        a, b = runs[0][0], runs[1][0]
        common = sorted(set(a) & set(b))
        for kind, sel in (("non-literal", False), ("sincere", True)):
            ids = [i for i in common if a[i]["sincere"] == sel]
            up = sum(b[i]["top"] > a[i]["top"] for i in ids)
            down = sum(b[i]["top"] < a[i]["top"] for i in ids)
            print(f"paired top-1 ({kind}, n={len(ids)}): second better {up}, worse {down}, sign test p = {sign_p(up, down):.3f}")
