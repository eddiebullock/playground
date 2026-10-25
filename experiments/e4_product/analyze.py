"""E4 analysis: does Gloss Reflect's full pipeline get the intended meaning?

  python analyze.py results/baseline_qwen3-8b.jsonl [more.jsonl ...]

Per file: top-1 match (top reading maps to the intended option), recall (any
reading maps to it), plain-reading-top rate on the sincere controls,
accuracy by clarity band, alignment of Gloss's spread with human
disagreement, and latency. Bootstrap 95% CIs over items.
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ITEMS = {i["id"]: i for i in json.load(open(HERE / "items.json"))}


def boot(xs, n=2000, seed=0):
    if not xs:
        return (float("nan"),) * 2
    r = random.Random(seed)
    ms = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def fmt(xs):
    if not xs:
        return "n=0"
    lo, hi = boot(xs)
    return f"{sum(xs) / len(xs):.3f} [{lo:.3f}, {hi:.3f}] n={len(xs)}"


def analyse(path):
    recs = [json.loads(l) for l in open(path)]
    hu = [r for r in recs if r["task"] != "Sincere"]
    sin = [r for r in recs if r["task"] == "Sincere"]
    top = lambda r: int(r["readings"][0]["label"] == r["gold"])
    anyr = lambda r: int(any(x["label"] == r["gold"] for x in r["readings"]))
    out = {"file": Path(path).name, "n": len(recs)}
    out["top1"] = fmt([top(r) for r in hu])
    out["recall"] = fmt([anyr(r) for r in hu])
    out["top1_given_recall"] = fmt([top(r) for r in hu if anyr(r)])
    out["sincere_plain_top"] = fmt([top(r) for r in sin])
    out["sincere_recall"] = fmt([anyr(r) for r in sin])
    by = defaultdict(list)
    for r in hu:
        by[r["task"]].append(top(r))
    out["top1_by_task"] = {t: f"{sum(v) / len(v):.2f} (n={len(v)})" for t, v in sorted(by.items())}
    band = defaultdict(list)
    for r in recs:
        band[r["clarity"]].append(top(r))
    out["top1_by_clarity"] = {b: f"{sum(v) / len(v):.2f} (n={len(v)}, {len(v) / len(recs):.0%})" for b, v in band.items()}
    labs = defaultdict(int)
    for r in hu:
        if not top(r):
            labs[r["readings"][0]["label"]] += 1
    out["top_errors_by_label"] = dict(labs)
    pairs = [(r["spread"], 1 - ITEMS[r["id"]]["human_acc"]) for r in hu if ITEMS[r["id"]]["human_acc"] is not None]
    out["rho_spread_vs_human_disagreement"] = f"{spearman(*zip(*pairs)):.3f} (n={len(pairs)})"
    ms = sorted(r["ms"] for r in recs)
    out["latency_s_median_p90"] = f"{ms[len(ms) // 2] / 1000:.0f} / {ms[int(0.9 * len(ms))] / 1000:.0f}"
    gen = sorted(r["ms_gen"] for r in recs if r.get("ms_gen"))
    if gen:
        out["writing_s_median"] = f"{gen[len(gen) // 2] / 1000:.0f}"
    out["n_readings_mean"] = round(sum(len(r["readings"]) for r in recs) / len(recs), 2)
    out["boundary_shown"] = sum(1 for r in recs if r.get("boundary"))
    return out


if __name__ == "__main__":
    res = [analyse(p) for p in sys.argv[1:]]
    print(json.dumps(res, indent=1))
    json.dump(res, open(HERE / "results/summary.json", "w"), indent=1)
